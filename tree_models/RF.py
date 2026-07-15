import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import ParameterGrid, GroupKFold, PredefinedSplit
from sklearn.metrics import fbeta_score, recall_score, log_loss
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 0. DATA LOADING AND SETUP
# ==========================================
print("Loading data...")
df = pd.read_csv("../multimodal_sports_injury_dataset.csv")
X = df.drop(["injury_occurred", "athlete_id", "session_id"], axis=1)
y = df["injury_occurred"]
groups = df["athlete_id"]

gkf = GroupKFold(n_splits=5)

# Setup output directories.
OUTPUT_DIR = Path("./preProcessingRes/rf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CALIB_DIR = OUTPUT_DIR / "calibration_results"
CALIB_DIR.mkdir(parents=True, exist_ok=True)

preprocessing_results_path = OUTPUT_DIR / "rf_best_preprocessing.txt"
rf_results_path = OUTPUT_DIR / "rf_best_parameters.txt"
calib_metrics_path = CALIB_DIR / "rf_calibration_metrics.txt"

# ==========================================
# PHASE 1: PREPROCESSING SCREENING
# ==========================================
missing_options = {
    "KNN": KNNImputer(n_neighbors=5),
    "MEDIAN": SimpleImputer(strategy="median"),
}

normalization_options = {
    "STD": StandardScaler(),
    "MinMax": MinMaxScaler(),
}

balancing_options = ["class_weight", "SMOTE"]

preprocessing_grid = list(
    product(missing_options.items(), normalization_options.items(), balancing_options)
)

rf_param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [10, 20, None],
    "min_samples_leaf": [1, 5],
    "max_features": ["sqrt", "log2"],
}

rf_grid_configs = list(ParameterGrid(rf_param_grid))

def build_preprocessor(
    missing_transformer, scaling_transformer, numeric_cols, categorical_cols
):
    # Build the fold-local preprocessing pipeline.
    num_pipe = Pipeline(
        [
            ("imputer", missing_transformer),
            ("scaler", scaling_transformer),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )

# ==========================================
# PHASE 1: COMPREHENSIVE PREPROCESSING SCREENING
# ==========================================
with open(preprocessing_results_path, "w", encoding="utf-8") as file:
    header = "Missing | Normalization | Balancing || Macro F2 | F2 C0 | F2 C1 | F2 C2 | Recall C0 | Recall C1 | Recall C2\n"
    file.write(header)
    file.write("-" * 110 + "\n")
    print("Starting preprocessing grid search for Random Forest...\n")

    best_f2_c2 = -np.inf
    best_config = None
    best_macro_f2 = None
    best_f2_c0 = None
    best_f2_c1 = None
    best_rec0 = None
    best_rec1 = None
    best_rec2 = None

    for (
        (missing_name, missing_transformer),
        (norm_name, norm_transformer),
        balancing_name,
    ) in preprocessing_grid:
        config_label = (
            f"missing:{missing_name}, norm:{norm_name}, balance:{balancing_name}"
        )

        macro_f2s, f2_c0s, f2_c1s, f2_c2s = [], [], [], []
        recall_c0s, recall_c1s, recall_c2s = [], [], []

        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

            preprocessor = build_preprocessor(
                missing_transformer, norm_transformer, num_cols, cat_cols
            )
            X_train_proc = preprocessor.fit_transform(X_train)
            X_test_proc = preprocessor.transform(X_test)

            # Apply the selected balancing strategy.
            if balancing_name == "SMOTE":
                sampler = SMOTE(random_state=42)
                X_train_proc, y_train = sampler.fit_resample(X_train_proc, y_train)
                class_weight = None
            else:
                class_weight = "balanced"

            # Train the Random Forest model for this configuration.
            model = RandomForestClassifier(
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1,
                n_estimators=300,
                max_depth=20,
                min_samples_leaf=1,
                max_features="sqrt",
                max_samples=None,
            )

            model.fit(X_train_proc, y_train)

            # Evaluate the held-out fold.
            y_pred = model.predict(X_test_proc)
            f2s = fbeta_score(
                y_test, y_pred, labels=[0, 1, 2], average=None, zero_division=0, beta=2
            )
            f2_c0s.append(f2s[0])
            f2_c1s.append(f2s[1])
            f2_c2s.append(f2s[2])
            macro_f2s.append(np.mean(f2s))

            recalls = recall_score(
                y_test, y_pred, labels=[0, 1, 2], average=None, zero_division=0
            )
            recall_c0s.append(recalls[0])
            recall_c1s.append(recalls[1])
            recall_c2s.append(recalls[2])

        avg_f2 = np.mean(macro_f2s)
        avg_f2_c0 = np.mean(f2_c0s)
        avg_f2_c1 = np.mean(f2_c1s)
        avg_f2_c2 = np.mean(f2_c2s)
        avg_rec0 = np.mean(recall_c0s)
        avg_rec1 = np.mean(recall_c1s)
        avg_rec2 = np.mean(recall_c2s)

        if avg_f2_c2 > best_f2_c2:
            best_f2_c2 = avg_f2_c2
            best_rec0 = avg_rec0
            best_rec1 = avg_rec1
            best_rec2 = avg_rec2
            best_macro_f2 = avg_f2
            best_f2_c0 = avg_f2_c0
            best_f2_c1 = avg_f2_c1
            best_config = {
                "missing": missing_name,
                "normalization": norm_name,
                "balancing": balancing_name,
            }

        res_str = (
            f"{config_label} || F2M: {avg_f2:.3f} | F20: {avg_f2_c0:.3f} | "
            f"F21: {avg_f2_c1:.3f} | F22: {avg_f2_c2:.3f} | "
            f"Rec0: {avg_rec0:.3f} | Rec1: {avg_rec1:.3f} | Rec2: {avg_rec2:.3f}\n"
        )
        file.write(res_str)
        print(res_str.strip())

    file.write("-" * 110 + "\n")
    file.write(
        f"BEST PREPROCESSING CONFIG (by F2 C2): {best_config} with F22: {best_f2_c2:.3f} "
        f"| F2M: {best_macro_f2:.3f} | F20: {best_f2_c0:.3f} | F21: {best_f2_c1:.3f}\n"
    )
    print(f"\nPreprocessing grid search completed! Best configuration: {best_config}")

def get_transformers_from_best_config(config):
    missing_transformer = missing_options[config["missing"]]
    norm_transformer = normalization_options[config["normalization"]]
    return missing_transformer, norm_transformer

# ==========================================
# PHASE 2: HYPERPARAMETER TUNING
# ==========================================
if best_config:
    best_missing_transformer, best_norm_transformer = get_transformers_from_best_config(
        best_config
    )

    with open(rf_results_path, "w", encoding="utf-8") as file:
        header = "RF Params || Macro F2 | F2 C0 | F2 C1 | F2 C2 | Recall C0 | Recall C1 | Recall C2\n"
        file.write(header)
        file.write("-" * 90 + "\n")
        print("\nStarting Random Forest grid search with best preprocessing...\n")

        best_rf_f2_c2 = -np.inf
        best_rf_config = None
        best_rf_macro_f2 = None
        best_rf_f2_c0 = None
        best_rf_f2_c1 = None

        for config in rf_grid_configs:
            macro_f2s, f2_c0s, f2_c1s, f2_c2s = [], [], [], []
            recall_c0s, recall_c1s, recall_c2s = [], [], []

            for train_idx, test_idx in gkf.split(X, y, groups=groups):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
                cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

                preprocessor = build_preprocessor(
                    best_missing_transformer, best_norm_transformer, num_cols, cat_cols
                )
                X_train_proc = preprocessor.fit_transform(X_train)
                X_test_proc = preprocessor.transform(X_test)

                # Apply the selected balancing strategy.
                if best_config["balancing"] == "SMOTE":
                    sampler = SMOTE(random_state=42)
                    X_train_proc, y_train = sampler.fit_resample(X_train_proc, y_train)
                    class_weight = None
                else:
                    class_weight = "balanced"

                # Train the tuned Random Forest configuration.
                model = RandomForestClassifier(
                    class_weight=class_weight,
                    random_state=42,
                    n_jobs=-1,
                    **config,
                )

                model.fit(X_train_proc, y_train)

                # Evaluate the held-out fold.
                y_pred = model.predict(X_test_proc)
                f2s = fbeta_score(
                    y_test, y_pred, labels=[0, 1, 2], average=None, zero_division=0, beta=2
                )
                f2_c0s.append(f2s[0])
                f2_c1s.append(f2s[1])
                f2_c2s.append(f2s[2])
                macro_f2s.append(np.mean(f2s))

                recalls = recall_score(
                    y_test, y_pred, labels=[0, 1, 2], average=None, zero_division=0
                )
                recall_c0s.append(recalls[0])
                recall_c1s.append(recalls[1])
                recall_c2s.append(recalls[2])

            avg_f2 = np.mean(macro_f2s)
            avg_f2_c0 = np.mean(f2_c0s)
            avg_f2_c1 = np.mean(f2_c1s)
            avg_f2_c2 = np.mean(f2_c2s)
            avg_rec0 = np.mean(recall_c0s)
            avg_rec1 = np.mean(recall_c1s)
            avg_rec2 = np.mean(recall_c2s)

            if avg_f2_c2 > best_rf_f2_c2:
                best_rf_f2_c2 = avg_f2_c2
                best_rf_config = config
                best_rf_macro_f2 = avg_f2
                best_rf_f2_c0 = avg_f2_c0
                best_rf_f2_c1 = avg_f2_c1

            config_str = (
                f"n:{config['n_estimators']}, d:{config['max_depth']}, "
                f"leaf:{config['min_samples_leaf']}, feat:{config['max_features']}"
            )
            res_str = (
                f"{config_str} || F2M: {avg_f2:.3f} | F20: {avg_f2_c0:.3f} | "
                f"F21: {avg_f2_c1:.3f} | F22: {avg_f2_c2:.3f} | "
                f"Rec0: {avg_rec0:.3f} | Rec1: {avg_rec1:.3f} | Rec2: {avg_rec2:.3f}\n"
            )
            file.write(res_str)
            print(res_str.strip())

        file.write("-" * 90 + "\n")
        file.write(
            f"BEST RF CONFIG (by F2 C2): {best_rf_config} with F22: {best_rf_f2_c2:.3f} "
            f"| F2M: {best_rf_macro_f2:.3f} | F20: {best_rf_f2_c0:.3f} | F21: {best_rf_f2_c1:.3f}\n"
        )
        print(
            f"\nRandom Forest grid search completed! Best configuration: {best_rf_config}"
        )

# ==========================================
# PHASE 3: FINAL MODEL CALIBRATION
# ==========================================
if best_rf_config is not None and best_config is not None:
    print("\n" + "="*60)
    print("PHASE 3 - FINAL MODEL CALIBRATION")
    print("="*60)

    with open(calib_metrics_path, "w", encoding="utf-8") as calib_file:
        calib_file.write(f"PHASE 3 - FINAL MODEL CALIBRATION (Method: Sigmoid / Platt Scaling)\n")
        calib_file.write("-" * 120 + "\n")
        
        # Extract a single isolated split from GroupKFold.
        train_idx, calib_idx = next(gkf.split(X, y, groups=groups))
        
        X_train_full, X_calib = X.iloc[train_idx], X.iloc[calib_idx]
        y_train_full, y_calib = y.iloc[train_idx], y.iloc[calib_idx]
        
        # Combine data for PredefinedSplit indexing.
        X_combined = pd.concat([X_train_full, X_calib], axis=0).reset_index(drop=True)
        y_combined = pd.concat([y_train_full, y_calib], axis=0).reset_index(drop=True)
        
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()

        # Build the end-to-end pipeline.
        steps = []
        
        preprocessor = build_preprocessor(
            best_missing_transformer, best_norm_transformer, num_cols, cat_cols
        )
        steps.append(('preprocessor', preprocessor))

        # Handle the balancing method based on the Phase 1 winner.
        rf_class_weight = "balanced" if best_config["balancing"] == "class_weight" else None
        
        if best_config["balancing"] == "SMOTE":
            steps.append(('smote', SMOTE(random_state=42)))

        base_model = RandomForestClassifier(
            class_weight=rf_class_weight,
            random_state=42,
            n_jobs=-1,
            **best_rf_config
        )
        steps.append(('model', base_model))

        pipeline = ImbPipeline(steps)

        # Train the uncalibrated pipeline for the pre-calibration metrics.
        print("Training the base pipeline on 80% of the data for pre-calibration testing...")
        pipeline.fit(X_train_full, y_train_full)
        y_calib_proba_pre = pipeline.predict_proba(X_calib)
        log_loss_pre = log_loss(y_calib, y_calib_proba_pre)

        # Setup the PredefinedSplit.
        test_fold = np.full(len(X_combined), -1)
        test_fold[len(X_train_full):] = 0 
        ps = PredefinedSplit(test_fold)

        # Fit calibration on the isolated validation fold.
        print("Calibrating on the isolated 20% using PredefinedSplit...")
        calibrated_model = CalibratedClassifierCV(
            estimator=pipeline, 
            method='sigmoid', 
            cv=ps
        )
        calibrated_model.fit(X_combined, y_combined)

        # Evaluate the calibrated probabilities.
        y_calib_proba_post = calibrated_model.predict_proba(X_calib)
        log_loss_post = log_loss(y_calib, y_calib_proba_post)

        res_str = (
            f"Base Configuration: {best_rf_config}\n"
            f"Base Preprocessing: {best_config}\n\n"
            f"Pre-Calibration Log Loss (on Isolated Fold):  {log_loss_pre:.4f}\n"
            f"Post-Calibration Log Loss (on Isolated Fold): {log_loss_post:.4f}\n"
            f"-> A lower Log Loss indicates that output probabilities are more aligned with clinical reality.\n"
        )

        calib_file.write(res_str)
        print(res_str)

        # Save the fitted artifacts.
        joblib.dump(pipeline, CALIB_DIR / "rf_final_base_pipeline.pkl")
        joblib.dump(calibrated_model, CALIB_DIR / "rf_final_calibrated_pipeline.pkl")
        print(f"Models successfully saved to:\n{CALIB_DIR}")

        # ==========================================
        # PHASE 4: CALIBRATION VISUALIZATION
        # ==========================================
        print("\n" + "="*60)
        print("PHASE 4 - CALIBRATION VISUALIZATION")
        print("="*60)
        print("Generating calibration curves for Class 2 (High Risk)...")

        target_class_idx = 2
        y_calib_binary = (y_calib == target_class_idx).astype(int)
        
        y_calib_proba_pre_c2 = y_calib_proba_pre[:, target_class_idx]
        y_calib_proba_post_c2 = y_calib_proba_post[:, target_class_idx]

        fig, ax = plt.subplots(figsize=(8, 8))

        CalibrationDisplay.from_predictions(
            y_calib_binary, 
            y_calib_proba_pre_c2, 
            n_bins=10, 
            name="Pre-Calibration (Base Pipeline)", 
            ax=ax, 
            color='red',
            linestyle='--'
        )

        CalibrationDisplay.from_predictions(
            y_calib_binary, 
            y_calib_proba_post_c2, 
            n_bins=10, 
            name="Post-Calibration (Sigmoid)", 
            ax=ax, 
            color='green'
        )

        ax.set_title("Calibration Curve (Reliability Diagram) - Class 2 (High Risk)")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives (True Frequency)")
        plt.grid(True, linestyle=':', alpha=0.7)

        plot_path = CALIB_DIR / "rf_calibration_curve_class2.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Calibration plot saved successfully to:\n{plot_path}")

print("\n" + "="*60)
print("PROCESS COMPLETED SUCCESSFULLY!")
print("Check the results in 'preProcessingRes/rf/'")
print("="*60)
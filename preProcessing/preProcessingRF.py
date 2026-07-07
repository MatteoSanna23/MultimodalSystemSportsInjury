import os
from itertools import product

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# 1. Data Loading
df = pd.read_csv("../multimodal_sports_injury_dataset.csv")
X = df.drop(["injury_occurred", "athlete_id", "session_id"], axis=1)
y = df["injury_occurred"]
groups = df["athlete_id"]


# 2. Preprocessing options to compare
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
gkf = GroupKFold(n_splits=5)


def build_preprocessor(
    missing_transformer, scaling_transformer, numeric_cols, categorical_cols
):
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


os.makedirs("./preProcessingRes/rf", exist_ok=True)

# 3. Grid search with subject-wise cross-validation on preprocessing only
with open("./preProcessingRes/rf/results_preprocessing_grid.txt", "w") as file:
    header = "Missing | Normalization | Balancing || Macro F1 | Recall C1 | Recall C2\n"
    file.write(header)
    file.write("-" * 110 + "\n")
    print("Starting preprocessing grid search for Random Forest...\n")

    best_f1 = -np.inf
    best_config = None
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

        macro_f1s, recall_c0s, recall_c1s, recall_c2s = [], [], [], []

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

            if balancing_name == "SMOTE":
                sampler = SMOTE(random_state=42)
                X_train_proc, y_train = sampler.fit_resample(X_train_proc, y_train)
                class_weight = None
            else:
                class_weight = "balanced"

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

            y_pred = model.predict(X_test_proc)
            macro_f1s.append(f1_score(y_test, y_pred, average="macro"))

            recalls = recall_score(
                y_test, y_pred, labels=[0, 1, 2], average=None, zero_division=0
            )
            recall_c0s.append(recalls[0])
            recall_c1s.append(recalls[1])
            recall_c2s.append(recalls[2])

        avg_f1 = np.mean(macro_f1s)
        avg_rec0 = np.mean(recall_c0s)
        avg_rec1 = np.mean(recall_c1s)
        avg_rec2 = np.mean(recall_c2s)

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_rec0 = avg_rec0
            best_rec1 = avg_rec1
            best_rec2 = avg_rec2
            best_config = {
                "missing": missing_name,
                "normalization": norm_name,
                "balancing": balancing_name,
            }

        res_str = f"{config_label} || F1: {avg_f1:.3f} | Rec0: {avg_rec0:.3f} | Rec1: {avg_rec1:.3f} | Rec2: {avg_rec2:.3f}\n"
        file.write(res_str)
        print(res_str.strip())

    file.write("-" * 110 + "\n")
    file.write(f"BEST PREPROCESSING CONFIG: {best_config} with F1: {best_f1:.3f}\n")
    print(f"\nPreprocessing grid search completed! Best configuration: {best_config}")

with open("./preProcessingRes/rf/best_preprocessing_config.txt", "w") as best_file:
    best_file.write(f"BEST PREPROCESSING CONFIG: {best_config}\n")
    best_file.write(f"BEST PREPROCESSING F1: {best_f1:.3f}\n")
    best_file.write(f"BEST PREPROCESSING Rec0: {best_rec0:.3f}\n")
    best_file.write(f"BEST PREPROCESSING Rec1: {best_rec1:.3f}\n")
    best_file.write(f"BEST PREPROCESSING Rec2: {best_rec2:.3f}\n")


def get_transformers_from_best_config(config):
    missing_transformer = missing_options[config["missing"]]
    norm_transformer = normalization_options[config["normalization"]]
    return missing_transformer, norm_transformer


best_missing_transformer, best_norm_transformer = get_transformers_from_best_config(
    best_config
)

# 4. Grid search for Random Forest using the best preprocessing configuration
with open("./preProcessingRes/rf/results_random_forest_grid.txt", "w") as file:
    header = "RF Params || Macro F1 | Recall C0 | Recall C1 | Recall C2\n"
    file.write(header)
    file.write("-" * 90 + "\n")
    print("Starting Random Forest grid search with best preprocessing...\n")

    best_rf_f1 = -np.inf
    best_rf_config = None

    for config in rf_grid_configs:
        macro_f1s, recall_c0s, recall_c1s, recall_c2s = [], [], [], []

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

            if best_config["balancing"] == "SMOTE":
                sampler = SMOTE(random_state=42)
                X_train_proc, y_train = sampler.fit_resample(X_train_proc, y_train)
                class_weight = None
            else:
                class_weight = "balanced"

            model = RandomForestClassifier(
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1,
                **config,
            )

            model.fit(X_train_proc, y_train)

            y_pred = model.predict(X_test_proc)
            macro_f1s.append(f1_score(y_test, y_pred, average="macro"))

            recalls = recall_score(
                y_test, y_pred, labels=[0, 1, 2], average=None, zero_division=0
            )
            recall_c0s.append(recalls[0])
            recall_c1s.append(recalls[1])
            recall_c2s.append(recalls[2])

        avg_f1 = np.mean(macro_f1s)
        avg_rec0 = np.mean(recall_c0s)
        avg_rec1 = np.mean(recall_c1s)
        avg_rec2 = np.mean(recall_c2s)

        if avg_f1 > best_rf_f1:
            best_rf_f1 = avg_f1
            best_rf_config = config

        config_str = (
            f"n:{config['n_estimators']}, d:{config['max_depth']}, "
            f"leaf:{config['min_samples_leaf']}, feat:{config['max_features']}"
        )
        res_str = f"{config_str} || F1: {avg_f1:.3f} | Rec0: {avg_rec0:.3f} | Rec1: {avg_rec1:.3f} | Rec2: {avg_rec2:.3f}\n"
        file.write(res_str)
        print(res_str.strip())

    file.write("-" * 90 + "\n")
    file.write(f"BEST RF CONFIG: {best_rf_config} with F1: {best_rf_f1:.3f}\n")
    print(
        f"\nRandom Forest grid search completed! Best configuration: {best_rf_config}"
    )

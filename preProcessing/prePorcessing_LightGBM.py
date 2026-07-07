import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, recall_score
from imblearn.over_sampling import SMOTE
import warnings
from lightgbm.basic import LightGBMError
from pathlib import Path

# Suppress verbose LightGBM logs to keep the terminal clean.
warnings.filterwarnings('ignore')

# 1. Data loading
df = pd.read_csv('multimodal_sports_injury_dataset.csv')
X = df.drop(['injury_occurred', 'athlete_id', 'session_id'], axis=1)
y = df['injury_occurred']
groups = df['athlete_id']

# 2. Preprocessing and balancing options
imputer_options = [
    ('KNN', KNNImputer(n_neighbors=5)),
    ('Median', SimpleImputer(strategy='median')),
    ('Native_LightGBM', 'passthrough')
]

scaler_options = [
    ('StdScaler', StandardScaler()),
    ('MinMaxScaler', MinMaxScaler())
]

balance_options = ['ClassWeight', 'SMOTE']

USE_GPU = True
FORCE_GPU = True
OUTPUT_DIR = Path(r"C:\Users\leozi\Desktop\uni\Magi\AI in Medicine\Multimodalproject\MultimodalSystemSportsInjury\preProcessing\preProcessingres\Light_GBM")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSING_RESULTS_FILE = OUTPUT_DIR / "lightgbm_best_preprocessing.txt"
PARAMETER_RESULTS_FILE = OUTPUT_DIR / "lightgbm_best_parameters.txt"

# 3. Parameters to test only after finding the best preprocessing combination
param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 300, 500],
}

grid_configs = list(ParameterGrid(param_grid))
gkf = GroupKFold(n_splits=5)


def build_model(use_gpu=True, **config):
    model_kwargs = dict(
        objective='multiclass',
        num_class=3,
        metric='multi_logloss',
        class_weight=None,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        **config,
    )

    if use_gpu:
        model_kwargs.update({
            'device_type': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'max_bin': 255,
        })

    return lgb.LGBMClassifier(**model_kwargs)


def build_preprocessor(num_strategy, scaler, cat_cols, num_cols):
    num_pipe = Pipeline([
        ('imputer', num_strategy),
        ('scaler', scaler),
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)],
        remainder='drop',
    )


def evaluate_combo(imp_obj, scl_obj, bal_name, config=None):
    macro_f1s, recall_c0s, recall_c1s, recall_c2s = [], [], [], []
    config = config or {}

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

        preprocessor = build_preprocessor(imp_obj, scl_obj, cat_cols, num_cols)

        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)

        sample_weights = None
        y_train_res = y_train

        if bal_name == 'ClassWeight':
            sample_weights = compute_sample_weight('balanced', y_train)
        elif bal_name == 'SMOTE':
            smote = SMOTE(random_state=42)
            X_train_proc, y_train_res = smote.fit_resample(X_train_proc, y_train)

        model = build_model(USE_GPU, **config)

        if sample_weights is not None:
            model.fit(X_train_proc, y_train_res, sample_weight=sample_weights)
        else:
            model.fit(X_train_proc, y_train_res)

        if FORCE_GPU and model.booster_.params.get('device_type', 'cpu') != 'gpu':
            raise RuntimeError('LightGBM did not use GPU as requested.')

        y_pred = model.predict(X_test_proc)
        macro_f1s.append(f1_score(y_test, y_pred, average='macro'))

        recalls = recall_score(y_test, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
        recall_c0s.append(recalls[0])
        recall_c1s.append(recalls[1])
        recall_c2s.append(recalls[2])

    return {
        'f1': float(np.mean(macro_f1s)),
        'recall_c0': float(np.mean(recall_c0s)),
        'recall_c1': float(np.mean(recall_c1s)),
        'recall_c2': float(np.mean(recall_c2s)),
    }


def format_config(config, keys):
    return " | ".join(f"{key}={config[key]}" for key in keys)


# 4. Phase 1: select the best preprocessing and balancing combination
print("LightGBM GPU mode enabled.")
with open(PREPROCESSING_RESULTS_FILE, "w", encoding="utf-8") as preprocessing_file:
    preprocessing_file.write("PHASE 1 - PREPROCESSING SELECTION\n")
    preprocessing_file.write("Imputer | Scaler | Balancing || Macro F1 | Recall C0 | Recall C1 | Recall C2\n")
    preprocessing_file.write("-" * 100 + "\n")
    print("Starting LightGBM preprocessing selection...\n")

    best_preproc = {
        'score': -np.inf,
        'imp_name': None,
        'imp_obj': None,
        'scl_name': None,
        'scl_obj': None,
        'bal_name': None,
        'metrics': None,
    }

    for imp_name, imp_obj in imputer_options:
        for scl_name, scl_obj in scaler_options:
            for bal_name in balance_options:
                combo_name = f"{imp_name} | {scl_name} | {bal_name}"
                try:
                    metrics = evaluate_combo(imp_obj, scl_obj, bal_name, config={})
                except Exception as exc:
                    fail_str = f"{combo_name} || FAILED ({type(exc).__name__})\n"
                    preprocessing_file.write(fail_str)
                    print(fail_str.strip())
                    continue

                res_str = (
                    f"{combo_name} || F1: {metrics['f1']:.3f} | "
                    f"Rec0: {metrics['recall_c0']:.3f} | "
                    f"Rec1: {metrics['recall_c1']:.3f} | "
                    f"Rec2: {metrics['recall_c2']:.3f}\n"
                )
                preprocessing_file.write(res_str)
                print(res_str.strip())

                if metrics['f1'] > best_preproc['score']:
                    best_preproc = {
                        'score': metrics['f1'],
                        'imp_name': imp_name,
                        'imp_obj': imp_obj,
                        'scl_name': scl_name,
                        'scl_obj': scl_obj,
                        'bal_name': bal_name,
                        'metrics': metrics,
                    }
                    print(
                        f"New best preprocessing: {imp_name} | {scl_name} | {bal_name} || "
                        f"F1: {metrics['f1']:.3f} | Rec0: {metrics['recall_c0']:.3f} | "
                        f"Rec1: {metrics['recall_c1']:.3f} | Rec2: {metrics['recall_c2']:.3f}"
                    )

    preprocessing_file.write("-" * 100 + "\n")
    if best_preproc['imp_name'] is not None:
        best_preproc_name = f"{best_preproc['imp_name']} | {best_preproc['scl_name']} | {best_preproc['bal_name']}"
        preprocessing_file.write(
            f"BEST PREPROCESSING: {best_preproc_name} || F1: {best_preproc['score']:.3f} | "
            f"Rec0: {best_preproc['metrics']['recall_c0']:.3f} | "
            f"Rec1: {best_preproc['metrics']['recall_c1']:.3f} | "
            f"Rec2: {best_preproc['metrics']['recall_c2']:.3f}\n"
        )
        print(
            f"\nBest preprocessing found: {best_preproc_name} || F1: {best_preproc['score']:.3f} | "
            f"Rec0: {best_preproc['metrics']['recall_c0']:.3f} | "
            f"Rec1: {best_preproc['metrics']['recall_c1']:.3f} | "
            f"Rec2: {best_preproc['metrics']['recall_c2']:.3f}"
        )
    else:
        preprocessing_file.write("No valid preprocessing configuration was found.\n")
        print("\nNo valid preprocessing configuration was found.")

with open(PARAMETER_RESULTS_FILE, "w", encoding="utf-8") as parameter_file:
    # 5. Phase 2: parameter search on the best preprocessing combination
    parameter_file.write("PHASE 2 - PARAMETER SEARCH ON BEST PREPROCESSING\n")
    parameter_file.write("Params || Macro F1 | Recall C0 | Recall C1 | Recall C2\n")
    parameter_file.write("-" * 100 + "\n")

    best_param_result = {
        'score': -np.inf,
        'combo_name': None,
        'metrics': None,
    }

    if best_preproc['imp_obj'] is not None:
        print("\nStarting parameter grid search on the best preprocessing configuration...\n")

        for config in grid_configs:
            try:
                metrics = evaluate_combo(
                    best_preproc['imp_obj'],
                    best_preproc['scl_obj'],
                    best_preproc['bal_name'],
                    config=config,
                )
            except Exception as exc:
                fail_str = f"{format_config(config, param_grid.keys())} || FAILED ({type(exc).__name__})\n"
                parameter_file.write(fail_str)
                print(fail_str.strip())
                continue

            config_str = format_config(config, param_grid.keys())
            res_str = (
                f"{config_str} || F1: {metrics['f1']:.3f} | "
                f"Rec0: {metrics['recall_c0']:.3f} | "
                f"Rec1: {metrics['recall_c1']:.3f} | "
                f"Rec2: {metrics['recall_c2']:.3f}\n"
            )
            parameter_file.write(res_str)
            print(res_str.strip())

            if metrics['f1'] > best_param_result['score']:
                best_param_result = {
                    'score': metrics['f1'],
                    'combo_name': config_str,
                    'metrics': metrics,
                }

        parameter_file.write("-" * 100 + "\n")
        if best_param_result['combo_name'] is not None:
            parameter_file.write(
                f"BEST PARAMS ON BEST PREPROCESSING: {best_param_result['combo_name']} || "
                f"F1: {best_param_result['score']:.3f} | "
                f"Rec0: {best_param_result['metrics']['recall_c0']:.3f} | "
                f"Rec1: {best_param_result['metrics']['recall_c1']:.3f} | "
                f"Rec2: {best_param_result['metrics']['recall_c2']:.3f}\n"
            )
            print(
                f"\nBest parameters on best preprocessing: {best_param_result['combo_name']} || "
                f"F1: {best_param_result['score']:.3f} | "
                f"Rec0: {best_param_result['metrics']['recall_c0']:.3f} | "
                f"Rec1: {best_param_result['metrics']['recall_c1']:.3f} | "
                f"Rec2: {best_param_result['metrics']['recall_c2']:.3f}"
            )
        else:
            parameter_file.write("No valid parameter configuration was found.\n")
            print("\nNo valid parameter configuration was found.")
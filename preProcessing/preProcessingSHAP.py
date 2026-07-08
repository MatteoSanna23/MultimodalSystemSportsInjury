import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings('ignore')

BASE_DIR = Path(r"C:\Users\leozi\Desktop\uni\Magi\AI in Medicine\Multimodalproject\MultimodalSystemSportsInjury")
DATA_FILE = BASE_DIR / 'multimodal_sports_injury_dataset.csv'
RESULTS_DIR = BASE_DIR / 'preProcessing' / 'preProcessingres' / 'SHAP'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SPECS = {
    'LightGBM': {
        'output_file': RESULTS_DIR / 'lightgbm_shap_report.txt',
        'preprocessing': {
            'imputer': 'Median',
            'scaler': 'StdScaler',
            'balancing': 'ClassWeight',
        },
        'params': {
            'max_depth': 3,
            'learning_rate': 0.05,
            'n_estimators': 300,
        },
        'use_gpu': True,
        'source_note': 'Best config taken from preProcessingres/Light_GBM/lightgbm_best_parameters.txt',
    },
    'XGBoost': {
        'output_file': RESULTS_DIR / 'xgboost_shap_report.txt',
        'preprocessing': {
            'imputer': 'Median',
            'scaler': 'MinMaxScaler',
            'balancing': 'ClassWeight',
        },
        'params': {
            'max_depth': 3,
            'learning_rate': 0.05,
            'n_estimators': 200,
        },
        'use_gpu': True,
        'source_note': 'Best config taken from preProcessingres/XGBoost/xgboost_tuning_results.txt and preprocessing_results.txt',
    },
    'RandomForest': {
        'output_file': RESULTS_DIR / 'random_forest_shap_report.txt',
        'preprocessing': {
            'imputer': 'KNN',
            'scaler': 'StdScaler',
            'balancing': 'ClassWeight',
        },
        'params': {
            'n_estimators': 200,
            'max_depth': 10,
            'max_features': 'sqrt',
            'min_samples_leaf': 5,
    },
    'use_gpu': False,
    'source_note': 'Best config taken from preProcessingres/rf/best_preprocessing_config.txt and results_random_forest_grid.txt',
    },
}

def load_dataset():
    df = pd.read_csv(DATA_FILE)
    X = df.drop(['injury_occurred', 'athlete_id', 'session_id'], axis=1)
    y = df['injury_occurred']
    return X, y

def build_preprocessor(imputer_name, scaler_name, cat_cols, num_cols):
    imputer_map = {
        'KNN': KNNImputer(n_neighbors=5),
        'Median': SimpleImputer(strategy='median'),
        'Native_LightGBM': 'passthrough',
    }
    scaler_map = {
        'StdScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
    }

    num_pipe = Pipeline([
        ('imputer', imputer_map[imputer_name]),
        ('scaler', scaler_map[scaler_name]),
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)],
        remainder='drop',
    )

def build_model(model_name, params, use_gpu):
    if model_name == 'LightGBM':
        model_params = dict(
            objective='multiclass',
            num_class=3,
            metric='multi_logloss',
            class_weight=None,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )
        if use_gpu:
            model_params.update({
                'device_type': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'max_bin': 255,
            })
        return lgb.LGBMClassifier(**model_params)

    if model_name == 'XGBoost':
        model_params = dict(
            objective='multi:softmax',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
            **params,
        )
        if use_gpu:
            model_params.update({
                'tree_method': 'hist',
                'device': 'cuda',
            })
        return xgb.XGBClassifier(**model_params)

    if model_name == 'RandomForest':
        return RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            **params,
        )

    raise ValueError(f'Unknown model name: {model_name}')

def get_feature_names(preprocessor, num_cols, cat_cols):
    num_names = list(num_cols)
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    return num_names + cat_names

def fit_and_explain(model_name, spec, X, y):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = build_preprocessor(
        spec['preprocessing']['imputer'],
        spec['preprocessing']['scaler'],
        cat_cols,
        num_cols,
    )

    X_processed = preprocessor.fit_transform(X)
    feature_names = get_feature_names(preprocessor, num_cols, cat_cols)

    sample_weights = None
    X_train = X_processed
    y_train = y

    if spec['preprocessing']['balancing'] == 'ClassWeight':
        sample_weights = compute_sample_weight('balanced', y)
    elif spec['preprocessing']['balancing'] == 'SMOTE':
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_processed, y)
    
    model = build_model(model_name, spec['params'], spec['use_gpu'])
    if sample_weights is not None:
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    if spec['use_gpu'] and model_name in {'LightGBM', 'XGBoost'}:
        print(f'{model_name} trained with GPU settings.')

    # Sample for SHAP explainer performance
    sample_size = min(300, X_processed.shape[0])
    if sample_size < X_processed.shape[0]:
        sample_indices = np.random.RandomState(42).choice(X_processed.shape[0], size=sample_size, replace=False)
        X_shap = X_processed[sample_indices]
    else:
        X_shap = X_processed

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    
    # Calculate mean absolute SHAP values for ranking (Global Report)
    if isinstance(shap_values, list):
        shap_stack = np.stack(shap_values, axis=0)
        mean_shap = np.mean(np.abs(shap_stack), axis=(0, 2))
    elif getattr(shap_values, 'ndim', 0) == 3:
        mean_shap = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)

    top_indices = np.argsort(mean_shap)[::-1][:20]

    # --- SHAP PLOTTING SECTION ---
    
    # Create a subfolder for the current model plots
    model_plot_dir = RESULTS_DIR / model_name
    model_plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure X_shap_df is a standard DataFrame
    X_shap_df = pd.DataFrame(X_shap, columns=feature_names)

    # Standardize shap_values format to iterate over classes correctly
    # (Fixes the issue where some models return a 3D array instead of a list)
    if isinstance(shap_values, list):
        num_classes = len(shap_values)
        class_shap_arrays = shap_values
    elif getattr(shap_values, 'ndim', 0) == 3:
        num_classes = shap_values.shape[2]
        class_shap_arrays = [shap_values[:, :, i] for i in range(num_classes)]
    else:
        num_classes = 1
        class_shap_arrays = [shap_values]

    class_names = ["Non_Injured", "At_Risk", "Injured"]

    # Loop securely over the correctly identified number of classes
    for class_idx in range(num_classes):
        # Prevent IndexError in case the model detects unexpected classes
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"
        
        # Extract the specific 2D matrix for the current class
        current_shap_values = class_shap_arrays[class_idx]
        
        # 1. SUMMARY PLOT (Saved in model subfolder)
        plt.figure()
        shap.summary_plot(current_shap_values, X_shap_df, show=False)
        plt.title(f"Summary Plot: {model_name} - {class_name}")
        plt.tight_layout()
        plt.savefig(model_plot_dir / f"{model_name}_class{class_idx}_summary.png")
        plt.close()
        print(f"Saved Summary Plot for {model_name} - {class_name}")

        # 2. DEPENDENCE PLOT (Saved in model subfolder)
        mean_shap_class = np.abs(current_shap_values).mean(axis=0)
        top_idx_class = np.argsort(mean_shap_class)[-1]
        top_feature_class = feature_names[top_idx_class]
        
        plt.figure()
        shap.dependence_plot(top_feature_class, current_shap_values, X_shap_df, show=False)
        plt.title(f"Dependence Plot: {top_feature_class} ({class_name})")
        plt.tight_layout()
        plt.savefig(model_plot_dir / f"{model_name}_class{class_idx}_dep_{top_feature_class}.png")
        plt.close()
        print(f"Saved Dependence Plot for {top_feature_class} ({class_name})")

    # --- REPORT GENERATION ---
    lines = []
    lines.append(f'MODEL: {model_name}')
    lines.append(f'SOURCE NOTE: {spec["source_note"]}')
    lines.append('PREPROCESSING:')
    lines.append(f'   Imputer: {spec["preprocessing"]["imputer"]}')
    lines.append(f'   Scaler: {spec["preprocessing"]["scaler"]}')
    lines.append(f'   Balancing: {spec["preprocessing"]["balancing"]}')
    lines.append('PARAMETERS:')
    for key, value in spec['params'].items():
        lines.append(f'   {key}: {value}')
    lines.append('')
    lines.append('TOP 20 SHAP FEATURES:')
    for rank, idx in enumerate(top_indices, start=1):
        lines.append(f'{rank:02d}. {feature_names[idx]} || Mean |SHAP|: {mean_shap[idx]:.6f}')

    return '\n'.join(lines) + '\n'

def main():
    print('Loading dataset...')
    X, y = load_dataset()

    for model_name, spec in MODEL_SPECS.items():
        print(f'\nProcessing {model_name}...')
        report = fit_and_explain(model_name, spec, X, y)
        spec['output_file'].write_text(report, encoding='utf-8')
        print(f'Wrote SHAP report to {spec["output_file"]}')

    print('\nProcess completed successfully.')
    print(f'Reports saved in: {RESULTS_DIR}')

if __name__ == '__main__':
    main()

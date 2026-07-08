import pandas as pd
import numpy as np
import xgboost as xgb
from itertools import product
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import fbeta_score, recall_score  # Modificato qui
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CUSTOM METRICS
# ==========================================
def f2_score(y_true, y_pred, average='macro', labels=None, zero_division=0):
    """
    Compute F2 score by setting beta=2 in sklearn's fbeta_score.
    """
    return fbeta_score(y_true, y_pred, beta=2, average=average, labels=labels, zero_division=zero_division)

# ==========================================
# 0. DATA LOADING AND SETUP
# ==========================================
print("Loading data...")
df = pd.read_csv('..//multimodal_sports_injury_dataset.csv')
X = df.drop(['injury_occurred', 'athlete_id', 'session_id'], axis=1)
y = df['injury_occurred']
groups = df['athlete_id']

gkf = GroupKFold(n_splits=5)

# ==========================================
# PHASE 1: COMPREHENSIVE PREPROCESSING SCREENING
# ==========================================
print("\n" + "="*60)
print("PHASE 1: PREPROCESSING SCREENING STARTED")
print("="*60)

imputer_options = [
    ('KNN', KNNImputer(n_neighbors=5)),
    ('Median', SimpleImputer(strategy='median')),
    ('Native_XGB', 'passthrough')
]
scaler_options = [('StdScaler', StandardScaler()), ('MinMaxScaler', MinMaxScaler())]
balance_options = ['ClassWeight', 'SMOTE']

best_prep_score_c2 = -np.inf
best_prep_config = {}

# Open log file for Phase 1
with open("preProcessingRes/XGBoost/xgboost_best_preprocessing.txt", "w") as file_prep:
    file_prep.write("PHASE 1: PREPROCESSING SCREENING\n")
    file_prep.write("-" * 150 + "\n")
    file_prep.write("Imputer | Scaler | Balancing || Macro F2 | F2 C0 | F2 C1 | F2 C2 | Rec0 | Rec1 | Rec2\n")
    file_prep.write("-" * 150 + "\n")

    for imp_name, imp_obj in imputer_options:
        for scl_name, scl_obj in scaler_options:
            for bal_name in balance_options:
                
                combo_name = f"{imp_name} | {scl_name} | {bal_name}"
                macro_f2s = []
                f2_c0s, f2_c1s, f2_c2s = [], [], []
                recall_c0s, recall_c1s, recall_c2s = [], [], []
                
                try:
                    for train_idx, test_idx in gkf.split(X, y, groups=groups):
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                        
                        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
                        cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
                        
                        num_pipe = Pipeline([('imputer', imp_obj), ('scaler', scl_obj)])
                        cat_pipe = Pipeline([
                            ('imputer', SimpleImputer(strategy='most_frequent')),
                            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                        ])
                        preprocessor = ColumnTransformer(
                            transformers=[('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)], remainder='drop'
                        )
                        
                        X_train_proc = preprocessor.fit_transform(X_train)
                        X_test_proc = preprocessor.transform(X_test)
                        
                        y_train_res = y_train
                        sample_weights = None
                        
                        if bal_name == 'ClassWeight':
                            sample_weights = compute_sample_weight('balanced', y_train)
                        elif bal_name == 'SMOTE':
                            smote = SMOTE(random_state=42)
                            X_train_proc, y_train_res = smote.fit_resample(X_train_proc, y_train)
                            
                        # Default Model for the preprocessing test
                        model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, random_state=42)
                        
                        if sample_weights is not None:
                            model.fit(X_train_proc, y_train_res, sample_weight=sample_weights)
                        else:
                            model.fit(X_train_proc, y_train_res)
                            
                        y_pred = model.predict(X_test_proc)
                        
                        macro_f2s.append(f2_score(y_test, y_pred, average='macro'))
                        
                        # Computing F2 per class
                        f2_per_class = f2_score(y_test, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
                        f2_c0s.append(f2_per_class[0])
                        f2_c1s.append(f2_per_class[1])
                        f2_c2s.append(f2_per_class[2])
                        
                        # Computing Recall per class
                        recalls = recall_score(y_test, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
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
                    
                    # Write to file and print to screen
                    res_str = f"{combo_name} || Macro F2: {avg_f2:.3f} | F2_C0: {avg_f2_c0:.3f} | F2_C1: {avg_f2_c1:.3f} | F2_C2: {avg_f2_c2:.3f} | Rec0: {avg_rec0:.3f} | Rec1: {avg_rec1:.3f} | Rec2: {avg_rec2:.3f}\n"
                    file_prep.write(res_str)
                    print(f"[PHASE 1] Tested: {res_str.strip()}")
                    
                    # Save the best configuration in memory (Criteria: Max F2 Classe 2)
                    if avg_f2_c2 > best_prep_score_c2:
                        best_prep_score_c2 = avg_f2_c2
                        best_prep_config = {
                            'name': combo_name, 
                            'imp_obj': imp_obj, 
                            'scl_obj': scl_obj, 
                            'bal_name': bal_name,
                            'macro_f2': avg_f2,
                            'f2_c0': avg_f2_c0,
                            'f2_c1': avg_f2_c1,
                            'rec0': avg_rec0,
                            'rec1': avg_rec1,
                            'rec2': avg_rec2
                        }
                        
                except Exception as e:
                    err_str = f"{combo_name} || FAILED ({type(e).__name__})\n"
                    file_prep.write(err_str)
                    print(f"[PHASE 1] Tested: {err_str.strip()}")

    # Append the best configuration at the bottom of the file
    file_prep.write("\n" + "="*60 + "\n")
    file_prep.write("PHASE 1 WINNER (BEST PREPROCESSING - CRITERIA: MAX F2 CLASS 2)\n")
    file_prep.write("="*60 + "\n")
    file_prep.write(f"Combination: {best_prep_config['name']}\n")
    file_prep.write(f"F2 Class 2 Score: {best_prep_score_c2:.3f}\n")
    file_prep.write(f"Macro F2: {best_prep_config['macro_f2']:.3f} | F2 C0: {best_prep_config['f2_c0']:.3f} | F2 C1: {best_prep_config['f2_c1']:.3f}\n")
    file_prep.write(f"Rec0: {best_prep_config['rec0']:.3f} | Rec1: {best_prep_config['rec1']:.3f} | Rec2: {best_prep_config['rec2']:.3f}\n")
    
print(f"\n=> Phase 1 Completed! Check 'preProcessingRes/XGBoost/xgboost_best_preprocessing.txt' for full logs.")
print(f"=> Phase 1 Winner (Max F2 C2): {best_prep_config['name']} with F2_C2: {best_prep_score_c2:.3f}")

# ==========================================
# PHASE 2: HYPERPARAMETER TUNING WITH BEST PREP
# ==========================================
print("\n" + "="*60)
print("PHASE 2: XGBOOST HYPERPARAMETER TUNING STARTED")
print("="*60)

param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 300, 500]
}
grid_configs = list(ParameterGrid(param_grid))
best_tune_score_c2 = -np.inf
best_tune_config = None
best_tune_metrics = None

with open("preProcessingRes/XGBoost/xgboost_best_parameters.txt", "w") as file_tune:
    header = f"FIXED PREPROCESSING: {best_prep_config['name']}\n"
    file_tune.write(header)
    file_tune.write("-" * 150 + "\n")
    file_tune.write("XGB Params || Macro F2 | F2 C0 | F2 C1 | F2 C2 | Rec0 | Rec1 | Rec2\n")
    file_tune.write("-" * 150 + "\n")

    for config in grid_configs:
        macro_f2s, f2_c0s, f2_c1s, f2_c2s = [], [], [], []
        recall_c0s, recall_c1s, recall_c2s = [], [], []
        
        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
            
            num_pipe = Pipeline([
                ('imputer', best_prep_config['imp_obj']),
                ('scaler', best_prep_config['scl_obj'])
            ])
            cat_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            preprocessor = ColumnTransformer(
                transformers=[('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)], remainder='drop'
            )
            
            X_train_proc = preprocessor.fit_transform(X_train)
            X_test_proc = preprocessor.transform(X_test)
            
            y_train_res = y_train
            sample_weights = None
            
            if best_prep_config['bal_name'] == 'ClassWeight':
                sample_weights = compute_sample_weight('balanced', y_train)
            elif best_prep_config['bal_name'] == 'SMOTE':
                smote = SMOTE(random_state=42)
                X_train_proc, y_train_res = smote.fit_resample(X_train_proc, y_train)
                
            model = xgb.XGBClassifier(
                objective='multi:softmax', num_class=3, eval_metric='mlogloss', random_state=42, **config
            )
            
            if sample_weights is not None:
                model.fit(X_train_proc, y_train_res, sample_weight=sample_weights)
            else:
                model.fit(X_train_proc, y_train_res)
                
            y_pred = model.predict(X_test_proc)
            
            # Calcolo metriche F2
            macro_f2s.append(f2_score(y_test, y_pred, average='macro'))
            f2_per_class = f2_score(y_test, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
            f2_c0s.append(f2_per_class[0])
            f2_c1s.append(f2_per_class[1])
            f2_c2s.append(f2_per_class[2])
            
            # Calcolo metriche Recall
            recalls = recall_score(y_test, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
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
        
        # Update best score based on F2 Class 2
        if avg_f2_c2 > best_tune_score_c2:
            best_tune_score_c2 = avg_f2_c2
            best_tune_config = config
            best_tune_metrics = {
                'macro_f2': avg_f2, 'f2_c0': avg_f2_c0, 'f2_c1': avg_f2_c1,
                'rec0': avg_rec0, 'rec1': avg_rec1, 'rec2': avg_rec2
            }
            
        config_str = f"d:{config['max_depth']}, lr:{config['learning_rate']}, n:{config['n_estimators']}"
        res_str = f"[{config_str}] || F2_Macro: {avg_f2:.3f} | F2_C0: {avg_f2_c0:.3f} | F2_C1: {avg_f2_c1:.3f} | F2_C2: {avg_f2_c2:.3f} | Rec0: {avg_rec0:.3f} | Rec1: {avg_rec1:.3f} | Rec2: {avg_rec2:.3f}"
        
        file_tune.write(res_str + "\n")
        print(f"[PHASE 2] {res_str}")

    # Append overall winner at the bottom of the tuning file
    file_tune.write("\n" + "-" * 150 + "\n")
    file_tune.write(f"OVERALL WINNER PHASE 2 (TUNING - CRITERIA: MAX F2 CLASS 2):\n")
    file_tune.write(f"Parameters: {best_tune_config}\n")
    file_tune.write(f"F2 Class 2: {best_tune_score_c2:.3f} | F2 Macro: {best_tune_metrics['macro_f2']:.3f} | F2 C0: {best_tune_metrics['f2_c0']:.3f} | F2 C1: {best_tune_metrics['f2_c1']:.3f}\n")
    file_tune.write(f"Rec0: {best_tune_metrics['rec0']:.3f} | Rec1: {best_tune_metrics['rec1']:.3f} | Rec2: {best_tune_metrics['rec2']:.3f}\n")

print("\n" + "="*60)
print("PROCESS COMPLETED SUCCESSFULLY!")
print("Check the results in 'preProcessingRes/XGBoost/'")
print("="*60)
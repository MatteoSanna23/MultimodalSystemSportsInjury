import pandas as pd
import numpy as np
import xgboost as xgb
from itertools import product
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, recall_score
from imblearn.over_sampling import SMOTE

# 1. Data loading
df = pd.read_csv('multimodal_sports_injury_dataset.csv')
X = df.drop(['injury_occurred', 'athlete_id', 'session_id'], axis=1)
y = df['injury_occurred']
groups = df['athlete_id']

# 2. Definition of preprocessing and model options
imputer_options = [
    ('KNN', KNNImputer(n_neighbors=5)),
    ('Median', SimpleImputer(strategy='median')),
    ('Native_XGB', 'passthrough') # Leave NaNs so XGBoost can handle them natively
]

scaler_options = [
    ('StdScaler', StandardScaler()),
    ('MinMaxScaler', MinMaxScaler())
]

balance_options = ['ClassWeight', 'SMOTE']

# Build the hyperparameter grid so every combination is evaluated.
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 300, 500],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

param_options = [
    ('Default', {}),
    *[
        (
            ", ".join(f"{key}={value}" for key, value in zip(param_grid.keys(), values)),
            dict(zip(param_grid.keys(), values))
        )
        for values in product(*param_grid.values())
    ]
]

gkf = GroupKFold(n_splits=5)
best_result = {
    'score': -np.inf,
    'combo_name': None,
    'metrics': None,
}

# 3. Open the results text file and start the experiment loops
with open("experiment_results.txt", "w") as file:
    header = "Imputer | Scaler | Balancing | Params || Macro F1 | Recall C1 | Recall C2\n"
    file.write(header)
    file.write("-" * 80 + "\n")
    print(header)

    # Nested loops to explore all parameter combinations
    for imp_name, imp_obj in imputer_options:
        for scl_name, scl_obj in scaler_options:
            for bal_name in balance_options:
                for p_name, p_dict in param_options:
                    
                    combo_name = f"{imp_name} | {scl_name} | {bal_name} | {p_name}"
                    macro_f1s, recall_class0_scores, recall_class1_scores, recall_class2_scores = [], [], [], []
                    
                    try:
                        for train_idx, test_idx in gkf.split(X, y, groups=groups):
                            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                            
                            num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
                            cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
                            
                            # Preprocessing pipelines
                            num_pipe = Pipeline([('imputer', imp_obj), ('scaler', scl_obj)])
                            cat_pipe = Pipeline([
                                ('imputer', SimpleImputer(strategy='most_frequent')),
                                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                            ])
                            
                            preprocessor = ColumnTransformer(
                                transformers=[('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)],
                                remainder='drop'
                            )
                            
                            X_train_proc = preprocessor.fit_transform(X_train)
                            X_test_proc = preprocessor.transform(X_test)
                            
                            # Balancing handling
                            sample_weights = None
                            y_train_res = y_train
                            
                            if bal_name == 'ClassWeight':
                                sample_weights = compute_sample_weight('balanced', y_train)
                            elif bal_name == 'SMOTE':
                                smote = SMOTE(random_state=42)
                                X_train_proc, y_train_res = smote.fit_resample(X_train_proc, y_train)
                                
                            # Model
                            model = xgb.XGBClassifier(
                                objective='multi:softmax', 
                                num_class=3, 
                                eval_metric='mlogloss', 
                                random_state=42, 
                                **p_dict
                            )
                            
                            if sample_weights is not None:
                                model.fit(X_train_proc, y_train_res, sample_weight=sample_weights)
                            else:
                                model.fit(X_train_proc, y_train_res)
                                
                            # Evaluation
                            y_pred = model.predict(X_test_proc)
                            fold_f1 = f1_score(y_test, y_pred, average='macro')
                            recalls = recall_score(y_test, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
                            fold_recall_c0 = recalls[0]
                            fold_recall_c1 = recalls[1]
                            fold_recall_c2 = recalls[2]


                            macro_f1s.append(fold_f1)
                            recall_class0_scores.append(fold_recall_c0)
                            recall_class1_scores.append(fold_recall_c1)
                            recall_class2_scores.append(fold_recall_c2)
                            
                        # Write results on success
                        res_str = (
                            f"{combo_name} || F1: {np.mean(macro_f1s):.3f} | "
                            f"Rec0: {np.mean(recall_class0_scores):.3f} | "
                            f"Rec1: {np.mean(recall_class1_scores):.3f} | "
                            f"Rec2: {np.mean(recall_class2_scores):.3f}\n"
                        )
                        file.write(res_str)
                        print(res_str.strip())

                        mean_macro_f1 = float(np.mean(macro_f1s))
                        if mean_macro_f1 > best_result['score']:
                            # Keep the best configuration according to macro F1.
                            best_result = {
                                'score': mean_macro_f1,
                                'combo_name': combo_name,
                                'metrics': {
                                    'recall_c0': float(np.mean(recall_class0_scores)),
                                    'recall_c1': float(np.mean(recall_class1_scores)),
                                    'recall_c2': float(np.mean(recall_class2_scores)),
                                },
                            }
                        
                    except Exception as e:
                        # Write errors (e.g. SMOTE fails if there are NaNs)
                        err_str = f"{combo_name} || FAILED ({type(e).__name__})\n"
                        file.write(err_str)
                        print(err_str.strip())

    file.write("\nBEST CONFIGURATION\n")
    if best_result['combo_name'] is not None:
        best_summary = (
            f"{best_result['combo_name']} || F1: {best_result['score']:.3f} | "
            f"Rec0: {best_result['metrics']['recall_c0']:.3f} | "
            f"Rec1: {best_result['metrics']['recall_c1']:.3f} | "
            f"Rec2: {best_result['metrics']['recall_c2']:.3f}\n"
        )
        file.write(best_summary)
        print("\nBest configuration:")
        print(best_summary.strip())
    else:
        file.write("No valid configuration was found.\n")
        print("\nNo valid configuration was found.")

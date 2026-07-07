import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, recall_score, classification_report

# 1. Data Loading (Assuming the CSV has already been loaded in 'df')
df = pd.read_csv('multimodal_sports_injury_dataset.csv')
# Separation of features (X), target (y) and groups (athlete_id)
X = df.drop(['injury_occurred', 'athlete_id', 'session_id'], axis=1)
y = df['injury_occurred']
groups = df['athlete_id']

# 2. Subject-Wise Validation Setup
# n_splits=5 means we divide the 156 athletes into 5 groups (~31 athletes per test fold)
gkf = GroupKFold(n_splits=5)

# Array to save metrics for each fold
macro_f1_scores = []
recall_class2_scores = []

# 3. Golden Loop of Cross Validation
for train_idx, test_idx in gkf.split(X, y, groups=groups):
    
    # Strict division for this fold: athletes in train are not in test
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Identify numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # 4. Imbalance Management (Class Weighting for XGBoost)
    # We calculate sample weights only based on the distribution of the Training Set
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    # 5. Preprocessing Pipeline (Interchangeable to test multiple approaches)
    # Create separate pipelines for numeric and categorical columns
    numeric_transformer = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    
    # BEWARE OF DATA LEAKAGE: fit_transform() ONLY on train, transform() on test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 6. XGBoost Setup and Training
    # objective='multi:softmax' indicates multi-class classification
    model = xgb.XGBClassifier(
        objective='multi:softmax', 
        num_class=3, 
        eval_metric='mlogloss',
        random_state=42
    )
    
    # We train by explicitly passing weights to balance the classes
    model.fit(X_train_processed, y_train, sample_weight=sample_weights)
    
    # 7. Fold Evaluation
    y_pred = model.predict(X_test_processed)
    
    # Calculation of required metrics
    fold_f1 = f1_score(y_test, y_pred, average='macro')
    # We extract recall specifically for Class 2 (Index 2)
    fold_recall_c2 = recall_score(y_test, y_pred, average=None)[2] 
    
    macro_f1_scores.append(fold_f1)
    recall_class2_scores.append(fold_recall_c2)
    
    print("Fold Complete. Macro F1:", round(fold_f1, 3), "| Recall Class 2:", round(fold_recall_c2, 3))

print("\n--- FINAL RESULTS ---")
print(f"Average Macro F1-Score: {np.mean(macro_f1_scores):.3f}")
print(f"Average Class 2 Recall: {np.mean(recall_class2_scores):.3f}")
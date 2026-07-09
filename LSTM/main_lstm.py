import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import fbeta_score, recall_score
import warnings

warnings.filterwarnings('ignore')

# 1. Import your custom modules
from lstm import InjuryPredictionLSTM
from sliding_windows import create_sliding_windows, SportsInjuryDataset

# ==========================================
# GLOBAL CONFIGURATIONS
# ==========================================
SEQ_LENGTH = 5              # 5-session sliding window (Training microcycle)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ==========================================
# DATA LOADING
# ==========================================
df = pd.read_csv('..//multimodal_sports_injury_dataset.csv')
X = df.drop(['injury_occurred'], axis=1) # Keep athlete_id and session_id temporarily for sequence grouping
y = df['injury_occurred']
groups = df['athlete_id']

gkf = GroupKFold(n_splits=5)

# Metrics tracking arrays
fold_f2_c2 = []
fold_rec_c2 = []

# Variables to handle the saving of the best model weights
best_overall_f2_c2 = -np.inf
best_model_weights = None
best_fold_identifier = None

print("\n" + "="*70)
print(f"STARTING LSTM TRAINING (Time Steps: {SEQ_LENGTH}) - Subject-Wise Cross-Validation")
print("="*70)

# ==========================================
# THE GOLDEN LOOP (SUBJECT-WISE CV)
# ==========================================
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
    print(f"\n--- FOLD {fold} ---")
    
    # 1. Raw Split
    X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
    y_train_raw, y_test_raw = y.iloc[train_idx], y.iloc[test_idx]
    
    # Isolate feature columns by dropping tracking IDs
    feature_cols_to_scale = X_train_raw.drop(['athlete_id', 'session_id'], axis=1).columns.tolist()
    num_cols = X_train_raw[feature_cols_to_scale].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train_raw[feature_cols_to_scale].select_dtypes(include=['object']).columns.tolist()
    
    # 2. Block 1 Winning Preprocessing Pipeline (Median + MinMaxScaler)
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', MinMaxScaler())])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
    
    # Fit strictly on the Training Fold to prevent Data Leakage
    X_train_proc_array = preprocessor.fit_transform(X_train_raw[feature_cols_to_scale])
    X_test_proc_array = preprocessor.transform(X_test_raw[feature_cols_to_scale])
    
    # 3. Re-assemble DataFrames to preserve IDs for the sliding window logic
    proc_feature_names = [f"f_{i}" for i in range(X_train_proc_array.shape[1])]
    
    df_train_proc = pd.DataFrame(X_train_proc_array, columns=proc_feature_names, index=X_train_raw.index)
    df_train_proc['athlete_id'] = X_train_raw['athlete_id']
    df_train_proc['session_id'] = X_train_raw['session_id']
    df_train_proc['injury_occurred'] = y_train_raw
    
    df_test_proc = pd.DataFrame(X_test_proc_array, columns=proc_feature_names, index=X_test_raw.index)
    df_test_proc['athlete_id'] = X_test_raw['athlete_id']
    df_test_proc['session_id'] = X_test_raw['session_id']
    df_test_proc['injury_occurred'] = y_test_raw
    
    # 4. Generate 3D Tensorial Shapes (Sliding Windows)
    X_train_t, y_train_t = create_sliding_windows(df_train_proc, SEQ_LENGTH, proc_feature_names, 'injury_occurred')
    X_test_t, y_test_t = create_sliding_windows(df_test_proc, SEQ_LENGTH, proc_feature_names, 'injury_occurred')
    
    # Initialize PyTorch DataLoaders
    train_dataset = SportsInjuryDataset(X_train_t, y_train_t)
    test_dataset = SportsInjuryDataset(X_test_t, y_test_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 5. Compute Cost-Sensitive Dynamic Class Weights for Loss Function
    classes = np.unique(y_train_t.numpy())
    weights = compute_class_weight('balanced', classes=classes, y=y_train_t.numpy())
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    
    # 6. Initialize Model Architecture, Criterion, and Optimizer
    input_dim = X_train_t.shape[2]
    model = InjuryPredictionLSTM(input_dim=input_dim, hidden_dim=64, num_layers=1, dropout_rate=0.3).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 7. Core Optimization Training Loop (Epochs)
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X) 
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Display progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}")
            
    # 8. Evaluation on Isolated Test Fold
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(DEVICE)
            outputs = model(batch_X)
            _, preds = torch.max(outputs, 1) 
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            
    # Metric Extraction (F2-Score and Recall for Class 2)
    fold_f2 = fbeta_score(all_targets, all_preds, beta=2.0, average=None, labels=[0, 1, 2], zero_division=0)[2]
    fold_rec = recall_score(all_targets, all_preds, average=None, labels=[0, 1, 2], zero_division=0)[2]
    
    fold_f2_c2.append(fold_f2)
    fold_rec_c2.append(fold_rec)
    
    print(f"   => F2 Class 2 (Fold {fold}): {fold_f2:.4f} | Recall C2: {fold_rec:.4f}")
    
    # 9. Weight Extraction Condition (Track the best performing fold brain)
    if fold_f2 > best_overall_f2_c2:
        best_overall_f2_c2 = fold_f2
        best_fold_identifier = fold
        import copy
        best_model_weights = copy.deepcopy(model.state_dict())

# ==========================================
# FINAL REPORT & WEIGHT EXPORTATION
# ==========================================
print("\n" + "="*70)
print("FINAL LSTM PERFORMANCE SUMMARY (Cross-Validation Mean)")
print("="*70)
mean_f2 = np.mean(fold_f2_c2)
std_f2 = np.std(fold_f2_c2)
mean_rec = np.mean(fold_rec_c2)
std_rec = np.std(fold_rec_c2)

print(f"F2 Score Mean (Class 2): {mean_f2:.4f} +/- {std_f2:.4f}")
print(f"Recall Mean   (Class 2): {mean_rec:.4f} +/- {std_rec:.4f}")
print("-" * 70)

# Secure export of the chosen parameters
if best_model_weights is not None:
    export_filename = "best_lstm_injury_model.pth"
    torch.save(best_model_weights, export_filename)
    print(f"Successfully saved brain assets from Fold {best_fold_identifier} to: {export_filename}")
    print(f"Peak validation F2 Class 2 reached: {best_overall_f2_c2:.4f}")
else:
    print("Error: Could not extract weights safely.")
print("="*70)

# Generate and save the TXT report
report_lines = []
report_lines.append("="*70)
report_lines.append(f"LSTM TRAINING REPORT (Time Steps: {SEQ_LENGTH})")
report_lines.append("="*70 + "\n")
report_lines.append("FOLD-BY-FOLD RESULTS (Class 2):")
for i in range(len(fold_f2_c2)):
    report_lines.append(f"Fold {i+1} || F2_C2: {fold_f2_c2[i]:.4f} | Recall_C2: {fold_rec_c2[i]:.4f}")

report_lines.append("\n" + "-"*70)
report_lines.append("FINAL AGGREGATED METRICS:")
report_lines.append(f"Mean F2_C2:     {mean_f2:.4f} +/- {std_f2:.4f}")
report_lines.append(f"Mean Recall_C2: {mean_rec:.4f} +/- {std_rec:.4f}")
report_lines.append("-"*70)

if best_model_weights is not None:
    report_lines.append(f"\nBest Model Saved: {export_filename} (from Fold {best_fold_identifier})")
    report_lines.append(f"Peak F2_C2:       {best_overall_f2_c2:.4f}")

report_text = "\n".join(report_lines)
report_filename = "lstm_training_report.txt"

with open(report_filename, "w") as f:
    f.write(report_text)

print(f"TXT report successfully saved to: {report_filename}")
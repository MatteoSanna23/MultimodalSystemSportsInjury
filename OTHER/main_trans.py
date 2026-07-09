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
import copy

warnings.filterwarnings('ignore')

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 1. Import your custom modules
from model import InjuryPredictionTransformer
from LSTM.sliding_windows import create_sliding_windows, SportsInjuryDataset


# ==========================================
# GLOBAL CONFIGURATIONS
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Grid Search Parameters
GRID_SEQ_LENGTHS = [3, 5, 7]
GRID_HIDDEN_DIMS = [32, 64]
GRID_EPOCHS = 15

# Mega Training Parameters
MEGA_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

print(f"Using device: {DEVICE}")

# ==========================================
# DATA LOADING
# ==========================================
print("Loading data...")
df = pd.read_csv("..//multimodal_sports_injury_dataset.csv")
X = df.drop(['injury_occurred'], axis=1) 
y = df['injury_occurred']
groups = df['athlete_id']

# ==========================================
# CORE FUNCTION: THE GOLDEN LOOP
# ==========================================
def run_cv_training(seq_length, hidden_dim, epochs, use_early_stopping=False, save_best_model=False):
    """
    Executes the 5-fold Subject-Wise CV for a specific Transformer configuration.
    Returns the mean F2-Score for Class 2.
    """
    gkf = GroupKFold(n_splits=5)
    
    fold_f2_c2 = []
    fold_rec_c2 = []
    
    best_overall_f2_c2 = -np.inf
    best_model_weights = None
    best_fold_identifier = None

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        # 1. Raw Split
        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train_raw, y_test_raw = y.iloc[train_idx], y.iloc[test_idx]
        
        feature_cols = X_train_raw.drop(['athlete_id', 'session_id'], axis=1).columns.tolist()
        num_cols = X_train_raw[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_train_raw[feature_cols].select_dtypes(include=['object']).columns.tolist()
        
        # 2. Preprocessing
        num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', MinMaxScaler())])
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer(transformers=[('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
        
        X_train_proc = preprocessor.fit_transform(X_train_raw[feature_cols])
        X_test_proc = preprocessor.transform(X_test_raw[feature_cols])
        
        # 3. Re-assemble DataFrames
        proc_feat_names = [f"f_{i}" for i in range(X_train_proc.shape[1])]
        
        df_train_proc = pd.DataFrame(X_train_proc, columns=proc_feat_names, index=X_train_raw.index)
        df_train_proc['athlete_id'], df_train_proc['session_id'], df_train_proc['injury_occurred'] = X_train_raw['athlete_id'], X_train_raw['session_id'], y_train_raw
        
        df_test_proc = pd.DataFrame(X_test_proc, columns=proc_feat_names, index=X_test_raw.index)
        df_test_proc['athlete_id'], df_test_proc['session_id'], df_test_proc['injury_occurred'] = X_test_raw['athlete_id'], X_test_raw['session_id'], y_test_raw
        
        # 4. Sliding Windows
        X_train_t, y_train_t = create_sliding_windows(df_train_proc, seq_length, proc_feat_names, 'injury_occurred')
        X_test_t, y_test_t = create_sliding_windows(df_test_proc, seq_length, proc_feat_names, 'injury_occurred')
        
        train_loader = DataLoader(SportsInjuryDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(SportsInjuryDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)
        
        # 5. Class Weights
        classes = np.unique(y_train_t.numpy())
        weights = compute_class_weight('balanced', classes=classes, y=y_train_t.numpy())
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
        
        # 6. Model Init
        input_dim = X_train_t.shape[2]
        model = InjuryPredictionTransformer(input_dim=input_dim, d_model=32, nhead=4, num_layers=1).to(DEVICE)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Tracking for Early Stopping
        fold_best_val_f2 = -np.inf
        fold_best_rec = 0.0
        patience_counter = 0
        best_epoch_weights = None
        
        # 7. Training Loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
                
            # Validation at the end of each epoch (Needed for Early Stopping)
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    preds = torch.max(model(batch_X.to(DEVICE)), 1)[1]
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(batch_y.numpy())
                    
            val_f2 = fbeta_score(all_targets, all_preds, beta=2.0, average=None, labels=[0, 1, 2], zero_division=0)[2]
            val_rec = recall_score(all_targets, all_preds, average=None, labels=[0, 1, 2], zero_division=0)[2]
            
            # Early Stopping Logic
            if val_f2 > fold_best_val_f2:
                fold_best_val_f2 = val_f2
                fold_best_rec = val_rec
                patience_counter = 0
                best_epoch_weights = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                
            if use_early_stopping and patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"      [Fold {fold}] Early stopping triggered at epoch {epoch+1}")
                break
                
        fold_f2_c2.append(fold_best_val_f2)
        fold_rec_c2.append(fold_best_rec)
        
        # Store global best model
        if save_best_model and fold_best_val_f2 > best_overall_f2_c2:
            best_overall_f2_c2 = fold_best_val_f2
            best_fold_identifier = fold
            best_model_weights = copy.deepcopy(best_epoch_weights)

    mean_f2 = np.mean(fold_f2_c2)
    mean_rec = np.mean(fold_rec_c2)
    
    if save_best_model and best_model_weights is not None:
        torch.save(best_model_weights, "best_transformer_mega_model.pth")
        
        # Save Mega Training TXT Report
        report = [
            "="*70,
            f"TRANSFORMER MEGA TRAINING REPORT (Time Steps: {seq_length}, Hidden: {hidden_dim})",
            "="*70,
            "FOLD-BY-FOLD RESULTS (Class 2):"
        ]
        for i in range(len(fold_f2_c2)):
            report.append(f"Fold {i+1} || F2_C2: {fold_f2_c2[i]:.4f} | Recall_C2: {fold_rec_c2[i]:.4f}")
        report.extend([
            "-"*70,
            "FINAL AGGREGATED METRICS:",
            f"Mean F2_C2:     {mean_f2:.4f} +/- {np.std(fold_f2_c2):.4f}",
            f"Mean Recall_C2: {mean_rec:.4f} +/- {np.std(fold_rec_c2):.4f}",
            "-"*70,
            f"Best Model Saved: best_transformer_mega_model.pth (from Fold {best_fold_identifier})"
        ])
        with open("transformer_mega_training_report.txt", "w") as f:
            f.write("\n".join(report))
            
    return mean_f2, mean_rec

# ==========================================
# PHASE 1: GRID SEARCH
# ==========================================
print("\n" + "="*70)
print(f"PHASE 1: TRANSFORMER GRID SEARCH (Epochs: {GRID_EPOCHS})")
print("="*70)

best_grid_f2 = -np.inf
best_grid_config = {}

with open("transformer_grid_search_results.txt", "w") as f:
    f.write("TRANSFORMER GRID SEARCH RESULTS\n" + "-"*50 + "\n")
    
    for seq in GRID_SEQ_LENGTHS:
        for hid in GRID_HIDDEN_DIMS:
            print(f"\nTesting Config -> Seq_Length: {seq} | Hidden_Dim: {hid}")
            
            mean_f2, mean_rec = run_cv_training(
                seq_length=seq, 
                hidden_dim=hid, 
                epochs=GRID_EPOCHS, 
                use_early_stopping=False, 
                save_best_model=False
            )
            
            res_str = f"Seq:{seq} | Hidden:{hid} || Mean F2_C2: {mean_f2:.4f} | Mean Rec_C2: {mean_rec:.4f}"
            print(f"Result: {res_str}")
            f.write(res_str + "\n")
            
            if mean_f2 > best_grid_f2:
                best_grid_f2 = mean_f2
                best_grid_config = {'seq_length': seq, 'hidden_dim': hid}

    f.write("-" * 50 + "\n")
    f.write(f"BEST CONFIG: {best_grid_config} with F2: {best_grid_f2:.4f}\n")

print(f"\nPhase 1 Complete! Best Configuration: {best_grid_config}")

# ==========================================
# PHASE 2: MEGA TRAINING
# ==========================================
print("\n" + "="*70)
print(f"PHASE 2: MEGA TRAINING (Epochs: {MEGA_EPOCHS}, Early Stopping: {EARLY_STOPPING_PATIENCE})")
print(f"Using Optimal Config: Seq_Length: {best_grid_config['seq_length']}, Hidden_Dim: {best_grid_config['hidden_dim']}")
print("="*70)

run_cv_training(
    seq_length=best_grid_config['seq_length'], 
    hidden_dim=best_grid_config['hidden_dim'], 
    epochs=MEGA_EPOCHS, 
    use_early_stopping=True, 
    save_best_model=True
)

print("\nProcess fully completed! Check 'transformer_grid_search_results.txt' and 'transformer_mega_training_report.txt'.")
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.calibration import calibration_curve
from torch.utils.data import DataLoader

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your modules (adjust paths if necessary)
from model import InjuryPredictionTransformer
from LSTM.sliding_windows import create_sliding_windows, SportsInjuryDataset

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LENGTH = 7
# FATAL ERROR FIX: The saved .pth checkpoint physically contains 32 hidden nodes, not 64.
HIDDEN_DIM = 32 
TARGET_FOLD = 4 # The best model was saved from Fold 4
OUTPUT_DIR = "calibration-results"

# Create folder for results
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class TemperatureScaler(nn.Module):
    """
    Applies Temperature Scaling. Extends nn.Module to optimize
    the 'temperature' parameter via PyTorch gradients.
    """
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5) # Standard initialization

    def forward(self, logits):
        return logits / self.temperature

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculates the Expected Calibration Error (ECE)
    """
    bin_edges = np.linspace(0., 1., n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
        if np.any(bin_mask):
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            bin_weight = np.sum(bin_mask) / len(y_prob)
            ece += bin_weight * np.abs(bin_acc - bin_conf)
    return ece

def main():
    print(f"1. Loading data and isolating Validation Fold (Fold {TARGET_FOLD})...")
    df = pd.read_csv(r"C:\Users\leozi\Desktop\uni\Magi\AI in Medicine\Multimodalproject\MultimodalSystemSportsInjury\multimodal_sports_injury_dataset.csv")
    X = df.drop(['injury_occurred'], axis=1) 
    y = df['injury_occurred']
    groups = df['athlete_id']

    gkf = GroupKFold(n_splits=5)
    
    # Isolate the exact Fold 4 test data to prevent data leakage
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        if fold == TARGET_FOLD:
            X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
            y_train_raw, y_test_raw = y.iloc[train_idx], y.iloc[test_idx]
            break
            
    # Preprocessing (replicated exactly as in training)
    feature_cols = X_train_raw.drop(['athlete_id', 'session_id'], axis=1).columns.tolist()
    num_cols = X_train_raw[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train_raw[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', MinMaxScaler())])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
    
    # Fit on train, transform on test
    preprocessor.fit(X_train_raw[feature_cols])
    X_test_proc = preprocessor.transform(X_test_raw[feature_cols])
    
    proc_feat_names = [f"f_{i}" for i in range(X_test_proc.shape[1])]
    df_test_proc = pd.DataFrame(X_test_proc, columns=proc_feat_names, index=X_test_raw.index)
    df_test_proc['athlete_id'] = X_test_raw['athlete_id']
    df_test_proc['session_id'] = X_test_raw['session_id']
    df_test_proc['injury_occurred'] = y_test_raw
    
    # Sliding Windows only on Test Set
    X_test_t, y_test_t = create_sliding_windows(df_test_proc, SEQ_LENGTH, proc_feat_names, 'injury_occurred')
    test_loader = DataLoader(SportsInjuryDataset(X_test_t, y_test_t), batch_size=32, shuffle=False)
    
    print("2. Loading the saved Transformer model (Best Model)...")
    input_dim = X_test_t.shape[2]
    # Use d_model parameter specifically for the Transformer
    model = InjuryPredictionTransformer(input_dim=input_dim, d_model=HIDDEN_DIM, nhead=4, num_layers=1, dropout_rate=0.3).to(DEVICE)
    
    # Update path if the weights file is in a different folder
    model.load_state_dict(torch.load("best_transformer_mega_model.pth", map_location=DEVICE))
    model.eval()
    
    # Extracting Logits and Labels
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            logits = model(batch_X.to(DEVICE))
            all_logits.append(logits.cpu())
            all_labels.append(batch_y)
            
    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    
    print("3. Optimizing Temperature Scaling...")
    # Calculate Initial Loss (Pre-Calibration)
    criterion = nn.CrossEntropyLoss()
    initial_loss = criterion(logits_tensor, labels_tensor).item()
    
    # Optimizing parameter T
    scaler = TemperatureScaler()
    optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)
    
    def eval_closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits_tensor), labels_tensor)
        loss.backward()
        return loss
        
    optimizer.step(eval_closure)
    ottimo_T = scaler.temperature.item()
    final_loss = criterion(scaler(logits_tensor), labels_tensor).item()
    
    print(f"   Optimal Temperature found: {ottimo_T:.4f}")
    
    print("4. Generating Plots and Numerical Results...")
    probs_uncalibrated = torch.softmax(logits_tensor, dim=1).numpy()
    
    with torch.no_grad():
        calibrated_logits = scaler(logits_tensor)
        probs_calibrated = torch.softmax(calibrated_logits, dim=1).numpy()
        
    # Isolate probabilities for Class 2 (High Risk / Injury)
    y_true_class2 = (labels_tensor.numpy() == 2).astype(int)
    prob_uncal_c2 = probs_uncalibrated[:, 2]
    prob_cal_c2 = probs_calibrated[:, 2]
    
    # Calculate ECE
    ece_uncal = expected_calibration_error(y_true_class2, prob_uncal_c2, n_bins=10)
    ece_cal = expected_calibration_error(y_true_class2, prob_cal_c2, n_bins=10)
    
    # Calculate Calibration Curves for plotting
    fraction_of_positives_uncal, mean_predicted_value_uncal = calibration_curve(y_true_class2, prob_uncal_c2, n_bins=10)
    fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(y_true_class2, prob_cal_c2, n_bins=10)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    plt.plot(mean_predicted_value_uncal, fraction_of_positives_uncal, "s-", label=f"Transformer (Uncalibrated) - ECE: {ece_uncal:.4f}")
    plt.plot(mean_predicted_value_cal, fraction_of_positives_cal, "o-", label=f"Transformer (Calibrated) - ECE: {ece_cal:.4f}")
    plt.ylabel("True Injury Frequency")
    plt.xlabel("Predicted Probability")
    plt.title("Reliability Diagram (Calibration Curve) - Transformer Class 2")
    plt.legend(loc="best")
    plt.grid(True)
    
    plot_path = os.path.join(OUTPUT_DIR, "transformer_calibration_curve_class2.png")
    plt.savefig(plot_path)
    print(f"   Plot saved to: {plot_path}")
    
    # Save Numerical Results
    txt_path = os.path.join(OUTPUT_DIR, "transformer_calibration_numerical_results.txt")
    with open(txt_path, "w") as f:
        f.write("=== TRANSFORMER CALIBRATION RESULTS ===\n")
        f.write(f"Model: best_transformer_mega_model.pth (Fold {TARGET_FOLD})\n")
        f.write(f"Optimal T Parameter: {ottimo_T:.4f}\n")
        f.write(f"Original Cross-Entropy Loss: {initial_loss:.4f}\n")
        f.write(f"Calibrated Cross-Entropy Loss: {final_loss:.4f}\n")
        f.write("\nExpected Calibration Error (ECE):\n")
        f.write(f"ECE (Uncalibrated): {ece_uncal:.4f}\n")
        f.write(f"ECE (Calibrated):   {ece_cal:.4f}\n")
        f.write("\nMean Probability Distribution for Class 2:\n")
        f.write(f"Mean predicted probability (Uncalibrated): {np.mean(prob_uncal_c2):.4f}\n")
        f.write(f"Mean predicted probability (Calibrated):   {np.mean(prob_cal_c2):.4f}\n")
        f.write(f"True frequency of injuries in the set:     {np.mean(y_true_class2):.4f}\n")
        
    print(f"   Text report saved to: {txt_path}")
    print(f"   Final ECE improved from {ece_uncal:.4f} to {ece_cal:.4f}")
    print("\nProcess successfully completed!")

if __name__ == "__main__":
    main()
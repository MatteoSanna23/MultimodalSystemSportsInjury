import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import warnings

warnings.filterwarnings('ignore')

# 1. Custom module imports
from lstm import InjuryPredictionLSTM
from sliding_windows import create_sliding_windows, SportsInjuryDataset
from torch.utils.data import DataLoader

# ==========================================
# GLOBAL CONFIGURATIONS
# ==========================================
SEQ_LENGTH = 7              # Using 7 days, the winning config from your Grid Search
HIDDEN_DIM = 32             # The winning config
MODEL_WEIGHTS_PATH = "best_lstm_mega_model.pth"
DATASET_PATH = "multimodal_sports_injury_dataset.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Starting plot generation on device: {DEVICE}")

# ==========================================
# PLOTTING FUNCTIONS
# ==========================================
def plot_clinical_confusion_matrix(y_true, y_pred, save_path="confusion_matrix_lstm.png"):
    """Generates and saves a clinical Confusion Matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    # Prevents division by zero if a class happens to be empty
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = np.true_divide(cm, cm.sum(axis=1)[:, np.newaxis])
        cm_normalized[np.isnan(cm_normalized)] = 0
    
    class_names = ['Healthy (Class 0)', 'Low Risk (Class 1)', 'High Risk (Class 2)']
    
    plt.figure(figsize=(10, 8))
    
    annot_data = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_data[i, j] = f"{cm_normalized[i, j]:.1%}\n({cm[i, j]})"
            
    sns.heatmap(cm_normalized, annot=annot_data, fmt='', cmap='Reds', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Recall (Interception Rate)'})
    
    plt.title('Clinical Confusion Matrix - LSTM System', fontsize=14, pad=20, fontweight='bold')
    plt.ylabel('Actual Athlete Condition', fontsize=12, fontweight='bold')
    plt.xlabel('LSTM System Prediction', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" => Saved: {save_path}")


def plot_athlete_risk_trajectory(model, athlete_id, df_proc, seq_length, feature_cols, device, save_path):
    """Tracks the risk evolution (Class 2) for a single athlete."""
    model.eval()
    
    # Filter and sort athlete data
    athlete_data = df_proc[df_proc['athlete_id'] == athlete_id].sort_values('session_id').reset_index(drop=True)
    
    if len(athlete_data) <= seq_length:
        print(f" [!] Athlete {athlete_id} does not have enough sessions (minimum {seq_length + 1}). Skipped.")
        return
        
    features = athlete_data[feature_cols].values
    risk_probabilities = []
    recovery_scores = [] 
    days = []
    
    # Identify the index of the 'recovery_score' column (crucial according to SHAP)
    try:
        rec_idx = feature_cols.index('recovery_score') 
    except ValueError:
        print(" [!] 'recovery_score' column not found. Using the first feature as a visual fallback.")
        rec_idx = 0 
    
    # Slide the window to predict risk day by day
    with torch.no_grad():
        for i in range(len(athlete_data) - seq_length + 1):
            window_x = features[i : i + seq_length]
            tensor_x = torch.tensor(np.array([window_x]), dtype=torch.float32).to(device)
            
            logits = model(tensor_x)
            probs = F.softmax(logits, dim=1)
            
            risk_prob_c2 = probs[0, 2].item() * 100 
            
            risk_probabilities.append(risk_prob_c2)
            recovery_scores.append(window_x[-1, rec_idx]) 
            days.append(f"Day {i+seq_length}")

    # Dual-axis plot creation
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color1 = '#d62728' # Red
    ax1.set_xlabel('Session Chronology (Sliding Window)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('High Injury Risk Probability [%]', color=color1, fontsize=12, fontweight='bold')
    line1, = ax1.plot(days, risk_probabilities, color=color1, marker='o', linewidth=3, markersize=8, label='Risk Probability (LSTM)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(-5, 105)
    
    # Critical threshold line
    ax1.axhline(y=50, color='r', linestyle='--', alpha=0.4, label='Clinical Alarm Threshold (50%)')

    color2 = '#1f77b4' # Blue
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Recovery Score (Normalized)', color=color2, fontsize=12, fontweight='bold')  
    line2, = ax2.plot(days, recovery_scores, color=color2, marker='x', linestyle=':', linewidth=2, markersize=8, label='Recovery Score')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(-0.05, 1.05)

    # Aggregated legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=11)
    
    plt.title(f'Injury Risk Temporal Trajectory - Athlete ID: {athlete_id}', fontsize=15, pad=20, fontweight='bold')
    
    # Reduce X-axis labels if there are too many
    if len(days) > 15:
        ax1.set_xticks(np.arange(0, len(days), step=int(len(days)/10)))
        
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" => Saved: {save_path}")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    print("1. Loading data and applying global Preprocessing...")
    df = pd.read_csv(DATASET_PATH)
    
    # Keep IDs aside for windowing logic
    athlete_ids = df['athlete_id']
    session_ids = df['session_id']
    targets = df['injury_occurred']
    
    X_raw = df.drop(['athlete_id', 'session_id', 'injury_occurred'], axis=1)
    
    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_raw.select_dtypes(include=['object']).columns.tolist()
    
    # Replication of the winning preprocessing (Median + MinMaxScaler)
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', MinMaxScaler())])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
    
    X_proc_array = preprocessor.fit_transform(X_raw)
    
    # If the original dataframe had categorical features, names change after OneHotEncoder
    # To be safe, we generate dummy names and attempt to preserve known numerical ones (like recovery_score)
    proc_feature_names = num_cols + [f"cat_{i}" for i in range(X_proc_array.shape[1] - len(num_cols))]
    
    df_proc = pd.DataFrame(X_proc_array, columns=proc_feature_names)
    df_proc['athlete_id'] = athlete_ids
    df_proc['session_id'] = session_ids
    df_proc['injury_occurred'] = targets
    
    print("2. Generating 3D Tensors (Sliding Windows)...")
    X_t, y_t = create_sliding_windows(df_proc, SEQ_LENGTH, proc_feature_names, 'injury_occurred')
    
    print("3. Initializing LSTM Model and loading saved weights...")
    input_dim = X_t.shape[2]
    model = InjuryPredictionLSTM(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_layers=1, dropout_rate=0.3).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
        model.eval()
        print("   Weights loaded successfully!")
    except Exception as e:
        print(f" [ERROR] Could not load {MODEL_WEIGHTS_PATH}. Did you run the mega training?\nDetails: {e}")
        exit()

    print("\n4. Generating global Confusion Matrix...")
    # Run inference on the entire dataset to create the matrix
    test_loader = DataLoader(SportsInjuryDataset(X_t, y_t), batch_size=64, shuffle=False)
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X.to(DEVICE))
            preds = torch.max(outputs, 1)[1]
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            
    plot_clinical_confusion_matrix(all_targets, all_preds)
    
    print("\n5. Generating Risk Trajectories for specific athletes...")
    # Find a couple of athletes who actually sustained an injury (Class 2)
    # to get interesting plots
    injured_athletes = df[df['injury_occurred'] == 2]['athlete_id'].unique()
    
    if len(injured_athletes) > 0:
        # Generate the plot for the first two injured athletes found
        for i, ath_id in enumerate(injured_athletes[:2]):
            save_name = f"risk_trajectory_athlete_{ath_id}.png"
            plot_athlete_risk_trajectory(model, ath_id, df_proc, SEQ_LENGTH, proc_feature_names, DEVICE, save_name)
    else:
        print("No Class 2 injuries found in the dataset. Generating plot for a random athlete.")
        plot_athlete_risk_trajectory(model, df['athlete_id'].iloc[0], df_proc, SEQ_LENGTH, proc_feature_names, DEVICE, "risk_trajectory_random.png")
        
    print("\nOperation completed! Check your working directory for the PNG files.")
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
import re
from pathlib import Path
import sys

warnings.filterwarnings('ignore')

# 1. Custom module imports
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from LSTM.sliding_windows import create_sliding_windows, SportsInjuryDataset
from model import InjuryPredictionGRU, InjuryPredictionTransformer

# ==========================================
# GLOBAL CONFIGURATIONS
# ==========================================
DEFAULT_SEQ_LENGTH = 7
DATASET_PATH = ROOT_DIR / "multimodal_sports_injury_dataset.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SETTINGS = {
    "gru": {
        "display_name": "GRU",
        "checkpoint_candidates": [
            CURRENT_DIR / "GRU_results" / "best_gru_mega_model.pth",
        ],
        "grid_report_candidates": [
            CURRENT_DIR / "GRU_results" / "gru_grid_search_results.txt",
        ],
    },
    "transformer": {
        "display_name": "Transformer",
        "checkpoint_candidates": [
            CURRENT_DIR / "Transformers_results" / "best_transformer_mega_model.pth",
        ],
        "grid_report_candidates": [
            CURRENT_DIR / "Transformers_results" / "transformer_grid_search_results.txt",
        ],
    },
}

print(f"Starting plot generation on device: {DEVICE}")

# ==========================================
# PLOTTING FUNCTIONS
# ==========================================
def plot_clinical_confusion_matrix(y_true, y_pred, model_name, save_path):
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
    
    plt.title(f'Clinical Confusion Matrix - {model_name} System', fontsize=14, pad=20, fontweight='bold')
    plt.ylabel('Actual Athlete Condition', fontsize=12, fontweight='bold')
    plt.xlabel(f'{model_name} System Prediction', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" => Saved: {save_path}")


def plot_athlete_risk_trajectory(model, model_name, athlete_id, df_proc, seq_length, feature_cols, device, save_path):
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
    line1, = ax1.plot(days, risk_probabilities, color=color1, marker='o', linewidth=3, markersize=8, label=f'Risk Probability ({model_name})')
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
    
    plt.title(f'Injury Risk Temporal Trajectory ({model_name}) - Athlete ID: {athlete_id}', fontsize=15, pad=20, fontweight='bold')
    
    # Reduce X-axis labels if there are too many
    if len(days) > 15:
        ax1.set_xticks(np.arange(0, len(days), step=int(len(days)/10)))
        
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" => Saved: {save_path}")


def find_checkpoint(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


def parse_best_config(grid_report_candidates):
    for report_path in grid_report_candidates:
        if not report_path.exists():
            continue

        text = report_path.read_text(encoding='utf-8', errors='ignore')
        seq_match = re.search(r"seq_length'\s*:\s*(\d+)", text)
        hidden_match = re.search(r"hidden_dim'\s*:\s*(\d+)", text)

        seq_length = int(seq_match.group(1)) if seq_match else DEFAULT_SEQ_LENGTH
        hidden_dim = int(hidden_match.group(1)) if hidden_match else None
        return seq_length, hidden_dim

    return DEFAULT_SEQ_LENGTH, None


def infer_hidden_from_state_dict(model_key, state_dict):
    if model_key == "lstm" and "lstm.weight_ih_l0" in state_dict:
        return state_dict["lstm.weight_ih_l0"].shape[0] // 4
    if model_key == "gru" and "gru.weight_ih_l0" in state_dict:
        return state_dict["gru.weight_ih_l0"].shape[0] // 3
    if model_key == "transformer" and "input_projection.weight" in state_dict:
        return state_dict["input_projection.weight"].shape[0]
    return None


def load_checkpoint_state_dict(checkpoint_path):
    try:
        checkpoint_obj = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        checkpoint_obj = torch.load(checkpoint_path, map_location=DEVICE)

    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        return checkpoint_obj["state_dict"]
    return checkpoint_obj


def build_model(model_key, input_dim, hidden_dim):
    if model_key == "lstm":
        return InjuryPredictionLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=1, dropout_rate=0.3).to(DEVICE)
    if model_key == "gru":
        return InjuryPredictionGRU(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=1, dropout_rate=0.3).to(DEVICE)
    if model_key == "transformer":
        return InjuryPredictionTransformer(input_dim=input_dim, d_model=hidden_dim, nhead=4, num_layers=1).to(DEVICE)
    raise ValueError(f"Unsupported model key: {model_key}")


def preprocess_full_dataset(df):
    athlete_ids = df['athlete_id']
    session_ids = df['session_id']
    targets = df['injury_occurred']

    X_raw = df.drop(['athlete_id', 'session_id', 'injury_occurred'], axis=1)

    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_raw.select_dtypes(include=['object']).columns.tolist()

    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', MinMaxScaler())])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])

    X_proc_array = preprocessor.fit_transform(X_raw)

    proc_feature_names = num_cols + [f"cat_{i}" for i in range(X_proc_array.shape[1] - len(num_cols))]

    df_proc = pd.DataFrame(X_proc_array, columns=proc_feature_names)
    df_proc['athlete_id'] = athlete_ids
    df_proc['session_id'] = session_ids
    df_proc['injury_occurred'] = targets

    return df_proc, proc_feature_names


def run_model_plots(model_key, model_cfg, df, df_proc, proc_feature_names):
    model_name = model_cfg["display_name"]

    checkpoint_path = find_checkpoint(model_cfg["checkpoint_candidates"])
    if checkpoint_path is None:
        print(f" [WARNING] Checkpoint not found for {model_name}. Skipping.")
        return

    seq_length, hidden_dim_report = parse_best_config(model_cfg["grid_report_candidates"])

    X_t, y_t = create_sliding_windows(df_proc, seq_length, proc_feature_names, 'injury_occurred')
    if len(X_t) == 0:
        print(f" [WARNING] No sliding windows generated for {model_name} (seq_length={seq_length}). Skipping.")
        return

    input_dim = X_t.shape[2]
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    hidden_dim_ckpt = infer_hidden_from_state_dict(model_key, state_dict)

    if hidden_dim_ckpt is not None:
        if hidden_dim_report is not None and hidden_dim_report != hidden_dim_ckpt:
            print(
                f" [WARNING] Hidden dim mismatch for {model_name}: "
                f"report={hidden_dim_report}, checkpoint={hidden_dim_ckpt}. "
                f"Using checkpoint value."
            )
        hidden_dim = hidden_dim_ckpt
    else:
        hidden_dim = hidden_dim_report

    if hidden_dim is None:
        print(f" [WARNING] Could not infer hidden dimension for {model_name}. Skipping.")
        return

    print(f"\n3. [{model_name}] Loading checkpoint: {checkpoint_path}")
    print(f"   Using seq_length={seq_length}, hidden_dim={hidden_dim}")

    model = build_model(model_key, input_dim, hidden_dim)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"4. [{model_name}] Generating global Confusion Matrix...")
    test_loader = DataLoader(SportsInjuryDataset(X_t, y_t), batch_size=64, shuffle=False)

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X.to(DEVICE))
            preds = torch.max(outputs, 1)[1]
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    plot_clinical_confusion_matrix(
        all_targets,
        all_preds,
        model_name=model_name,
        save_path=CURRENT_DIR / f"confusion_matrix_{model_key}.png"
    )

    print(f"5. [{model_name}] Generating Risk Trajectories for specific athletes...")
    injured_athletes = df[df['injury_occurred'] == 2]['athlete_id'].unique()

    if len(injured_athletes) > 0:
        for ath_id in injured_athletes[:2]:
            save_name = CURRENT_DIR / f"risk_trajectory_{model_key}_athlete_{ath_id}.png"
            plot_athlete_risk_trajectory(model, model_name, ath_id, df_proc, seq_length, proc_feature_names, DEVICE, save_name)
    else:
        print("No Class 2 injuries found in the dataset. Generating plot for a random athlete.")
        plot_athlete_risk_trajectory(
            model,
            model_name,
            df['athlete_id'].iloc[0],
            df_proc,
            seq_length,
            proc_feature_names,
            DEVICE,
            CURRENT_DIR / f"risk_trajectory_{model_key}_random.png"
        )


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":

    print("1. Loading data and applying global Preprocessing...")
    df = pd.read_csv(DATASET_PATH)

    df_proc, proc_feature_names = preprocess_full_dataset(df)

    for model_key, model_cfg in MODEL_SETTINGS.items():
        run_model_plots(model_key, model_cfg, df, df_proc, proc_feature_names)

    print("\nOperation completed! Check the OTHER folder for generated PNG files.")
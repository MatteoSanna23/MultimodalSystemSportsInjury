import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, fbeta_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import moduli custom
from model import InjuryPredictionTransformer
from sliding_windows import create_sliding_windows, SportsInjuryDataset

# Configurazione
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 32 # Corretto per allinearsi al peso salvato
SEQ_LENGTH = 7 

# 1. Caricamento Dataset (con controllo nomi colonne)
df = pd.read_csv('synthetic_unseen_test_dataset.csv')
required_cols = ['athlete_id', 'session_id', 'injury_occurred']
feature_cols = [c for c in df.columns if c not in required_cols]

# 2. Preprocessing
num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', MinMaxScaler())])
cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer(transformers=[('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])

X_proc = preprocessor.fit_transform(df[feature_cols])
proc_feat_names = [f"f_{i}" for i in range(X_proc.shape[1])]
df_proc = pd.DataFrame(X_proc, columns=proc_feat_names, index=df.index)
df_proc[required_cols] = df[required_cols]

# 3. Sliding Windows
X_test, y_test = create_sliding_windows(df_proc, SEQ_LENGTH, proc_feat_names, 'injury_occurred')
loader = DataLoader(SportsInjuryDataset(X_test, y_test), batch_size=32)

# 4. Caricamento Modello
model = InjuryPredictionTransformer(input_dim=X_test.shape[2], d_model=HIDDEN_DIM, nhead=4, num_layers=1).to(DEVICE)
model.load_state_dict(torch.load("best_transformer_mega_model.pth", map_location=DEVICE))
model.eval()

# 5. Inferenza
all_preds, all_targets = [], []
with torch.no_grad():
    for batch_X, batch_y in loader:
        logits = model(batch_X.to(DEVICE))
        all_preds.extend(torch.max(logits, 1)[1].cpu().numpy())
        all_targets.extend(batch_y.numpy())

# 6. Calcolo Metriche F2 (Class 2 Target)
# labels=[0, 1, 2] indica che vogliamo la metrica per ciascuna classe. [2] estrae solo la classe High Risk.
f2_c2 = fbeta_score(all_targets, all_preds, beta=2.0, average=None, labels=[0, 1, 2])[2]
recall_c2 = recall_score(all_targets, all_preds, average=None, labels=[0, 1, 2])[2]

print("\n--- PERFORMANCE SU DATASET UNSEEN (Target: Class 2) ---")
print(f"F2-Score (Class 2): {f2_c2:.4f}")
print(f"Recall (Class 2):   {recall_c2:.4f}")

# Salvataggio report
with open("report_performance_unseen.txt", "w") as f:
    f.write(f"--- PERFORMANCE SU DATASET UNSEEN ---\n")
    f.write(f"F2-Score (Class 2): {f2_c2:.4f}\n")
    f.write(f"Recall (Class 2):   {recall_c2:.4f}\n")

# Matrice di Confusione
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Low', 'High'], yticklabels=['Healthy', 'Low', 'High'])
plt.title(f"Conf. Matrix Unseen - F2_C2: {f2_c2:.3f}")
plt.ylabel("Reale")
plt.xlabel("Predetto")
plt.savefig("confusion_matrix_unseen.png")
print("Report e matrice salvati.")
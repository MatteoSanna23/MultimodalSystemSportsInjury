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
from sklearn.metrics import fbeta_score, recall_score, classification_report
import warnings

warnings.filterwarnings('ignore')

# 1. Importiamo i vostri moduli custom!
from lstm import InjuryPredictionLSTM
from sliding_windows import create_sliding_windows, SportsInjuryDataset

# ==========================================
# CONFIGURAZIONI GLOBALI
# ==========================================
SEQ_LENGTH = 5              # Finestra di 5 sessioni (Microciclo)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Utilizzando il device: {DEVICE}")

# ==========================================
# CARICAMENTO DATI
# ==========================================
df = pd.read_csv('..//multimodal_sports_injury_dataset.csv')
X = df.drop(['injury_occurred'], axis=1) # Teniamo momentaneamente athlete_id e session_id per il grouping
y = df['injury_occurred']
groups = df['athlete_id']

gkf = GroupKFold(n_splits=5)

# Array per tracciare le metriche sui fold
fold_f2_c2 = []
fold_rec_c2 = []

print("\n" + "="*60)
print(f"INIZIO TRAINING LSTM (Time Steps: {SEQ_LENGTH}) - Validazione Subject-Wise")
print("="*60)

# ==========================================
# START
# ==========================================
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
    print(f"\n--- FOLD {fold} ---")
    
    # 1. Split base
    X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
    y_train_raw, y_test_raw = y.iloc[train_idx], y.iloc[test_idx]
    
    # Isolare le colonne vere (escludendo gli ID) per il preprocessing
    feature_cols_to_scale = X_train_raw.drop(['athlete_id', 'session_id'], axis=1).columns.tolist()
    num_cols = X_train_raw[feature_cols_to_scale].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train_raw[feature_cols_to_scale].select_dtypes(include=['object']).columns.tolist()
    
    # 2. Pipeline Vincente del Blocco 1 (Median + MinMaxScaler)
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', MinMaxScaler())])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
    
    # Fittiamo SOLO sul train
    X_train_proc_array = preprocessor.fit_transform(X_train_raw[feature_cols_to_scale])
    X_test_proc_array = preprocessor.transform(X_test_raw[feature_cols_to_scale])
    
    # 3. Ri-assemblaggio dei DataFrame per la funzione sliding_windows
    # Creiamo nomi fittizi per le colonne trasformate
    proc_feature_names = [f"f_{i}" for i in range(X_train_proc_array.shape[1])]
    df_train_proc = pd.DataFrame(X_train_proc_array, columns=proc_feature_names, index=X_train_raw.index)
    # questo DataFrame attualmente 
    df_train_proc['athlete_id'] = X_train_raw['athlete_id']
    df_train_proc['session_id'] = X_train_raw['session_id']
    df_train_proc['injury_occurred'] = y_train_raw
    
    df_test_proc = pd.DataFrame(X_test_proc_array, columns=proc_feature_names, index=X_test_raw.index)
    df_test_proc['athlete_id'] = X_test_raw['athlete_id']
    df_test_proc['session_id'] = X_test_raw['session_id']
    df_test_proc['injury_occurred'] = y_test_raw
    
    # 4. Creazione Tensori 3D (Finestre Scorrevoli)
    X_train_t, y_train_t = create_sliding_windows(df_train_proc, SEQ_LENGTH, proc_feature_names, 'injury_occurred')
    X_test_t, y_test_t = create_sliding_windows(df_test_proc, SEQ_LENGTH, proc_feature_names, 'injury_occurred')
    
    # Creazione DataLoaders PyTorch, DataLoader che da definizione servono a gestire i batch e lo shuffle dei dati
    train_dataset = SportsInjuryDataset(X_train_t, y_train_t)
    test_dataset = SportsInjuryDataset(X_test_t, y_test_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 5. Calcolo ClassWeight dinamico per PyTorch
    # PyTorch vuole i pesi direttamente nella Loss Function come tensore
    classes = np.unique(y_train_t.numpy())
    weights = compute_class_weight('balanced', classes=classes, y=y_train_t.numpy())
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    
    # 6. Inizializzazione Modello, Loss e Ottimizzatore
    input_dim = X_train_t.shape[2] # Numero di feature elaborate
    model = InjuryPredictionLSTM(input_dim=input_dim, hidden_dim=64, num_layers=1, dropout_rate=0.3).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 7. Ciclo di Addestramento (Epochs)
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X) # outputs sono i raw logits
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Stampa progressi ogni 10 epoche per non intasare il terminale
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}")
            
    # 8. Valutazione sul Test Set (Fold corrente)
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(DEVICE)
            outputs = model(batch_X)
            # Prendiamo la classe con la probabilità (logit) più alta
            _, preds = torch.max(outputs, 1) 
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            
    # Calcolo Metriche ($F_2$ Score e Recall per la Classe 2)
    fold_f2 = fbeta_score(all_targets, all_preds, beta=2.0, average=None, labels=[0, 1, 2])[2]
    fold_rec = recall_score(all_targets, all_preds, average=None, labels=[0, 1, 2])[2]
    
    fold_f2_c2.append(fold_f2)
    fold_rec_c2.append(fold_rec)
    
    print(f"   => F2 Classe 2 (Fold {fold}): {fold_f2:.4f} | Recall C2: {fold_rec:.4f}")

# ==========================================
# RISULTATI FINALI
# ==========================================
print("\n" + "="*60)
print("RISULTATI FINALI LSTM (Medie sui 5 Fold)")
print("="*60)
print(f"F2 Score Medio (Classe 2): {np.mean(fold_f2_c2):.4f} +/- {np.std(fold_f2_c2):.4f}")
print(f"Recall Media (Classe 2): {np.mean(fold_rec_c2):.4f} +/- {np.std(fold_rec_c2):.4f}")
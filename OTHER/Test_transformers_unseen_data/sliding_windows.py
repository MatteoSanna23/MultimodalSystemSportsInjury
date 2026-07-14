import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def create_sliding_windows(df, seq_length, feature_cols, target_col):
    """
    Trasforma il dataset tabellare 2D in tensori 3D per la LSTM.
    Raggruppa per atleta e rispetta l'ordine cronologico.
    """
    X_list = []
    y_list = []
    
    # Raggruppiamo per atleta in modo da non accavallare finestre tra persone diverse
    for athlete_id, group in df.groupby('athlete_id'):
        # FONDAMENTALE: Ordinare per session_id (cronologia)
        group = group.sort_values('session_id')
        
        features = group[feature_cols].values   # Valori delle feature per l'atleta
        targets = group[target_col].values  # Valori dei target per l'atleta
        
        # Facciamo scorrere la finestra
        for i in range(len(group) - seq_length + 1):
            # Estraiamo N sessioni consecutive
            window_x = features[i : i + seq_length]
            # Il target è l'infortunio (o meno) nell'ultima sessione della finestra
            window_y = targets[i + seq_length - 1] 
            
            X_list.append(window_x)
            y_list.append(window_y)
            
    # Convertiamo in tensori PyTorch (Float per X, Long/Int per le classi y)
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_list), dtype=torch.long)
    
    return X_tensor, y_tensor

# Classe Dataset standard per PyTorch
class SportsInjuryDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
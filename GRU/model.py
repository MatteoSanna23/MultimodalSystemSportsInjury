import torch
import torch.nn as nn
import math

# ==========================================
# 1. GATED RECURRENT UNIT (GRU)
# ==========================================
class InjuryPredictionGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, output_dim=3, dropout_rate=0.3):
        super(InjuryPredictionGRU, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # gru_out shape: (batch, seq_len, hidden_dim)
        gru_out, hn = self.gru(x)
        # Estraiamo l'ultimo time step
        last_time_step = gru_out[:, -1, :]
        out = self.dropout(last_time_step)
        return self.fc(out)
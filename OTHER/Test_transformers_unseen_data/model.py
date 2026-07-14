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


# ==========================================
# 2. TIME-SERIES TRANSFORMER ENCODER
# ==========================================
class PositionalEncoding(nn.Module):
    """Inietta informazioni sull'ordine cronologico dei giorni nella sequenza."""
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

class InjuryPredictionTransformer(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=1, output_dim=3, dropout_rate=0.3):
        super(InjuryPredictionTransformer, self).__init__()
        
        # Proiettiamo le feature di input nello spazio dimensionale del Transformer (d_model)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2, 
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # 1. Proiezione lineare + Codifica Posizionale
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # 2. Passaggio nei layer di Self-Attention
        x = self.transformer_encoder(x)
        
        # 3. Pooling (Prendiamo la rappresentazione media della sequenza o l'ultimo giorno)
        # In ambito clinico, la media dei pesi di attenzione descrive l'intera settimana
        x = torch.mean(x, dim=1) 
        
        x = self.dropout(x)
        return self.fc(x)
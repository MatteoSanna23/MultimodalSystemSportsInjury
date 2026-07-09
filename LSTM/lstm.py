import torch
import torch.nn as nn

class InjuryPredictionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, output_dim=3, dropout_rate=0.3):
        super(InjuryPredictionLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Il layer temporale
        # batch_first=True significa che ci aspetta (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Livello di regolarizzazione per combattere l'overfitting
        self.dropout = nn.Dropout(dropout_rate)
        
        # Livello finale di classificazione (3 Nodi: Classe 0, 1, 2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x.shape = (batch_size, seq_length, num_features)
        
        # lstm_out contiene gli output di TUTTI i time step
        # hn contiene la memoria (hidden state) dell'ultimo step
        lstm_out, (hn, cn) = self.lstm(x)
        
        # A noi interessa solo cosa ha imparato la rete alla FINE della finestra (l'ultimo giorno)
        last_time_step = lstm_out[:, -1, :] 
        
        # Passiamo l'ultimo stato attraverso il dropout e poi al layer lineare
        out = self.dropout(last_time_step)
        out = self.fc(out)
        
        return out
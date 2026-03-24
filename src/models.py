import xgboost as xgb
import torch
import torch.nn as nn

def get_xgboost_model():
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
    )
    return model

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        
        out, _ = self.lstm(x)  # out: (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])  # Lấy output của bước thời gian cuối cùng
        return out
    
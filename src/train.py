import os
import json
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from tqdm import tqdm

from models import get_xgboost_model, LSTMModel
from data_loader import load_from_csv
from features import build_features

# Hàm tạo dữ liệu chuỗi cho LSTM
def create_sequence_data(X: np.ndarray, y: np.ndarray, seq_length: int):
    X_seq, y_seq = [], []
    for i in range(seq_length, len(X)):
        X_seq.append(X[i-seq_length:i])
        y_seq.append(y[i])
        
    return np.array(X_seq), np.array(y_seq)

# Hàm chia dữ liệu thành tập train và test theo thứ tự thời gian
def time_series_train_test_split(X: np.array, y: np.array, train_size = 0.7, test_size=0.15):
    train_idx = int(len(X) * train_size)
    val_idx = int(len(X) * (train_size + test_size))
    
    X_train = X[:train_idx]
    X_val = X[train_idx:val_idx]
    X_test = X[val_idx:]
    
    y_train = y[:train_idx]
    y_val = y[train_idx:val_idx]
    y_test = y[val_idx:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Hàm lưu metrics vào file JSON
def save_metrics(metrics: dict, path: str = "results/metrics.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
        
def main():
    
    # Kích hoạt GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dữ liệu
    df = load_from_csv("data/btc_data.csv")
    
    # Tạo feature 
    df_features = build_features(df)
    
    # Chọn feature 
    feature_cols = [
        "log_returns",
        "volume",
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_14",
        "rolling_std_14",
        "momentum_3",
        "momentum_7",
        "momentum_14"
    ]
    
    # Tạo target
    target_cols= "target"
    
    X = df_features[feature_cols].values 
    y = df_features[target_cols].values
    
    # Tạo sequence data cho LSTM
    seq_length = 20
    X_seq, y_seq = create_sequence_data(X, y, seq_length)
    print(f"Shape of sequence data: X_seq={X_seq.shape}, y_seq={y_seq.shape}")
    
    # Chia dữ liệu thành train, test
    X_train, X_val, X_test, y_train, y_val, y_test = time_series_train_test_split(X_seq, y_seq)
    print(f"Shape of train data: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Shape of validation data: X_val={X_val.shape}, y_val={y_val.shape}")
    print(f"Shape of test data: X_test={X_test.shape}, y_test={y_test.shape}")

    # convert sang Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1,1)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1,1)
    
    # Tạo DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    
    # model
    lstm_model = LSTMModel(
        input_size=X_train.shape[2], 
        hidden_size=32, 
        num_layers=2, 
        output_size=1,
        dropout=0.2
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(lstm_model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    epochs = 20
    best_val_loss = float("inf")
    for epoch in range(epochs):
        lstm_model.train()
        epoch_loss = 0.0
        
        train_loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} Train", leave=False)
        
        for X_batch, y_batch in train_dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = lstm_model(X_batch)
            loss= criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
            
        avg_loss = epoch_loss / len(train_dataloader)
        
        # validation 
        lstm_model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_dataloader:
                X_val_batch = X_val_batch.to(device)
                y_val_batch = y_val_batch.to(device)
                
                val_outputs = lstm_model(X_val_batch)
                val_loss = criterion(val_outputs, y_val_batch)
                val_epoch_loss += val_loss.item()
        
        avg_val_loss = val_epoch_loss / len(val_dataloader)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")


        # Lưu checkpoint tốt nhất
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(lstm_model.state_dict(), "results/best_lstm_model.pth")
            print(f"Best model saved with val loss: {best_val_loss:.4f}")

        
if __name__ == "__main__":
    main()


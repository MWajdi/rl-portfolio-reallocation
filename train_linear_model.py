import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

def load_data_from_json(json_file): 
    """
    Loads the JSON file created by `train_data_generation` and returns
    a sorted list of (timestamp_dt, X_window, y_val).
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    
    records = []
    for ts_str, (x_window, y_val) in data.items():
        dt = datetime.fromisoformat(ts_str)  # Convert string to datetime
        records.append((dt, x_window, y_val))
    
    # Sort by datetime
    records.sort(key=lambda r: r[0])

    # Extract X and y
    X_all = []
    y_all = []
    
    for _, x_win, y_val in records:
        X_all.append(x_win)
        y_all.append(y_val)
    
    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32)

    return X_all, y_all

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        # Return (features, target)
        return self.X[index], self.y[index]


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, model_path="models/linear_model.pth"):
    # 1) Choose a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2) Move the model to GPU (if available)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(1, epochs + 1):
        # ---- TRAINING PHASE ----
        model.train()
        train_losses = []
        
        for batch_x, batch_y in train_loader:
            # 3) Move batch to the same device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            pred = model(batch_x).squeeze(1)
            loss = F.mse_loss(pred, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        train_loss_mean = np.mean(train_losses)
        
        # ---- VALIDATION PHASE ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                pred = model(batch_x).squeeze(1)
                loss = F.mse_loss(pred, batch_y)
                val_losses.append(loss.item())
        
        val_loss_mean = np.mean(val_losses)
        
        train_loss_history.append(train_loss_mean)
        val_loss_history.append(val_loss_mean)

        print(f"[Epoch {epoch}/{epochs}] "
              f"Train MSE: {train_loss_mean:.4f} | "
              f"Val MSE: {val_loss_mean:.4f}")
    
    # Save model weights
    torch.save(model.state_dict(), model_path)

    # --- Two separate plots (log scale) ---
    plt.figure()
    plt.plot(range(1, epochs + 1), train_loss_history)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Train Loss (Log Scale) - {model_path}")
    plt.show()

    plt.figure()
    plt.plot(range(1, epochs + 1), val_loss_history)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Eval Loss (Log Scale) - {model_path}")
    plt.show()
    
    return model



class Args:
    def __init__(self,
    model_path: str = "models/linear_model.pth",
    train_path: str = "binance_datasets/BTC_train_data.json",
    test_path: str = "binance_datasets/BTC_test_data.json",
    epochs: int = 10,
    lr: float = 1e-3
    ):
        self.model_path = model_path
        self.train_path = train_path
        self.test_path = test_path
        self.epochs = epochs
        self.lr = lr

        

def main(args):
    # Load train and test data from json
    X_train, y_train = load_data_from_json(args.train_path)
    X_test, y_test = load_data_from_json(args.test_path)


    # Create PyTorch datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    batch_size = 256
    
    # Create PyTorch data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define the linear model
    input_dim = X_train.shape[1]
    model = nn.Linear(in_features=input_dim, out_features=1)
    
    # Train the model
    trained_model = train_model(model, train_loader, test_loader, args.epochs, args.lr, args.model_path)
    
    return trained_model
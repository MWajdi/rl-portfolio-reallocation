import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from torch.utils.data import WeightedRandomSampler
import gc

def load_data_from_json(json_file): 
    """
    Loads the JSON file created by `train_data_generation` and returns
    the data for a binary LSTM model (up/down prediction).
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
    
    # For sequential data, we need the full windows
    # For binary classification, we convert y to binary (1 if price goes up, 0 if down)
    for i in range(len(records) - 1):
        _, x_win, _ = records[i]
        _, _, next_price = records[i + 1]
        current_price = x_win[-1]  # Last value in the window is current price
        
        # Binary label: 1 if price goes up, 0 if down
        binary_label = 1 if next_price > current_price else 0
        
        X_all.append(x_win)
        y_all.append(binary_label)
    
    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32)
    
    # Reshape X for LSTM [batch, seq_len, features]
    # If X is just a sequence of prices, add feature dimension
    if len(X_all.shape) == 2:
        X_all = X_all.reshape(X_all.shape[0], X_all.shape[1], 1)

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

class BtcLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.3):
        super(BtcLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Add batch normalization for better training stability
        self.bn = nn.BatchNorm1d(hidden_size)
        
        # Multiple fully connected layers with dropout
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Get the last time step output
        out = out[:, -1, :]
        
        # Apply batch normalization
        out = self.bn(out)
        
        # Multiple layers with dropout
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, model_path="models/lstm_model.pth"):
    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)

    # Binary cross-entropy loss for binary classification
    criterion = nn.BCELoss()
    
    # Adam optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping parameters
    patience = 10
    best_val_loss = float('inf')
    counter = 0
    early_stop = False
    
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    for epoch in range(1, epochs + 1):
        if early_stop:
            print("Early stopping triggered")
            break
            
        # ---- TRAINING PHASE ----
        model.train()
        train_losses = []
        train_preds = []
        train_true = []
        
        for batch_x, batch_y in train_loader:
            # Move batch to device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            pred = model(batch_x).squeeze(1)
            loss = criterion(pred, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Store predictions and true values for accuracy calculation
            train_preds.extend(pred.cpu().detach().numpy())
            train_true.extend(batch_y.cpu().detach().numpy())
            
            # Clear memory
            del batch_x, batch_y, pred, loss
            torch.cuda.empty_cache()
        
        train_loss_mean = np.mean(train_losses)
        train_preds_class = (np.array(train_preds) > 0.5).astype(int)
        train_accuracy = accuracy_score(train_true, train_preds_class)
        
        # ---- VALIDATION PHASE ----
        model.eval()
        val_losses = []
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                pred = model(batch_x).squeeze(1)
                loss = criterion(pred, batch_y)
                
                val_losses.append(loss.item())
                
                # Store predictions and true values for accuracy calculation
                val_preds.extend(pred.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())
                
                # Clear memory
                del batch_x, batch_y, pred, loss
                torch.cuda.empty_cache()
        
        val_loss_mean = np.mean(val_losses)
        val_preds_class = (np.array(val_preds) > 0.5).astype(int)
        val_accuracy = accuracy_score(val_true, val_preds_class)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss_mean)
        
        # Append history
        train_loss_history.append(train_loss_mean)
        val_loss_history.append(val_loss_mean)
        train_acc_history.append(train_accuracy)
        val_acc_history.append(val_accuracy)

        print(f"[Epoch {epoch}/{epochs}] "
              f"Train Loss: {train_loss_mean:.4f} | "
              f"Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss_mean:.4f} | "
              f"Val Acc: {val_accuracy:.4f}")
        
        # Early stopping check
        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            counter = 0
            # Save the best model
            torch.save(model.state_dict(), model_path)
        else:
            counter += 1
            if counter >= patience:
                early_stop = True
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
    
    # --- Plot loss (log scale) ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Train')
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation')
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title(f"Loss (Log Scale)")
    plt.legend()
    
    # --- Plot accuracy ---
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_acc_history) + 1), train_acc_history, label='Train')
    plt.plot(range(1, len(val_acc_history) + 1), val_acc_history, label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Load the best model
    best_model = BtcLSTM(
        input_size=model.lstm.input_size,
        hidden_size=model.hidden_size,
        num_layers=model.num_layers,
        dropout=model.dropout1.p
    ).to(device)
    best_model.load_state_dict(torch.load(model_path))
    
    return best_model

def evaluate_model(model, test_loader, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            
            pred = model(batch_x).squeeze(1)
            
            all_preds.extend(pred.cpu().numpy())
            all_true.extend(batch_y.numpy())
            
            # Clear memory
            del batch_x, batch_y, pred
            torch.cuda.empty_cache()
    
    # Convert predictions to binary classes
    all_preds_class = (np.array(all_preds) > threshold).astype(int)
    all_true = np.array(all_true)
    
    # Calculate metrics
    accuracy = accuracy_score(all_true, all_preds_class)
    report = classification_report(all_true, all_preds_class)
    roc_auc = roc_auc_score(all_true, all_preds)
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}')
    print('Classification Report:')
    print(report)
    
    return accuracy, roc_auc, all_preds, all_true

class Args:
    def __init__(self,
    model_path: str = "models/lstm_model.pth",
    train_path: str = "binance_datasets/BTC_train_data.json",
    test_path: str = "binance_datasets/BTC_test_data.json",
    epochs: int = 50,
    lr: float = 1e-3,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    batch_size: int = 64
    ):
        self.model_path = model_path
        self.train_path = train_path
        self.test_path = test_path
        self.epochs = epochs
        self.lr = lr
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size

def main(args):
    # Load train and test data from json
    X_train, y_train = load_data_from_json(args.train_path)
    X_test, y_test = load_data_from_json(args.test_path)
    
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    print(f"Positive samples in train: {sum(y_train)}/{len(y_train)} ({sum(y_train)/len(y_train)*100:.2f}%)")
    print(f"Positive samples in test: {sum(y_test)}/{len(y_test)} ({sum(y_test)/len(y_test)*100:.2f}%)")

    # Create PyTorch datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    # Calculate class weights for addressing imbalance
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train.astype(int)]
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    # Create PyTorch data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=sampler
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    # Define the LSTM model
    input_size = X_train.shape[2]  # Number of features
    model = BtcLSTM(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    print(f"Created LSTM model with input_size={input_size}, hidden_size={args.hidden_size}, num_layers={args.num_layers}")
    
    # Train the model
    trained_model = train_model(
        model, 
        train_loader, 
        test_loader, 
        args.epochs, 
        args.lr, 
        args.model_path
    )
    
    # Evaluate the model
    print("\nEvaluating model on test data:")
    evaluate_model(trained_model, test_loader)
    
    return trained_model

if __name__ == "__main__":
    args = Args()
    main(args)

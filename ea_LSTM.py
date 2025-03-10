import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# 1. Data Preprocessing
def preprocess_data(data_path):
    """
    Preprocess Bitcoin data for LSTM model.
    
    Args:
        data_path (str): Path to Bitcoin data JSON file
        
    Returns:
        tuple: (train_X, train_y, val_X, val_y, test_X, test_y, scaler)
    """
    # Load data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Convert data to sequences and targets
    timestamps = list(data.keys())
    # Sort timestamps to ensure chronological order
    timestamps.sort()
    
    sequences = []
    targets = []
    
    for timestamp in timestamps:
        # Each entry has format [sequence_array, target_value]
        sequence = data[timestamp][0]
        target = data[timestamp][1]
        
        sequences.append(sequence)
        targets.append(target)
    
    # Convert to numpy arrays
    sequences = np.array(sequences, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    
    # Normalize data
    # We need to fit the scaler on both sequences and targets to ensure consistent scaling
    # Combine all data points to fit the scaler
    all_prices = np.concatenate([sequences.flatten(), targets])
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(all_prices.reshape(-1, 1))
    
    # Scale sequences and targets using the same scaler
    sequences_scaled = np.array([scaler.transform(seq.reshape(-1, 1)).flatten() for seq in sequences])
    targets_scaled = scaler.transform(targets.reshape(-1, 1)).flatten()
    
    print(f"Original price range: [{np.min(all_prices):.2f}, {np.max(all_prices):.2f}]")
    print(f"Normalized price range: [{np.min(sequences_scaled):.6f}, {np.max(sequences_scaled):.6f}]")
    
    # Calculate split indices with buffer to prevent data leakage
    total_samples = len(sequences_scaled)
    
    # Training data: from start to (70% - 100 timestamps)
    train_end_raw = int(0.7 * total_samples)
    train_end = max(0, train_end_raw - 100)  # Remove last 100 samples to prevent leakage
    
    # Validation data: from (70% - 100 timestamps) to (85% - 100 timestamps)
    val_end_raw = int(0.85 * total_samples)
    val_end = max(train_end, val_end_raw - 100)  # Remove last 100 samples to prevent leakage
    
    # Test data: from (85% - 100 timestamps) to end
    # The test set starts where validation ends (already adjusted to prevent leakage)
    
    # Split data chronologically to avoid data leakage
    train_X = sequences_scaled[:train_end]
    train_y = targets_scaled[:train_end]
    val_X = sequences_scaled[train_end:val_end]
    val_y = targets_scaled[train_end:val_end]
    test_X = sequences_scaled[val_end:]
    test_y = targets_scaled[val_end:]
    
    # Print split information
    print(f"Split information to prevent data leakage:")
    print(f"Total samples: {total_samples}")
    print(f"Train set: 0 to {train_end} (removed last 100 from 70% split)")
    print(f"Validation set: {train_end} to {val_end} (removed last 100 from 85% split)")
    print(f"Test set: {val_end} to {total_samples}")
    
    return train_X, train_y, val_X, val_y, test_X, test_y, scaler

# 2. LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, x):
        # Reshape input to [batch_size, sequence_length, input_size]
        batch_size = x.size(0)
        seq_length = x.size(1)
        x = x.view(batch_size, seq_length, 1)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# 3. Evaluation Function
def evaluate_model(model, X, y, scaler=None):
    """
    Evaluate the LSTM model.
    
    Args:
        model (nn.Module): LSTM model
        X (torch.Tensor): Input sequences
        y (torch.Tensor): Target values
        scaler (MinMaxScaler, optional): Scaler for transforming back to original scale
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    model.eval()
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        # Forward pass
        y_pred = model(X).squeeze()
        
        # Ensure dimensions match
        if y_pred.dim() == 0 and y.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
        
        # Calculate loss on normalized data
        loss = criterion(y_pred, y).item()
        
        # Convert tensors to numpy arrays
        y_pred_np = y_pred.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # Calculate metrics on normalized data
        mse_normalized = mean_squared_error(y_np, y_pred_np)
        rmse_normalized = np.sqrt(mse_normalized)
        mae_normalized = mean_absolute_error(y_np, y_pred_np)
        
        # Always calculate metrics on both normalized and original scales when scaler is available
        if scaler is not None:
            # Create copies to avoid modifying original arrays
            y_pred_original = y_pred_np.copy().reshape(-1, 1)
            y_original = y_np.copy().reshape(-1, 1)
            
            # Apply inverse transform
            y_pred_original = scaler.inverse_transform(y_pred_original).flatten()
            y_original = scaler.inverse_transform(y_original).flatten()
            
            # Calculate metrics on original scale
            mse_original = mean_squared_error(y_original, y_pred_original)
            rmse_original = np.sqrt(mse_original)
            mae_original = mean_absolute_error(y_original, y_pred_original)
            r2_original = r2_score(y_original, y_pred_original)
            
            # Calculate MAPE on original scale
            mask = y_original != 0
            mape_original = np.mean(np.abs((y_original[mask] - y_pred_original[mask]) / y_original[mask])) * 100
            
            return {
                'normalized_loss': loss,
                'normalized_rmse': rmse_normalized,
                'normalized_mae': mae_normalized,
                
                # Original scale metrics (what we really care about)
                'loss': mse_original,
                'rmse': rmse_original,
                'mae': mae_original,
                'r2': r2_original,
                'mape': mape_original,
                
                'predictions': y_pred_np,
                'original_predictions': y_pred_original,
                'original_targets': y_original
            }
        
        # If no scaler, just return normalized metrics
        r2_normalized = r2_score(y_np, y_pred_np)
        
        # Calculate MAPE on normalized data
        mask = y_np != 0
        if np.any(mask):
            mape_normalized = np.mean(np.abs((y_np[mask] - y_pred_np[mask]) / y_np[mask])) * 100
        else:
            mape_normalized = np.nan
        
        return {
            'loss': loss,
            'rmse': rmse_normalized,
            'mae': mae_normalized,
            'r2': r2_normalized,
            'mape': mape_normalized,
            'predictions': y_pred_np
        }

# 4. Training Function
def train_model(model, train_X, train_y, val_X, val_y, epochs=50, batch_size=32, learning_rate=0.001, scaler=None):
    """
    Train the LSTM model.
    
    Args:
        model (nn.Module): LSTM model
        train_X (np.array): Training sequences
        train_y (np.array): Training targets
        val_X (np.array): Validation sequences
        val_y (np.array): Validation targets
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        scaler (MinMaxScaler, optional): Scaler for inverse transform
        
    Returns:
        tuple: (model, history) - Trained model and training history
    """
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Convert numpy arrays to torch tensors
    train_X = torch.FloatTensor(train_X)
    train_y = torch.FloatTensor(train_y)
    val_X = torch.FloatTensor(val_X)
    val_y = torch.FloatTensor(val_y)
    
    # Create data loaders
    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize history dictionary
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': {'rmse': [], 'mae': [], 'r2': [], 'mape': []}
    }
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X).squeeze()
            
            # Ensure dimensions match
            if outputs.dim() == 0 and batch_y.dim() == 1:
                outputs = outputs.unsqueeze(0)
                
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_X.size(0)
        
        # Calculate average training loss (normalized scale)
        train_loss = running_loss / len(train_loader.dataset)
        
        # Evaluate on validation set
        val_metrics = evaluate_model(model, val_X, val_y, scaler)
        
        # Track normalized validation loss for monitoring training
        val_loss = val_metrics.get('normalized_loss', val_metrics['loss'])
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        for metric in ['rmse', 'mae', 'r2', 'mape']:
            history['val_metrics'][metric].append(val_metrics[metric])
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Print both normalized and original scale metrics
            normalized_rmse = val_metrics.get('normalized_rmse', val_metrics['rmse'])
            original_rmse = val_metrics.get('rmse', val_metrics['rmse'])
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss (norm): {train_loss:.6f}')
            print(f'  Val Loss (norm): {val_loss:.6f}, Val RMSE (norm): {normalized_rmse:.6f}')
            
            if 'original_predictions' in val_metrics:
                print(f'  Val RMSE (original scale): {original_rmse:.2f}')
            print()
    
    return model, history

# 5. Visualization Function
def visualize_results(history, test_results):
    """
    Visualize training results and predictions.
    
    Args:
        history (dict): Training history
        test_results (dict): Test evaluation results
    """
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot training and validation loss
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training and Validation Loss')
    
    # Plot test predictions vs actual values
    if 'original_predictions' in test_results and 'original_targets' in test_results:
        # Use unscaled predictions
        predictions = test_results['original_predictions']
        actuals = test_results['original_targets']
    else:
        # Use scaled predictions
        predictions = test_results['predictions']
        actuals = test_results.get('targets', np.zeros_like(predictions))
    
    ax2.plot(actuals, label='Actual Prices')
    ax2.plot(predictions, label='Predicted Prices')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Price')
    ax2.legend()
    ax2.set_title('Test Set: Actual vs Predicted Prices')
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics
    print("\nModel Evaluation:")
    print(f"RMSE: {test_results['rmse']:.4f}")
    print(f"MAE: {test_results['mae']:.4f}")
    print(f"R²: {test_results['r2']:.4f}")
    print(f"MAPE: {test_results['mape']:.4f}%")

# Main function
def main(data_path):
    """
    Main function to run the LSTM Bitcoin price prediction.
    
    Args:
        data_path (str): Path to the Bitcoin data JSON file
    """
    # Preprocess data
    train_X, train_y, val_X, val_y, test_X, test_y, scaler = preprocess_data(data_path)
    
    print(f"Data shapes: Train: {train_X.shape}, Validation: {val_X.shape}, Test: {test_X.shape}")
    
    # Display some statistics about the data
    print(f"\nData Statistics:")
    print(f"Training set - Mean: {train_y.mean():.6f}, Min: {train_y.min():.6f}, Max: {train_y.max():.6f}")
    print(f"Validation set - Mean: {val_y.mean():.6f}, Min: {val_y.min():.6f}, Max: {val_y.max():.6f}")
    print(f"Test set - Mean: {test_y.mean():.6f}, Min: {test_y.min():.6f}, Max: {test_y.max():.6f}")
    
    # Create model
    input_size = 1  # Single feature (price)
    hidden_size = 64
    output_size = 1  # Single output (next price)
    model = LSTMModel(input_size, hidden_size, num_layers=1, output_size=output_size)
    
    # Print model architecture
    print(f"\nModel Architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train model
    model, history = train_model(
        model, 
        train_X, train_y, 
        val_X, val_y, 
        epochs=30,  # Reduced epochs for quicker training 
        batch_size=32, 
        learning_rate=0.001,
        scaler=scaler
    )
    
    # Evaluate on test set
    test_X_tensor = torch.FloatTensor(test_X)
    test_y_tensor = torch.FloatTensor(test_y)
    test_results = evaluate_model(model, test_X_tensor, test_y_tensor, scaler)
    test_results['targets'] = test_y
    
    # Display test results summary
    print("\nTest Results:")
    if 'original_predictions' in test_results:
        print(f"RMSE (original scale): {test_results['rmse']:.2f}")
        print(f"MAE (original scale): {test_results['mae']:.2f}")
        print(f"R² score: {test_results['r2']:.4f}")
        print(f"MAPE: {test_results['mape']:.2f}%")
    else:
        print(f"RMSE (normalized): {test_results['rmse']:.6f}")
        print(f"MAE (normalized): {test_results['mae']:.6f}")
        print(f"R² score: {test_results['r2']:.4f}")
        print(f"MAPE: {test_results['mape']:.2f}%")
    
    # Visualize results
    visualize_results(history, test_results)
    
    return model, scaler

# Run the model
if __name__ == "__main__":
    # Replace with your actual data path
    data_path = "BTC_train_data.json"
    
    model, scaler = main(data_path)
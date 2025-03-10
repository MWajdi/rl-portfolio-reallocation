import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Binance BTC & ETH CSV files
df_btc = pd.read_csv("BTC_data.csv")
df_eth = pd.read_csv("ETH_data.csv")

# Convert timestamp column to datetime
df_btc["timestamp"] = pd.to_datetime(df_btc["timestamp"])
df_eth["timestamp"] = pd.to_datetime(df_eth["timestamp"])

# Merge BTC & ETH data on timestamps
df = pd.merge(df_btc[['timestamp', 'close']], df_eth[['timestamp', 'close']], on="timestamp", suffixes=('_BTC', '_ETH'))

# Extract close prices
close_prices = df[['close_BTC', 'close_ETH']].values.T  # Shape: (num_assets, num_time_steps)

# Compute price relatives (x_t = close_t / close_t-1) for each asset
price_relatives = close_prices[:, 1:] / close_prices[:, :-1]  # Shape: (num_assets, num_time_steps - 1)

# OLMAR parameters
epsilon = 10
window = 5
num_assets = price_relatives.shape[0]

# Initialize equal portfolio weights
b_t = np.ones(num_assets) / num_assets  # Equal weights initially

# Store portfolio values
portfolio_values = [1000.0]  # Start with $1 capital

# Iterate over the dataset from the 'window' period onwards
for t in range(window, price_relatives.shape[1]):
    # Get past price relatives for moving average calculation
    past_prices = price_relatives[:, t-window:t]  # Shape: (num_assets, window)
    
    # Compute predicted price relative using moving average
    x_hat_t = np.mean(past_prices, axis=1)  # Moving average for each asset
    
    # Compute average of predicted price relatives
    x_bar_t = np.mean(x_hat_t)

    # Apply OLMAR update rule
    b_new = b_t + epsilon * (x_hat_t - x_bar_t)

    # Normalize weights (ensure sum = 1, no negative values)
    b_new = np.maximum(b_new, 0)
    b_new /= np.sum(b_new)

    # Compute portfolio return
    daily_returns = price_relatives[:, t]  # Price relatives for current time step
    portfolio_value = portfolio_values[-1] * np.dot(b_new, daily_returns)  # Dot product of weights & returns
    portfolio_values.append(portfolio_value)

    # Update portfolio weight
    b_t = b_new

# Convert portfolio values to DataFrame for visualization
df_results = pd.DataFrame({
    "timestamp": df["timestamp"][window:],  # Adjust timestamps
    "portfolio_value": portfolio_values
})

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(df_results["timestamp"], df_results["portfolio_value"], label="OLMAR Portfolio (BTC & ETH)", color="blue")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.title("OLMAR Portfolio Performance on BTC & ETH")
plt.legend()
plt.grid()
plt.show()

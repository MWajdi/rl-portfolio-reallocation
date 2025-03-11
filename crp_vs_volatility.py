import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

def plot_rolling_total_variance_relative(data, start_time, window_size):
    """
    Plot the rolling total variance using absolute relative differences over a specified window size.

    Parameters:
        data (pd.DataFrame): The BTC data with a 'timestamp' column and 'close' column.
        start_time (str): The start timestamp in the format 'YYYY-MM-DD HH:MM:SS'.
        window_size (int): The number of data points in each window.
    """
    # Convert the 'timestamp' column to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Filter the data to start from the specified start_time
    filtered_data = data[data['timestamp'] >= start_time]
    
    # Compute absolute relative differences
    close_prices = filtered_data['close']
    relative_differences = abs(close_prices.pct_change().dropna()).tolist()  # Convert to list
    
    # Initialize a list to store rolling total variances
    rolling_variances = []
    total_variance = sum(relative_differences[0:window_size - 1])
    rolling_variances.append(total_variance)

    # Iterate over the data in rolling windows
    for i in range(len(relative_differences) - window_size + 1):
        # Sum the absolute relative differences for the current window
        total_variance += relative_differences[i + window_size - 1] - relative_differences[i]
        
        # Append the total variance to the list
        rolling_variances.append(total_variance)
    
    # Create a DataFrame for the rolling variances
    rolling_df = pd.DataFrame({
       'timestamp': filtered_data['timestamp'].iloc[:-window_size +1],  # Align timestamps
        'total_variance': rolling_variances
    })
    
    return rolling_df


def crp_performance_vs_start_date(data, window_size, w_cash=0.2, w_btc=0.8, initial_capital=1000.0):
    """
    Trace how the performance of a CRP strategy changes as a function of the start date.

    Parameters:
        data (pd.DataFrame): The BTC data with a 'timestamp' column and 'close' column.
        window_size (int): The number of days for each backtest period.
        w_cash, w_btc (float): Weights for cash and BTC (sum must be 1).
        initial_capital (float): Initial total capital in USD.

    Returns:
        None (Displays a plot).
    """
    # Convert the 'timestamp' column to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values("timestamp").reset_index(drop=True)

    # Define start dates (every 1000 data points)
    start_indices = np.arange(0, len(data) - window_size, 5000)
    start_dates = data['timestamp'].iloc[start_indices]

    # Store CRP performance and rolling volatility
    crp_final_values = []
    rolling_volatility = []
    rolling_variances = []

    rolling_variance_df = plot_rolling_total_variance_relative(data, '2020-01-01 00:00:00', window_size=window_size)
    rolling_variances.append(rolling_variance_df['total_variance'].mean())

    for start_index in tqdm(start_indices, desc="Computing CRP performance"):
        start_date = data['timestamp'].iloc[start_index]
        end_date = start_date + pd.Timedelta(days=window_size)

        # Filter data for the given window
        df_btc = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]

        if df_btc.empty or len(df_btc) < 2:
            continue

        # Compute CRP performance
        capital = initial_capital
        first_btc_price = df_btc["close"].iloc[0]

        holdings_cash = w_cash * capital
        holdings_btc = (w_btc * capital) / first_btc_price

        for p_btc in df_btc["close"]:
            current_value = (holdings_cash * 1.0) + (holdings_btc * p_btc)
            holdings_cash = w_cash * current_value
            holdings_btc = (w_btc * current_value) / p_btc

        crp_final_values.append(current_value)

        

        # Compute rolling total variance (relative differences)


    # Convert results to numpy arrays
    start_dates = np.array(start_dates[:len(crp_final_values)])
    crp_final_values = np.array(crp_final_values)
    rolling_volatility = np.array(rolling_volatility)
    rolling_variances = np.array(rolling_variances)

    # Plot results
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot CRP performance
    ax1.set_xlabel("Start Date")
    ax1.set_ylabel("Final Portfolio Value (USD)", color="b")
    ax1.plot(start_dates, crp_final_values, "bo-", label="CRP Performance")
    ax1.tick_params(axis="y", labelcolor="b")

 

    # Add rolling total variance on tertiary axis
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.2))  # Offset the third axis
    ax3.set_ylabel("Rolling Total Variance (Relative)", color="g")
    ax3.plot(rolling_variance_df['timestamp'], rolling_variance_df['total_variance'], "g", alpha=0.6, label="Rolling Total Variance")
    ax3.tick_params(axis="y", labelcolor="g")

    # Title and legend
    plt.title("CRP Performance vs Start Date with Rolling Volatility and Total Variance")
    fig.tight_layout()
    plt.show()


# Load BTC data
btc_data_path = "BTC_data.csv"  # Adjust path after uploading
data = pd.read_csv(btc_data_path, parse_dates=["timestamp"])

# Define parameters
window_size = 75000  # One year

# Run analysis
crp_performance_vs_start_date(data, window_size)
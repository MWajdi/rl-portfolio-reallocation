import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def trace_total_variation_vs_window_size(data, start_time, max_window_size):
    """
    Trace the total variation as a function of window size starting from a given start date.

    Parameters:
        data (pd.DataFrame): The BTC data with a 'timestamp' column and 'close' column.
        start_time (str): The start timestamp in the format 'YYYY-MM-DD HH:MM:SS'.
        max_window_size (int): The maximum number of data points in a window.

    Returns:
        None (Displays a plot).
    """
    # Convert the 'timestamp' column to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Filter the data to start from the specified start_time
    filtered_data = data[data['timestamp'] >= start_time]

    # Extract close prices
    close_prices = filtered_data['close'].values

    # List to store total variations for different window sizes
    window_sizes = range(2, max_window_size + 1)  # Start from window size 2
    total_variations = []
    total_variation =0
    # Compute total variation for each window size using tqdm for progress tracking
    for window_size in tqdm(window_sizes, desc="Computing total variation"):
        total_variation += abs(close_prices[window_size] - close_prices[window_size-1])
        total_variations.append(total_variation)

    # Plot total variation as a function of window size
    plt.figure(figsize=(10, 6))
    plt.plot(window_sizes, total_variations, color='b', label="Total Variation")
    plt.title("Total Variation as a Function of Window Size")
    plt.xlabel("Window Size")
    plt.ylabel("Total Variation")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_rolling_total_variance(data, start_time, window_size):
    """
    Plot the rolling total variance over a specified window size.

    Parameters:
        data (pd.DataFrame): The BTC data with a 'timestamp' column and 'close' column.
        start_time (str): The start timestamp in the format 'YYYY-MM-DD HH:MM:SS'.
        window_size (int): The number of data points in each window.
    """
    # Convert the 'timestamp' column to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Filter the data to start from the specified start_time
    filtered_data = data[data['timestamp'] >= start_time]
    
    # Precompute the absolute differences between consecutive close prices
    close_prices = filtered_data['close']
    absolute_differences = close_prices.diff().abs().dropna().tolist()  # Convert to list
    
    # Initialize a list to store rolling total variances
    rolling_variances = []
    total_variance = sum(absolute_differences[0:0 + window_size - 1])
    rolling_variances.append(total_variance)
    # Iterate over the data in rolling windows
    for i in range(0,len(absolute_differences) - window_size + 1):  # Adjusted range
        # Sum the absolute differences for the current window
        total_variance += absolute_differences[i + window_size - 1] - absolute_differences[i]
        
        # Append the total variance to the list
        rolling_variances.append(total_variance)
    
    # Create a DataFrame for the rolling variances
    rolling_df = pd.DataFrame({
        'timestamp': filtered_data['timestamp'].iloc[window_size - 1:],  # Align timestamps
        'total_variance': rolling_variances
    })
    
    # Plot the rolling total variance
    plt.figure(figsize=(10, 6))
    plt.plot(rolling_df['timestamp'], rolling_df['total_variance'],  color='b')
    plt.title(f'Rolling Total Variance (Window Size = {window_size})')
    plt.xlabel('Timestamp')
    plt.ylabel('Total Variance')
    plt.grid(True)
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt

def calculate_total_variance_relative(data, start_time, end_time):
    """
    Calculate the total variance using the sum of absolute relative differences 
    between consecutive close prices for a specified period.

    Parameters:
        data (pd.DataFrame): The BTC data with a 'timestamp' column and 'close' column.
        start_time (str): The start timestamp in the format 'YYYY-MM-DD HH:MM:SS'.
        end_time (str): The end timestamp in the format 'YYYY-MM-DD HH:MM:SS'.

    Returns:
        float: The total relative variance over the specified period.
    """
    # Convert the 'timestamp' column to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Filter the data for the specified period
    filtered_data = data[(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)]
    
    # Calculate the absolute relative differences between consecutive close prices
    close_prices = filtered_data['close']
    relative_differences = abs(close_prices.pct_change().dropna())  # Percentage change

    # Sum the relative differences to get the total variance
    total_variance = relative_differences.sum()
    
    return total_variance

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
    
    # Plot the rolling total variance
    plt.figure(figsize=(10, 6))
    plt.plot(rolling_df['timestamp'], rolling_df['total_variance'], color='b')
    plt.title(f'Rolling Total Variance (Relative Differences) - Window Size = {window_size}')
    plt.xlabel('Timestamp')
    plt.ylabel('Total Variance (Relative Differences)')
    plt.grid(True)
    plt.show()

# Load BTC data
btc_data_path = "BTC_data.csv"  # Adjust path after uploading
data = pd.read_csv(btc_data_path, parse_dates=["timestamp"])

# Define parameters
start_time = '2020-01-01 00:00:00'
window_size = 40000  # Number of points in each window

# Plot the rolling total variance using relative differences
plot_rolling_total_variance_relative(data, start_time, window_size)

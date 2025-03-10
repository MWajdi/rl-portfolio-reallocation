import pandas as pd
import time
from binance.client import Client
from datetime import datetime

# Initialize Binance client
client = Client()

# Define parameters
symbol = "ETHUSDT"
interval = Client.KLINE_INTERVAL_5MINUTE  # Modify as needed
start_time = "1 Jan, 2020"  # Start date
data = []

for i in range(200):
    # Fetch data
    klines = client.get_historical_klines(symbol, interval, start_time, limit=1000)
    
    if not klines:
        break  # Stop if no data is returned

    # Append to dataset
    data.extend(klines)

    # Convert last timestamp to readable date
    last_timestamp = klines[-1][0]
    last_date = datetime.utcfromtimestamp(last_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
    
    # Print progress
    print(f"Fetched up to: {last_date}")

    # Move start_time forward
    start_time = last_timestamp
    
    # Pause to avoid rate limits
    time.sleep(1)

# Convert to DataFrame
df = pd.DataFrame(data, columns=[
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

# Save to CSV
df.to_csv("ETC_data_large.csv", index=False)
print("Data saved successfully.")

import pandas as pd
import json
from tqdm import tqdm

def create_train_json(input_csv, output_json, window_size=100, max_samples=10000):
  """
  Reads input_csv (either a path to a CSV file or a DataFrame), extracts the 'close' column and timestamp,
  and produces a new JSON file (output_json) where each entry is:
    {
    timestamp: [x_1, x_2, ..., x_window_size, y]
    }
  with x_i = close price at time t-i, and
     y = close price at time t (the 'next' one).

  Parameters:
  - input_csv: Path to the input CSV file or a DataFrame.
  - output_json: Path to save the output JSON file.
  - window_size: Number of past close prices to use as X.
  - max_samples: Maximum number of (X, Y) samples to generate.
  """
  if isinstance(input_csv, str):
    df = pd.read_csv(input_csv)
  elif isinstance(input_csv, pd.DataFrame):
    df = input_csv
  else:
    raise ValueError("input_csv must be a path to a CSV file or a pandas DataFrame")
  
  if "timestamp" not in df.columns or "close" not in df.columns:
    raise ValueError("Input CSV must contain 'timestamp' and 'close' columns")
  
  # Ensure proper types
  df["timestamp"] = pd.to_datetime(df["timestamp"])
  close_prices = df["close"].astype(float).values
  timestamps = df["timestamp"].astype(str).values  # Convert timestamps to string for JSON keys
  
  data = {}
  for i in tqdm(range(len(close_prices) - window_size), desc="Processing", unit=" samples"):
    x_window = close_prices[i : i + window_size]
    y_next = close_prices[i + window_size]
    timestamp_next = timestamps[i + window_size]  # Use timestamp of Y as key
    
    data[timestamp_next] = (list(x_window), y_next)
    
    if len(data) >= max_samples:
      break  # Stop if max_samples is reached
  
  with open(output_json, "w") as f:
    json.dump(data, f, indent=4)
  
  print(f"Saved {len(data)} samples to {output_json}")

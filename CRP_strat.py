import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def crp_backtest(btc_csv, eth_csv, 
                 w_cash=0.2, w_btc=0.4, w_eth=0.4, 
                 initial_capital=1000.0,
                 start_date=None, end_date=None):
    """
    Implements a 3-asset Constant Rebalanced Portfolio (CRP).

    Arguments:
    btc_csv, eth_csv : Paths to CSVs with columns:
                        ['timestamp', 'close', ...]
    w_cash, w_btc, w_eth  : Target weights (sum = 1).
    initial_capital : Initial total capital in USD.
    start_date      : (Optional) Filter data from this date forward (YYYY-MM-DD).
    end_date        : (Optional) Filter data until this date (YYYY-MM-DD).

    Returns:
    df_result : A DataFrame containing 'timestamp' and 'portfolio_value' columns.
    """

    df_btc = pd.read_csv(btc_csv, parse_dates=["timestamp"])
    df_eth = pd.read_csv(eth_csv, parse_dates=["timestamp"])
    
    df_btc = df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"})
    df_eth = df_eth[["timestamp", "close"]].rename(columns={"close": "close_eth"})
    
    df = pd.merge(df_btc, df_eth, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    if start_date:
        df = df[df["timestamp"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["timestamp"] <= pd.to_datetime(end_date)]
    
    if df.empty:
        raise ValueError("No data left after applying date filters!")
    
    if abs((w_cash + w_btc + w_eth) - 1.0) > 1e-9:
        raise ValueError("Weights must sum to 1.0!")
    
    capital = initial_capital
    first_btc_price = df["close_btc"].iloc[0]
    first_eth_price = df["close_eth"].iloc[0]
    
    holdings_cash = w_cash * capital
    holdings_btc = (w_btc * capital) / first_btc_price
    holdings_eth = (w_eth * capital) / first_eth_price
    
    portfolio_values = []
    timestamps = []
    
    for idx, row in df.iterrows():
        ts = row["timestamp"]
        p_btc = row["close_btc"]
        p_eth = row["close_eth"]
        
        current_value = (holdings_cash * 1.0) + (holdings_btc * p_btc) + (holdings_eth * p_eth)
        
        portfolio_values.append(current_value)
        timestamps.append(ts)
        
        holdings_cash = w_cash * current_value
        holdings_btc = (w_btc * current_value) / p_btc
        holdings_eth = (w_eth * current_value) / p_eth
    
    df_result = pd.DataFrame({"timestamp": timestamps, "portfolio_value": portfolio_values})
    return df_result
def compute_discrete_ratios(result, step=288):
    
    ratios = [result[i] / result[i - step] for i in range(step, len(result),step)]

    return ratios

from scipy.stats import gmean



if __name__ == "__main__":
    strategies = [
        ("Strategy 1", 0.2, 0.4, 0.4),


    ]
    
    plt.figure(figsize=(12, 6))
    
    for name, w_cash, w_btc, w_eth in strategies:
        print(name)
        df_res = crp_backtest(
            btc_csv="binance_datasets/BTC_data.csv", 
            eth_csv="binance_datasets/ETH_data.csv", 
            w_cash=w_cash, w_btc=w_btc, w_eth=w_eth,
            initial_capital=1000.0,
            start_date="2020-01-01",
            end_date="2021-11-26"
        )
    df_ratios_result = compute_discrete_ratios(np.array(df_res["portfolio_value"]), step=288)

    print( sum(df_ratios_result)/len(df_ratios_result))


    # Plot histogram
    plt.hist(df_ratios_result, bins='auto', edgecolor='black', alpha=0.7)
    plt.xlabel("Successive 288-Step Portfolio Value Ratios")
    plt.ylabel("Frequency")
    plt.title("Histogram of Portfolio Value Ratios")
    plt.show()
    
    plt.plot(df_res["timestamp"], df_res["portfolio_value"], label=name)
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value (USD)")
    plt.title("Comparison of Different CRP Strategies")
    plt.legend()
    plt.show()

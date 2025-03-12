import pandas as pd
import numpy as np

experiment_names = ['portfolio_value', 'sharpe_ratio']

# Experiments parameters

def create_experiment(reward_method):
    return {
            # Start and end time for dataset
            "start_time": "2020-01-01 00:00:00",
            "end_time": "2021-01-01 00:00:00",

            # Prediction model settings
            "model_type": "linear_regression",
            "train_prediction_model": False,
            "prediction_model_epochs": 200,
            "prediction_model_lr": 2 * 1e-5,

            "use_existing_prediction_model": True,  
            "prediction_model_paths": ["models/btc_1_year_linear_model.pth", "models/eth_1_year_linear_model.pth"],

            "train_rl_model": False,

            # Dataset split
            "train_prop": 0.8,

            # Environment settings
            "n_assets": 2,                    
            "window_size": 10,               
            "action_step_size": 0.25,          
            "episode_length": 10,            
            "reward_method": reward_method,  
            "g1": 0.5,                       
            "g2": 0.5,                        
            "prediction_method": "regression",  
            "use_prediction_model": True,    

            # PPO Algorithm settings
            "total_timesteps": 500000,      
            "rl_model_lr": 1e-3,            
            "num_envs": 8,                  
            "num_steps": 72,                
            "anneal_lr": True,              
            "gamma": 0.98,                  
            "gae_lambda": 0.95,             
            "num_minibatches": 4,           
            "update_epochs": 4,             
            "norm_adv": True,               
            "clip_coef": 0.2,               
            "clip_vloss": True,             
            "ent_coef": 0.01,               
            "vf_coef": 0.5,                 
            "max_grad_norm": 0.5,           
            "target_kl": None,            

            # Evaluation settings
            "num_eval_episodes": 10000   
        }

experiments = {
    "experiment_1_year_test": create_experiment("expirement_1_year_test"),
    "sharpe_ratio": create_experiment("sharpe_ratio")
}


# Initialize experiment results
experiment_results = {
    experiment_name: {
        "returns": {},
        "metrics": {}
    }
    for experiment_name in experiment_names
}

# Create experiment results directory
import os
for experiment_name in experiment_names:
    experiment_results_dir = f"results/{experiment_name}"
    os.makedirs(experiment_results_dir, exist_ok=True)  # Create directory if it doesn't exist


# Load historical price data
BTC_data_complete = pd.read_csv('binance_datasets/BTC_data.csv')
ETH_data_complete = pd.read_csv('binance_datasets/ETH_data.csv')

for experiment_name, params in experiments.items():
    print("#"*100)
    print(f"\n ---------------- Running experiment: {experiment_name} ----------------\n")
    print("#"*100, "\n")
    print(f"Parameters: {params}")
    
    # Limit dataset to specific time frame
    start_time = params.get('start_time')
    end_time = params.get('end_time')

    if start_time is not None and end_time is not None:
        BTC_data = BTC_data_complete[(BTC_data_complete['timestamp'] >= start_time) & (BTC_data_complete['timestamp'] <= end_time)]
        ETH_data = ETH_data_complete[(ETH_data_complete['timestamp'] >= start_time) & (ETH_data_complete['timestamp'] <= end_time)]

    print(f"Length of BTC data: {len(BTC_data)}")
    print(f"Length of ETH data: {len(ETH_data)}")

    # Get the closing prices
    BTC_close = BTC_data['close']
    ETH_close = ETH_data['close']

    # Create closing prices matrix
    closing_prices = np.column_stack((BTC_close, ETH_close))
    print("closing_prices.shape = ",closing_prices.shape)

    # Split the data into training and testing sets
    train_prop = params.get('train_prop')
    train_size = int(train_prop * closing_prices.shape[0])
    train_data = closing_prices[:train_size]
    test_data = closing_prices[train_size:]

    # Split original dataframes
    BTC_train = BTC_data[:train_size]
    BTC_test = BTC_data[train_size:]
    ETH_train = ETH_data[:train_size]
    ETH_test = ETH_data[train_size:]

    print(f"Length of training data: {len(train_data)}")
    print(f"Length of testing data: {len(test_data)}")

    window_size = params.get('window_size')
    model_type = params.get('model_type')

    # Train prediction model if needed
    if params.get("train_prediction_model"):
        epochs = params.get('prediction_model_epochs')
        lr = params.get('prediction_model_lr')

        # If linear regression model
        if model_type == 'linear_regression':
            import train_linear_model
            from train_data_generation import create_train_json

            # Create json sequences files for training and testing data
            create_train_json(BTC_train, f"binance_datasets/BTC_{experiment_name}_train.json", window_size=window_size, max_samples=np.inf)
            create_train_json(ETH_train, f"binance_datasets/ETH_{experiment_name}_train.json", window_size=window_size, max_samples=np.inf)

            create_train_json(BTC_test, f"binance_datasets/BTC_{experiment_name}_test.json", window_size=window_size, max_samples=np.inf)
            create_train_json(ETH_test, f"binance_datasets/ETH_{experiment_name}_test.json", window_size=window_size, max_samples=np.inf)


            args = train_linear_model.Args(
                model_path = f"models/BTC_{experiment_name}_linear_model.pth",
                train_path = f"binance_datasets/BTC_{experiment_name}_train.json",
                test_path = f"binance_datasets/BTC_{experiment_name}_test.json",
                epochs = epochs,
                lr = lr
            )

            btc_linear_model = train_linear_model.main(args)

            args = train_linear_model.Args(
                model_path = f"models/ETH_{experiment_name}_linear_model.pth",
                train_path = f"binance_datasets/ETH_{experiment_name}_train.json",
                test_path = f"binance_datasets/ETH_{experiment_name}_test.json",
                epochs = epochs,
                lr = lr
            )

            eth_linear_model = train_linear_model.main(args)

    # Load prediction model
    if model_type == 'linear_regression':
        import ppo
        import torch
        import torch.nn as nn

        # Define prediction models for each asset
        btc_linear_model = model = nn.Linear(in_features=window_size, out_features=1)

        if params.get("use_existing_prediction_model"):
            btc_linear_model.load_state_dict(torch.load(params.get("prediction_model_paths")[0]))
        else:
            btc_linear_model.load_state_dict(torch.load(f"models/BTC_{experiment_name}_linear_model.pth"))
        btc_linear_model.to("cuda")  # Move model to GPU if needed

        eth_linear_model = nn.Linear(in_features=window_size, out_features=1)  # Adjust input_dim

        if params.get("use_existing_prediction_model"):
            eth_linear_model.load_state_dict(torch.load(params.get("prediction_model_paths")[1]))
        else:
            eth_linear_model.load_state_dict(torch.load(f"models/ETH_{experiment_name}_linear_model.pth"))
        eth_linear_model.to("cuda")

        prediction_models = [btc_linear_model, eth_linear_model]

    # Set environment parameters
    if model_type == 'linear_regression' or model_type == 'lstm_regression':
        prediction_method = "regression"
    elif model_type == 'linear_directional' or model_type == 'lstm_directional':
        prediction_method = "directional"

    args = ppo.Args(
        # Experiment metadata
        exp_name=experiment_name,  
        seed=1,  
        torch_deterministic=True,  
        cuda=True,  
        track=False,  

        # Environment settings (retrieved from params)
        n_assets=params.get("n_assets", 2),  
        window_size=params.get("window_size", 288),  
        action_step_size=params.get("action_step_size", 0.1),  
        episode_length=params.get("episode_length", 288),  
        reward_method=params.get("reward_method", "portfolio_value"),  
        g1=params.get("g1", 0.5),  
        g2=params.get("g2", 0.5),  
        closing_prices=train_data,  
        prediction_method=prediction_method,  
        prediction_models=prediction_models if params.get("use_prediction_model", True) else None,  

        # PPO Algorithm settings
        total_timesteps=params.get("total_timesteps", 500000),  
        learning_rate=params.get("rl_model_lr", 1e-4),  
        num_envs=params.get("num_envs", 8),  
        num_steps=params.get("num_steps", 72),  
        anneal_lr=params.get("anneal_lr", True),  
        gamma=params.get("gamma", 0.98),  
        gae_lambda=params.get("gae_lambda", 0.95),  
        num_minibatches=params.get("num_minibatches", 4),  
        update_epochs=params.get("update_epochs", 4),  
        norm_adv=params.get("norm_adv", True),  
        clip_coef=params.get("clip_coef", 0.2),  
        clip_vloss=params.get("clip_vloss", True),  
        ent_coef=params.get("ent_coef", 0.01),  
        vf_coef=params.get("vf_coef", 0.5),  
        max_grad_norm=params.get("max_grad_norm", 0.5),  
        target_kl=params.get("target_kl", None),  

        # Computed at runtime
        batch_size=0,  
        minibatch_size=0,  
        num_iterations=0  
    )

    if params.get("train_rl_model"):
        ppo.main(args)

    # Evaluate the model
    import torch
    import numpy as np
    import pandas as pd
    import gymnasium as gym
    import ppo  # Import PPO script
    from portfolio_env import make_wrapped_env
    import matplotlib.pyplot as plt

    # -----------------------------
    # 1️⃣ Load the trained model
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.num_envs = 8 # Single environment for evaluation

    envs = gym.vector.SyncVectorEnv(
            [ppo.make_env(args.env_id, args) for _ in range(args.num_envs)]
        )
    
    
    # Load trained PPO agent
    agent = ppo.Agent(envs).to(device)
    agent.load_state_dict(torch.load(f"models/ppo_agent_{args.exp_name}.pt"))
    agent.eval()  # Set to evaluation mode

        
    # -----------------------------
    # 2️⃣ Load the test dataset
    # -----------------------------

    # Ensure dataset has enough data
    assert test_data.shape[0] > envs.get_attr("episode_length")[0], "Test data is too short for evaluation!"

    # -----------------------------
    # 3️⃣ Run Parallel Evaluation Episodes
    # -----------------------------

    # Number of evaluation runs
    num_eval_episodes = params.get("num_eval_episodes", 10000)

    # Store results
    results = []

    # Number of batches (each batch runs `args.num_envs` environments in parallel)
    num_batches = num_eval_episodes // args.num_envs

    for batch in range(num_batches):
        # Randomly select time windows for each environment
        start_times = np.random.randint(
            window_size, test_data.shape[0] - args.episode_length, size=args.num_envs
        )
        
        # Assign separate time windows to each environment
        test_windows = np.array([
            test_data[start_time : start_time + args.episode_length]
            for start_time in start_times
        ])

        # Set closing prices for evaluation
        args.closing_prices = test_windows

        # Reset the environments and get initial observations
        next_obs, _ = envs.reset()
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)  # Convert to Tensor

        next_done = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
        total_rewards = np.zeros(args.num_envs)
        all_returns = [[] for _ in range(args.num_envs)]  # Portfolio values over time
        all_portfolio_weights = [[] for _ in range(args.num_envs)]  # Portfolio allocations

        while not next_done.all():  # Continue until all environments are done
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(next_obs)

            next_obs, rewards, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = torch.tensor(np.logical_or(terminations, truncations), dtype=torch.bool, device=device)

            total_rewards += rewards  # Track rewards per environment
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)  # Ensure tensor format

            # Collect historical returns and portfolio weights for each environment
            for i in range(args.num_envs):
                all_returns[i].append(envs.get_attr("historical_returns")[i][-1])
                all_portfolio_weights[i].append(envs.get_attr("portfolio")[i])

        # Calculate final portfolio value & Sharpe ratio for each environment
        final_values = [returns[-2] for returns in all_returns]
        mean_returns = [np.mean(returns) for returns in all_returns]
        std_returns = [np.std(returns) + 1e-8 for returns in all_returns]  # Avoid division by zero
        sharpe_ratios = [m / s for m, s in zip(mean_returns, std_returns)]

        # Store results for all environments
        for i in range(args.num_envs):
            results.append({
                "returns": all_returns[i],
                "portfolio_weights": all_portfolio_weights[i],
                "episode": batch * args.num_envs + i,  # Unique episode index
                "start_time": start_times[i],
                "final_value": final_values[i],
                "sharpe_ratio": sharpe_ratios[i],
                "total_reward": total_rewards[i]
            })

            print(f"Episode {batch * args.num_envs + i + 1}/{num_eval_episodes}: "
                f"Final Value = {final_values[i]:.4f}, Sharpe Ratio = {sharpe_ratios[i]:.4f}")
                
        
    # Store results in experiment_results
    experiment_results[experiment_name]["results"] = results

    # -----------------------------
    # 4️⃣ Display and Plot Results
    # -----------------------------
    df_results = pd.DataFrame(results).drop(columns=['returns','portfolio_weights'])
    print(df_results)

    # Save results to CSV
    df_results.to_csv(f"results/{experiment_name}/results.csv", index=False)

    # Plot portfolio values over time for a few episodes
    plt.figure(figsize=(12, 5))
    for i in range(min(5, num_eval_episodes)):  # Plot first 5 episodes
        returns = results[i]["returns"][:-1]
        plt.plot(range(len(returns)), returns, label=f"Episode {i}")

    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value")
    plt.title("Portfolio Growth Over Time")
    plt.legend()
    plt.savefig(f"results/{experiment_name}/portfolio_growth.png")  # Save the figure
    plt.show()

    envs.close()

    # Plot portfolio weights over time for a single episode, separate plot for each asset
    episode_index = 0  # Change this to the desired episode index

    portfolio_weights = results[episode_index]["portfolio_weights"]
    portfolio_weights = np.array(portfolio_weights)

    num_assets = portfolio_weights.shape[1]

    plt.figure(figsize=(12, 5 * num_assets))
    for j in range(num_assets):
        plt.subplot(num_assets, 1, j + 1)
        plt.plot(range(len(portfolio_weights)), portfolio_weights[:, j], label=f"Asset {j+1}")
        plt.xlabel("Time Step")
        plt.ylabel("Portfolio Weights")
        plt.title(f"Portfolio Weights Over Time - Asset {j+1}")
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"results/{experiment_name}/portfolio_weights.png")  # Save the figure
    plt.show()

    # Measure average final portfolio value and Sharpe ratio across all episode
    final_values = df_results["final_value"].values
    sharpe_ratios = df_results["sharpe_ratio"].values

    avg_final_value = np.mean(final_values)
    avg_sharpe_ratio = np.mean(sharpe_ratios)

    print(f"Average Final Portfolio Value: {avg_final_value:.4f}")
    print(f"Average Sharpe Ratio: {avg_sharpe_ratio:.4f}")

    # Measure standard deviation of final portfolio value and Sharpe ratio across all episodes
    std_final_value = np.std(final_values)
    std_sharpe_ratio = np.std(sharpe_ratios)

    print(f"Standard Deviation of Final Portfolio Value: {std_final_value:.4f}")
    print(f"Standard Deviation of Sharpe Ratio: {std_sharpe_ratio:.4f}")

    # Save average results to experiment_results
    experiment_results[experiment_name]["metrics"]["avg_final_value"] = avg_final_value
    experiment_results[experiment_name]["metrics"]["avg_sharpe_ratio"] = avg_sharpe_ratio
    experiment_results[experiment_name]["metrics"]["std_final_value"] = std_final_value
    experiment_results[experiment_name]["metrics"]["std_sharpe_ratio"] = std_sharpe_ratio

    # Save metrics to CSV
    df_metrics = pd.DataFrame(experiment_results[experiment_name]["metrics"], index=[0])
    df_metrics.to_csv(f"results/{experiment_name}/metrics.csv", index=False)






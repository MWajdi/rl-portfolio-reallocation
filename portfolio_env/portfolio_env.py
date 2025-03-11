from typing import Optional
import numpy as np
import gymnasium as gym
import itertools
import matplotlib.pyplot as plt
import torch
from typing import Optional, List
import torch.nn as nn


class PortfolioEnv(gym.Env):
    def _generate_discrete_weights(self):
        """Generate all valid weight vectors where sum(w) = 1 with discretization."""
        num_steps = int(1 / self.action_step_size) + 1
        candidates = itertools.product(range(num_steps), repeat=self.n_assets + 1)
        valid_weights = [np.array(c) * self.action_step_size for c in candidates if sum(c) == num_steps - 1]
        return valid_weights

    def _get_prediction(self, i, prices):
        """Get prediction."""
        model = self.prediction_models[i]
        x_test = torch.tensor(prices, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            prediction = model(x_test).squeeze(-1)
            
        return prediction.cpu().numpy()


    def __init__(self, n_assets: int = 1, window_size: int = 1, action_step_size: float = 0.1, 
                 closing_prices: np.ndarray = np.array([]), episode_length: int = 1000, prediction_models: Optional[List[nn.Module]] = None, 
                 prediction_method = "directional", reward_method: str = "portfolio_value", g1: float = 0.5, g2: float = 0.5):
        
        self.n_assets = n_assets
        self.closing_prices = closing_prices
        self.window_size = window_size

        self.portfolio = np.array([1] + [0] * self.n_assets) # Starting portfolio with 100% cash
        self.action_step_size = action_step_size
        self.historical_returns = [1.0]  

        self.initial_time_step = window_size + 1 # Initial time step, chosen randomly in reset()
        self.time_step = window_size + 1 # Current time step
        self.episode_length = episode_length

        # ----- Prediction models ------
        self.prediction_method = prediction_method

        # Prediction models for each asset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prediction_models = [
            model.to(self.device) if model is not None else None 
            for model in prediction_models
        ] if prediction_models is not None else [None] * self.n_assets

        # ----- Reward parameters ------
        self.reward_method = reward_method # Reward method, can be set to "portfolio_value" or "sharpe_ratio"
        self.g1 = g1 # Weight for Sharpe ratio in reward calculation ("sharpe_ratio" method)
        self.g2 = g2 # Weight for portfolio return in reward calculation ("sharpe_ratio" method)


        # ----- Observation space ------
        if self.prediction_method == "directional":
            self.observation_space = gym.spaces.Dict(
                {
                "current_portfolio": gym.spaces.Box(0, 1, shape=(1 + self.n_assets,), dtype=float),
                "price_history": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.window_size, self.n_assets), dtype=float
                ),
                "movement_prediction": gym.spaces.MultiDiscrete([2] * self.n_assets)
                }
            )
        elif self.prediction_method == "regression":
            self.observation_space = gym.spaces.Dict(
                {
                "current_portfolio": gym.spaces.Box(0, 1, shape=(1 + self.n_assets,), dtype=float),
                "price_history": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.window_size, self.n_assets), dtype=float
                ),
                "movement_prediction": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.n_assets,), dtype=float
                )
                }
            )
        else:
            raise ValueError(f"Unknown prediction method: {prediction_method}")
        

        # ----- Action space ------
        # Generate all valid discrete weight allocations
        self.all_weights = self._generate_discrete_weights()

        # Discrete action space: each action corresponds to an index in self.all_weights
        self.action_space = gym.spaces.Discrete(len(self.all_weights))
    
    def get_predictions(self):
        """Get prediction."""
        price_window = self.closing_prices[self.time_step - self.window_size : self.time_step]
        return [self._get_prediction(i, price_window[:,i]) for i in range(self.n_assets)]

    def _get_obs(self):
        """Get observation."""
        price_window = self.closing_prices[self.time_step - self.window_size : self.time_step]
        return {
            "current_portfolio": self.portfolio,
            "price_history": price_window,
            "movement_prediction": self.get_predictions()
        }
    
    def _get_info(self):
        """Get info."""
        return {} # TODO
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Call Gymnasium's seed function to initialize self.np_random
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Reset portfolio
        self.portfolio = np.array([1] + [0] * self.n_assets)
        self.historical_returns = [1.0]  

        # Reset time step
        # Random time step between window_size + 1 and the end of the closing prices - episode_length
        self.initial_time_step = self.np_random.integers(self.window_size + 1, len(self.closing_prices) - self.episode_length)
        self.time_step = self.initial_time_step

        observation = self._get_obs()
        info = self._get_info() # TODO

        return observation, info
    
    def _calculate_reward(self):
        """Calculate reward based on the method used in the article."""

        if self.reward_method == "portfolio_value":
            # Use sum of all past returns 
            return self.historical_returns[-1]

        elif self.reward_method == "sharpe_ratio":
            if len(self.historical_returns) > 1:
                mean_return = np.mean(self.historical_returns)
                std_return = np.std(self.historical_returns) + 1e-8  # Avoid div by zero
                sharpe_ratio = mean_return / std_return
            else:
                sharpe_ratio = 0  # Default if insufficient data

            # Use weighted combination of Sharpe ratio and portfolio return
            portfolio_net_return = self.historical_returns[-1]
            return self.g1 * sharpe_ratio + self.g2 * portfolio_net_return  

        else:
            raise ValueError(f"Unknown reward method: {self.reward_method}")


    def _graph_returns(self):
        """Graph the returns."""
        plt.plot(range(len(self.historical_returns)), self.historical_returns)
        plt.title("Portfolio returns")
        plt.xlabel("Time step")
        plt.ylabel("Portfolio value")
        plt.show()            

    def step(self, action):
        # Retrieve the portfolio weights from the discrete action space
        asset_weights = self.all_weights[action]  # Shape: (n_assets + 1,)

        # Ensure cash allocation is included
        self.portfolio = asset_weights  # Shape: (n_assets + 1,)


        # Calculate compounding returns
        portfolio_returns = (self.closing_prices[self.time_step + 1] - self.closing_prices[self.time_step]) / self.closing_prices[self.time_step]
        weighted_return = np.sum(self.portfolio[1:] * portfolio_returns) 

        # Update historical compounding returns
        # Compounding returns are defined as V(t) = V(t-1) * (1 + r(t)) where r(t) = w1 * r1(t) + ... + wn * rn(t), and r_i(t) is the return of asset i at time t
        self.historical_returns.append(self.historical_returns[-1] * (1 + weighted_return))         

        # Calculate reward
        reward = self._calculate_reward()

        # An environment is completed if and only if final time step is reached
        terminated = (self.time_step == self.initial_time_step + self.episode_length - 1)

        if terminated:
            # self._graph_returns()
            pass

        # Move to the next time step
        self.time_step += 1

        # Get observation, info
        observation = self._get_obs()
        info = self._get_info()

        truncated = False # No truncation condition

        return observation, reward, terminated, truncated, info
    

# Register the environment
gym.register(
    id="gymnasium_env/PortfolioEnv-v0",
    entry_point=PortfolioEnv,
)
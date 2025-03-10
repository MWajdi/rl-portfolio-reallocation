import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import FlattenObservation
from portfolio_env.portfolio_env import PortfolioEnv

# Register the environment
register(
    id="PortfolioEnv-v0",
    entry_point="portfolio_env.portfolio_env:PortfolioEnv",  # Ensure correct path
)

# Wrap the environment when using gym.make()
def make_wrapped_env(**kwargs):
    env = gym.make("PortfolioEnv-v0", **kwargs)
    env = FlattenObservation(env)  # Flatten observations for compatibility
    return env

# Verify registration (debugging check)
if "PortfolioEnv-v0" not in gym.envs.registry:
    raise ValueError("ERROR: PortfolioEnv-v0 is not registered in Gymnasium!")
else:
    print(" PortfolioEnv-v0 successfully registered!")

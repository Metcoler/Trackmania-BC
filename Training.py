from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from Enviroment import RacingGameEnviroment

# Create a function to instantiate your environment
def make_env():
    return RacingGameEnviroment()

# Create a vectorized environment
env = DummyVecEnv([make_env])

# Instantiate and train the DQN agent
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=int(1e5))

# Save the trained model
model.save("dqn_racing_game")


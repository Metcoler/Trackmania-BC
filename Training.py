import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Import your custom environment
from Enviroment import RacingGameEnviroment

# Create the environment
env = RacingGameEnviroment(map_name="AI Training #2")
env = DummyVecEnv([lambda: env])

# Define the callback to save the model every 10,000 time steps
checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path='./logs/',
                                         name_prefix='model')

# Load the model if there's a saved model
try:
    model = PPO.load("logs\model_2400000_steps.zip", env=env, tensorboard_log="./logs/", n_steps=2**13)
    print("Model loaded successfully.")
except Exception as e:
    print("No existing model found. Creating a new one.")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/", n_steps=2**13)

# Train the model
model.learn(total_timesteps=int(10_000_000), callback=checkpoint_callback)

# Save the final model
model.save("ppo_racing_game")

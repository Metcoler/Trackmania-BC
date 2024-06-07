from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from Enviroment import RacingGameEnviroment

# Create the environment
env = RacingGameEnviroment(map_name="tmrl-test")
env = DummyVecEnv([lambda: env])

# Define the callback to save the model every 100_000 time steps
checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path='./logs/',
                                         name_prefix='modell')

# Load the model if there's a saved model
try:
    model = PPO.load("logs\\second_stage_training.zip", env=env, tensorboard_log="./logs/", n_steps=RacingGameEnviroment.STEPS, learning_rate=0.000001)
    print("Model loaded successfully.")
except Exception as e:
    print("No existing model found. Creating a new one.")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/", n_steps=RacingGameEnviroment.STEPS, learning_rate=0.000001)

print("ucim sa")
# Train the model
model.learn(total_timesteps=int(10_000_000), callback=checkpoint_callback)

# Save the final model
model.save("ppo_racing_game")

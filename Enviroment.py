import gymnasium as gym
import numpy as np
import socket
import vgamepad
import time

from Car import Car
from Map import Map


class RacingGameEnviroment(gym.Env):

    def __init__(self) -> None:
        super().__init__()
        print("Creating the RacingGameEnviroment")
        # Observations: distances, car position, car direction, car speed   
        self.observation_space = gym.spaces.Box(low=0.0, high=320.0, shape=(Car.NUM_LASERS + 9,))

        # Actions: throttle, brake, left, right as binary number
        #self.action_space = self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(8)
        
        # Create the car and the map
        self.car = Car()
        self.map = Map("small_map")

        # Gamepad
        self.controller = vgamepad.VX360Gamepad()

        # Connect to the game
        self.inet_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.inet_socket.connect(("127.0.0.1", 9002))
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.reset_game()
        observation = self.observation_info()
        return observation

    def step(self, action):
        self.action_discrete(action)
        observation, info = self.observation_info()
        distances = observation[:Car.NUM_LASERS]
        reward = self.compute_reward(info)
        done = info["done"] == 1.0

        if any(distances < 3):
            print("crashed")
            done = True
            reward = -100.0
        truncated = False
        return observation, reward, done, truncated, info

    def observation_info(self):
        return self.car.get_data(self.inet_socket, self.map)
    
    def action(self, action):
        self.controller.reset()
        self.controller.right_trigger_float(action[0])
        self.controller.left_trigger_float(action[1])
        self.controller.left_joystick_float(action[2], 0)
        self.controller.update()
    
    def action_discrete(self, action):
        # binary number... break, gas, left, right
        self.controller.reset()
        ##if action >> 3 & 1:
        ##    self.set_brake()

        if action >> 2 & 1:
            self.set_gas()
        
        if action >> 1 & 1 and action & 1:
            self.set_steer(0.0)
        elif action >> 1 & 1:
            self.set_steer(-1.0)
        elif action & 1:
            self.set_steer(1.0)
            
        self.controller.update()

    def set_gas(self):
        self.controller.right_trigger_float(1.0)

    def set_brake(self):
        self.controller.left_trigger_float(1.0)

    

    def set_steer(self, steer_angle):
        self.controller.left_joystick_float(steer_angle, 0)

    def reset_game(self):
        
        self.controller.press_button(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_B)
        self.controller.update()
        time.sleep(0.3)
        self.controller.release_button(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_B)
        self.controller.update()
        self.car.reset()
        time.sleep(1)

    def compute_reward(self, info):
        return info["speed"] + info["side_speed"] + info["distance"] - info["time"]

if __name__ == "__main__":

    env = RacingGameEnviroment()
    env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
        if done:
            env.reset_game()
            env.reset()
    env.close()
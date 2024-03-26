import gymnasium as gym
import numpy as np
import socket
import vgamepad
import time

from Car import Car
from Map import Map


class RacingGameEnviroment(gym.Env):

    def __init__(self, map_name) -> None:
        super().__init__()
        print("Creating the RacingGameEnviroment")
        # Observations: distances, x, z, dx, dz, speed, side_speed, next point dot product 
        self.observation_space = gym.spaces.Box(low=0.0, high=32000.0, shape=(Car.NUM_LASERS + 7,))

        # Actions: throttle, brake, left, right as binary number
        self.action_space = self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        #self.action_space = gym.spaces.Discrete(8)
        
        # Create the car and the map
        self.map = Map(map_name)
        self.car = Car(self.map)
        
        # Gamepad
        self.controller = vgamepad.VX360Gamepad()
        self.controller.reset()

        self.max_episode_steps = 2**13
        self.current_step = 0
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.reset_game()
        self.current_step = 0
        distances, info = self.observation_info()
        observation = np.array(distances + [info['x'], info['z'], info['dx'], info['dz'], info['speed'], info['side_speed'], info['next_point_direction']])
        return observation, info

    def step(self, action):
        #self.action_discrete(action)
        self.action(action)
        distances, info = self.observation_info()

        observation = np.array(distances + [info['x'], info['z'], info['dx'], info['dz'], info['speed'], info['side_speed'], info['next_point_direction']])
    
        reward = self.compute_reward(info) + action[0] * 0.1
        done = info["done"] == 1.0

        

        if done:
            reward = 100
    
        elif self.current_step >= self.max_episode_steps - 1:
            done = True
            reward = (info['map_progress'] / len(self.map.path_tiles)) * 100



        for distance in distances:
            if distance < 3:
                reward -= 1
                break

        truncated = False
        print(self.current_step, ":", reward, end='\r')
        
        self.current_step += 1
        return observation, reward, done, truncated, info

    def observation_info(self):
        return self.car.get_data()
    
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
        self.controller.reset()
        self.controller.press_button(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_B)
        self.controller.update()
        time.sleep(0.3)
        self.controller.release_button(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_B)
        self.controller.update()
        self.car.reset()
        self.controller.reset()
        time.sleep(1)

    def compute_reward(self, info):
        speed = ((info["speed"]**2 +  info["side_speed"]**2) ** 0.5) / 1000
        return  speed*info['next_point_direction'] + info['map_progress'] * 10

if __name__ == "__main__":
    map_name = "small_map"
    env = RacingGameEnviroment(map_name)
    env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
        if done:
            env.reset_game()
            env.reset()
    env.close()
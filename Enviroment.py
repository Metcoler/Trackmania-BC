import gymnasium as gym
import numpy as np
import socket
import vgamepad
import time

from Car import Car
from Map import Map


class RacingGameEnviroment(gym.Env):
    STEPS = 2**13

    def __init__(self, map_name, max_time=60, never_quit=False) -> None:
        super().__init__()
        print("Creating the RacingGameEnviroment")
        # Observations: distances, instructions, speed, side_speed, next point dot product, 
        self.observation_space = gym.spaces.Box(low=0.0, high=1000.0, shape=(Car.NUM_LASERS + Car.SIGHT_TILES + 6,))

        # Actions: throttle, brake, left, right as binary number
        self.action_space = self.action_space = gym.spaces.Box(low=-0.2, high=0.2, shape=(3,), dtype=np.float32)
        #self.action_space = gym.spaces.Discrete(8)
        
        # Create the car and the map
        self.map = Map(map_name)
        self.car = Car(self.map)
        
        # Gamepad
        self.controller = vgamepad.VX360Gamepad()
        self.controller.reset()

        self.max_episode_steps = RacingGameEnviroment.STEPS
        self.max_time = max_time
        if never_quit:
            self.max_episode_steps = float('inf')
        self.current_step = 0
        self.race_terminated = 0 
        self.never_quit = never_quit

    
    def reset(self, seed=None):
        
        super().reset(seed=seed)
        self.reset_game()
        self.current_step = 0
        self.previous_observation_info = distances, instructions, info = self.observation_info()
        self.previous_action = np.array([0.0, 0.0, 0.0])
        self.previous_observation = observation = np.array(distances + instructions + [info['speed'], info['side_speed'], info['next_point_direction']] + list(self.previous_action))
        
        self.race_terminated = 0 

        
        return observation, info

    def step(self, action):
        done = False
        truncated = False
        if self.current_step >= self.max_episode_steps - 1:
            done = True
        self.current_step += 1


        
        if self.race_terminated != 0:
            distances, instructions, info = self.previous_observation_info
            observation = self.previous_observation
            print(self.current_step, ":",  self.race_terminated, " "*20,  end='\r')
            return observation, self.race_terminated, done, truncated, info
        else:
            try:
                self.perform_action(action)
            except:
                ...
                # TODO [nan, nan, nan] shouldnt be action... Where did it come from    
            distances, instructions, info = self.observation_info()
            observation = np.array(distances + instructions + [info['speed'], info['side_speed'], info['next_point_direction']] + list(self.previous_action))
            self.previous_observation_info = distances, instructions, info
            self.previous_observation = observation
        
            
        if info["time"] >= self.max_time:
            self.controller.reset()
            self.controller.update()
            self.race_terminated = 1

        reward = self.compute_reward(info, distances, action)
        
        if info["done"] == 1.0:
            self.controller.reset()
            self.controller.update()
            self.race_terminated = 1
        elif info["next_point_direction"] < -0.5  and not self.never_quit:
            self.controller.reset()
            self.controller.update()
            self.race_terminated = -1
        else:
            
            for distance in distances:
                if distance < 2 and not self.never_quit:
                    self.race_terminated = -1
                    self.controller.reset()
                    self.controller.update()
                    reward = 0
                    break
            if info["speed"] < 3 and info["time"] > 3 and not self.never_quit:
                self.race_terminated = -1
                self.controller.reset()
                self.controller.update()
                reward = 0

        
        
        
        print(self.current_step, ":", info["speed"], " "*20,  end='\r')
        
        
        return observation, reward, done, truncated, info

    def observation_info(self):
        return self.car.get_data()
    
    def perform_action(self, delta_action):
        if self.race_terminated != 0:
            self.controller.reset()
            self.controller.update()
            return
        
        action = self.previous_action + delta_action
        action = np.clip(action, -1.0, 1.0)
        self.previous_action = action
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


        
    """
    def compute_reward(self, info, distances, action):
    
        normalized_speed = info['speed'] / 1000
        normalized_time = info['time'] / 40
        normalized_distance = info['total_progress'] / len(self.map.block_path)
        distance_per_time = 0
        if normalized_time > 0:
            distance_per_time = normalized_distance / normalized_time
        reward = info['map_progress'] + distance_per_time + normalized_speed
        reward += action[0] * 0.1
        print(self.current_step, ":", reward, end='\r')

        # punishments
        reward -= action[1] * 0.1
        for distance in distances:
            if distance < 3:
                reward -= 1
                break
        if info['speed'] < 3:
            reward -= 1
        return reward
    """
    
    def compute_reward(self, info, distances, action):
        # 1) progres po trati - hlavný signál
        progress = info["map_progress"]          # -1, 0, 1

        # 2) rýchlosť a zarovnanie k ďalšiemu bodu
        speed = info["speed"]                    # 0..1000
        align = info.get("next_point_direction", 0.0)  # -1..1, takto to máš v Car.get_data

        reward = 0.0

        # hlavný reward za posun po tiles
        reward += 2.0 * progress                 # výrazne zvýrazní rozdiel medzi -1,0,1

        # bonus za rýchlosť (max ~2, ak speed ~1000)
        reward += 0.002 * speed

        # bonus za smerovanie k ďalšiemu bodu
        reward += 0.5 * align

        # jemné preferovanie plynu a trest za brzdu
        reward += 0.05 * action[0]
        reward -= 0.05 * action[1]

        # trest za blízkosť steny
        if min(distances) < 3.0:
            reward -= 2.0

        # trest za státie
        if speed < 3:
            reward -= 1.0

        return reward



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
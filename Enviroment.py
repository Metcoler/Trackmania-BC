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
        # Observation normalization constants (kept explicit for easier tuning).
        self.laser_max_distance = 320.0
        self.path_instruction_abs_max = 4.0
        self.speed_abs_max = 1000.0
        self.side_speed_abs_max = 1000.0

        # Observations: [lasers N] + [path instructions M] + [speed, side_speed, next_point_dir] + [prev action 3]
        obs_dim = Car.NUM_LASERS + Car.SIGHT_TILES + 6
        obs_low = np.array(
            [0.0] * Car.NUM_LASERS + [-1.0] * (Car.SIGHT_TILES + 6),
            dtype=np.float32,
        )
        obs_high = np.array(
            [1.0] * Car.NUM_LASERS + [1.0] * (Car.SIGHT_TILES + 6),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(obs_dim,),
            dtype=np.float32,
        )

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
        self.previous_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.previous_observation = observation = self.build_observation(
            distances=distances,
            instructions=instructions,
            info=info,
        )
        
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
            safe_action = np.asarray(action, dtype=np.float32)
            if safe_action.shape != (3,) or not np.all(np.isfinite(safe_action)):
                safe_action = np.zeros(3, dtype=np.float32)
            self.perform_action(safe_action)
            distances, instructions, info = self.observation_info()
            observation = self.build_observation(
                distances=distances,
                instructions=instructions,
                info=info,
            )
            self.previous_observation_info = distances, instructions, info
            self.previous_observation = observation
         
             
        timed_out = False
        if info["time"] >= self.max_time:
            self.controller.reset()
            self.controller.update()
            self.race_terminated = 0
            timed_out = True
            truncated = True

        reward = self.compute_reward(info, distances, safe_action)
        
        if info["done"] == 1.0:
            self.controller.reset()
            self.controller.update()
            self.race_terminated = 1
        elif not timed_out and info["next_point_direction"] < -0.5 and not self.never_quit:
            self.controller.reset()
            self.controller.update()
            self.race_terminated = -1
        elif not timed_out:
             
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

    def _fit_vector(self, values, expected_size: int, pad_value: float):
        v = np.asarray(values, dtype=np.float32).reshape(-1)
        if v.size < expected_size:
            v = np.pad(v, (0, expected_size - v.size), constant_values=pad_value)
        elif v.size > expected_size:
            v = v[:expected_size]
        return v

    def build_observation(self, distances, instructions, info):
        distances_vec = self._fit_vector(
            values=distances,
            expected_size=Car.NUM_LASERS,
            pad_value=self.laser_max_distance,
        )
        instructions_vec = self._fit_vector(
            values=instructions,
            expected_size=Car.SIGHT_TILES,
            pad_value=0.0,
        )

        # Normalize to stable ranges for tanh-based policy.
        distances_norm = np.clip(distances_vec / self.laser_max_distance, 0.0, 1.0)
        instructions_norm = np.clip(
            instructions_vec / self.path_instruction_abs_max, -1.0, 1.0
        )
        speed_norm = float(
            np.clip(info.get("speed", 0.0) / self.speed_abs_max, -1.0, 1.0)
        )
        side_speed_norm = float(
            np.clip(info.get("side_speed", 0.0) / self.side_speed_abs_max, -1.0, 1.0)
        )
        next_point_direction = float(
            np.clip(info.get("next_point_direction", 1.0), -1.0, 1.0)
        )

        return np.concatenate(
            [
                distances_norm,
                instructions_norm,
                np.array(
                    [speed_norm, side_speed_norm, next_point_direction],
                    dtype=np.float32,
                ),
                self.previous_action.astype(np.float32, copy=False),
            ]
        ).astype(np.float32, copy=False)
    
    def perform_action(self, delta_action):
        if self.race_terminated != 0:
            self.controller.reset()
            self.controller.update()
            return

        delta_action = np.asarray(delta_action, dtype=np.float32)
        if delta_action.shape != (3,) or not np.all(np.isfinite(delta_action)):
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

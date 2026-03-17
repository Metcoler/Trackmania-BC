import gymnasium as gym
import numpy as np
import socket
import vgamepad
import time

from Car import Car
from Map import Map


class RacingGameEnviroment(gym.Env):
    # Increased step budget for TM GA training; fast-fail guards (start idle / stall / touches)
    # prevent wasting too much time on dead individuals.
    STEPS = 2**14

    def __init__(
        self,
        map_name,
        max_time=60,
        never_quit=False,
        action_mode: str = "delta",
        dt_ref: float = 1.0 / 100.0,
        dt_ratio_clip: float = 3.0,
        max_touches: int = 1,
        touch_distance_threshold: float = 2.0,
        touch_release_distance_threshold: float = 4.0,
        wall_ride_max_contact_time: float = 0.5,
        stall_speed_threshold: float = 3.0,
        stall_release_speed_threshold: float = 6.0,
        stall_min_time: float = 3.0,
        stall_progress_epsilon: float = 0.5,
        stuck_timeout_speed_threshold: float = 2.5,
        stuck_timeout_duration: float = 2.5,
        start_idle_max_time: float = 5.0,
        start_idle_progress_epsilon: float = 0.5,
        start_idle_speed_threshold: float = 3.0,
    ) -> None:
        super().__init__()
        print("Creating the RacingGameEnviroment")
        # Observation normalization constants (kept explicit for easier tuning).
        self.laser_max_distance = 320.0
        self.path_instruction_abs_max = 4.0
        self.speed_abs_max = 1000.0
        self.side_speed_abs_max = 1000.0
        self.dt_ref = float(dt_ref)
        if not np.isfinite(self.dt_ref) or self.dt_ref <= 0.0:
            raise ValueError("dt_ref must be a positive finite number.")
        self.dt_ratio_clip = float(dt_ratio_clip)
        if not np.isfinite(self.dt_ratio_clip) or self.dt_ratio_clip <= 0.0:
            raise ValueError("dt_ratio_clip must be a positive finite number.")

        # Observations:
        # [lasers N] + [path instructions M] + [speed, side_speed, next_point_dir, dt_ratio] + [prev action 3]
        obs_dim = Car.NUM_LASERS + Car.SIGHT_TILES + 7
        obs_low = np.array(
            [0.0] * Car.NUM_LASERS
            + [-1.0] * Car.SIGHT_TILES
            + [-1.0, -1.0, -1.0, 0.0]
            + [-1.0, -1.0, -1.0],
            dtype=np.float32,
        )
        obs_high = np.array(
            [1.0] * Car.NUM_LASERS
            + [1.0] * Car.SIGHT_TILES
            + [1.0, 1.0, 1.0, self.dt_ratio_clip]
            + [1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        action_mode = str(action_mode).strip().lower()
        if action_mode not in {"delta", "target"}:
            raise ValueError("action_mode must be 'delta' or 'target'.")
        self.action_mode = action_mode

        # Action semantics:
        # - delta: per-step action delta (legacy behavior)
        # - target: policy output is treated as direct target action
        if self.action_mode == "delta":
            self.action_space = gym.spaces.Box(low=-0.2, high=0.2, shape=(3,), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
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
        self.max_touches = max(1, int(max_touches))
        self.touch_distance_threshold = float(touch_distance_threshold)
        self.touch_release_distance_threshold = float(touch_release_distance_threshold)
        self.wall_ride_max_contact_time = float(wall_ride_max_contact_time)
        self.stall_speed_threshold = float(stall_speed_threshold)
        self.stall_release_speed_threshold = float(stall_release_speed_threshold)
        self.stall_min_time = float(stall_min_time)
        self.stall_progress_epsilon = float(stall_progress_epsilon)
        self.stuck_timeout_speed_threshold = float(stuck_timeout_speed_threshold)
        self.stuck_timeout_duration = float(stuck_timeout_duration)
        self.start_idle_max_time = float(start_idle_max_time)
        self.start_idle_progress_epsilon = float(start_idle_progress_epsilon)
        self.start_idle_speed_threshold = float(start_idle_speed_threshold)
        self.touch_count = 0
        self._laser_touch_latched = False
        self._stall_touch_latched = False
        self._wall_contact_since_time = None
        self._stuck_since_time = None
        self.previous_game_time = None
        self.current_dt_ratio = 1.0

    
    def reset(self, seed=None):
        
        super().reset(seed=seed)
        self.reset_game()
        self.current_step = 0
        self.previous_game_time = None
        self.current_dt_ratio = 1.0
        self.previous_observation_info = distances, instructions, info = self.observation_info()
        self.previous_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.previous_observation = observation = self.build_observation(
            distances=distances,
            instructions=instructions,
            info=info,
        )
        
        self.race_terminated = 0 
        self.touch_count = 0
        self._laser_touch_latched = False
        self._stall_touch_latched = False
        self._wall_contact_since_time = None
        self._stuck_since_time = None

        
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
        info["touch_count"] = int(self.touch_count)
        info["max_touches"] = int(self.max_touches)
        
        if info["done"] == 1.0:
            self.controller.reset()
            self.controller.update()
            self.race_terminated = 1
        elif not timed_out and info["next_point_direction"] < -0.5 and not self.never_quit:
            self.controller.reset()
            self.controller.update()
            self.race_terminated = -1
        elif not timed_out:
            speed = float(info.get("speed", 0.0))
            total_progress = float(info.get("total_progress", 0.0))
            current_time = float(info.get("time", 0.0))

            # Fast-fail: car is still essentially at start and not moving after a few seconds.
            # This avoids wasting evaluation time on dead individuals.
            if (
                (current_time > self.start_idle_max_time)
                and (total_progress <= self.start_idle_progress_epsilon)
                and (speed < self.start_idle_speed_threshold)
                and (not self.never_quit)
            ):
                self.race_terminated = -1
                self.controller.reset()
                self.controller.update()
                reward = 0
                info["touch_reason"] = "start_idle"
                info["touch_count"] = int(self.touch_count)
                print(self.current_step, ":", info["speed"], " "*20,  end='\r')
                return observation, reward, done, truncated, info

            # Fast timeout: car already moved away from start but remains essentially stopped
            # for a while. This avoids waiting until the global max_time when the agent is dead.
            if (
                (not self.never_quit)
                and (total_progress > self.stall_progress_epsilon)
                and (speed < self.stuck_timeout_speed_threshold)
            ):
                if self._stuck_since_time is None:
                    self._stuck_since_time = current_time
                elif (current_time - float(self._stuck_since_time)) >= self.stuck_timeout_duration:
                    self.controller.reset()
                    self.controller.update()
                    self.race_terminated = 0
                    truncated = True
                    reward = 0
                    info["timeout_reason"] = "stuck_after_progress"
                    info["touch_count"] = int(self.touch_count)
                    print(self.current_step, ":", info["speed"], " "*20,  end='\r')
                    return observation, reward, done, truncated, info
            else:
                self._stuck_since_time = None

            min_distance = min(distances) if len(distances) > 0 else self.laser_max_distance
            if min_distance > self.touch_release_distance_threshold:
                self._laser_touch_latched = False
                self._wall_contact_since_time = None

            # Anti wall-ride: if the car stays in wall contact continuously for too long
            # (using the same collision threshold as touch detection), terminate the episode.
            if (
                (not self.never_quit)
                and (total_progress > self.stall_progress_epsilon)
                and (min_distance < self.touch_distance_threshold)
            ):
                if self._wall_contact_since_time is None:
                    self._wall_contact_since_time = current_time
                elif (current_time - float(self._wall_contact_since_time)) >= self.wall_ride_max_contact_time:
                    self.controller.reset()
                    self.controller.update()
                    if self.touch_count <= 0:
                        self.touch_count = 1
                    self.race_terminated = -int(self.touch_count)
                    reward = 0
                    info["touch_count"] = int(self.touch_count)
                    info["touch_reason"] = "wall_ride"
                    print(self.current_step, ":", info["speed"], " "*20,  end='\r')
                    return observation, reward, done, truncated, info
            elif min_distance > self.touch_release_distance_threshold:
                self._wall_contact_since_time = None

            laser_touch_event = (
                (min_distance < self.touch_distance_threshold)
                and (not self._laser_touch_latched)
                and (not self.never_quit)
            )
            if laser_touch_event:
                self._laser_touch_latched = True

            if speed > self.stall_release_speed_threshold:
                self._stall_touch_latched = False

            stall_touch_event = (
                (speed < self.stall_speed_threshold)
                and (float(info.get("time", 0.0)) > self.stall_min_time)
                and (total_progress > self.stall_progress_epsilon)
                and (not self._stall_touch_latched)
                and (not self.never_quit)
            )
            if stall_touch_event:
                self._stall_touch_latched = True

            # Count at most one touch event per frame to avoid double-counting
            # (e.g. wall proximity + low speed in the same sample).
            touch_reason = None
            if laser_touch_event:
                touch_reason = "laser"
            elif stall_touch_event:
                touch_reason = "stall"

            if touch_reason is not None:
                self.touch_count += 1
                info["touch_count"] = int(self.touch_count)
                info["touch_reason"] = touch_reason
                reward = 0

                if self.touch_count >= self.max_touches:
                    self.race_terminated = -int(self.touch_count)
                    self.controller.reset()
                    self.controller.update()

        
        
        
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
        dt_ratio = self._update_dt_ratio(float(info.get("time", 0.0)))
        info["dt_ratio"] = dt_ratio

        return np.concatenate(
            [
                distances_norm,
                instructions_norm,
                np.array(
                    [speed_norm, side_speed_norm, next_point_direction, dt_ratio],
                    dtype=np.float32,
                ),
                self.previous_action.astype(np.float32, copy=False),
            ]
        ).astype(np.float32, copy=False)

    def _update_dt_ratio(self, current_time: float) -> float:
        current_time = float(current_time)
        dt = self.dt_ref
        if np.isfinite(current_time) and current_time >= 0.0:
            if (
                self.previous_game_time is not None
                and np.isfinite(self.previous_game_time)
                and current_time >= float(self.previous_game_time)
            ):
                dt = max(1e-6, current_time - float(self.previous_game_time))
            self.previous_game_time = current_time

        dt_ratio = float(np.clip(dt / self.dt_ref, 0.0, self.dt_ratio_clip))
        self.current_dt_ratio = dt_ratio
        return dt_ratio
    
    def perform_action(self, action_input):
        if self.race_terminated != 0:
            self.controller.reset()
            self.controller.update()
            return

        action_input = np.asarray(action_input, dtype=np.float32)
        if action_input.shape != (3,) or not np.all(np.isfinite(action_input)):
            return

        if self.action_mode == "delta":
            scaled_delta = action_input * float(self.current_dt_ratio)
            action = self.previous_action + scaled_delta
            action = np.clip(action, -1.0, 1.0)
        else:
            # Experimental target mode:
            # - policy outputs are expected in [-1, 1]
            # - gas/brake are thresholded to binary triggers (0/1) after remap to [0, 1]
            #   so near-zero outputs can still produce decisive throttle/brake actions
            # - keep steer in [-1, 1]
            # Neutral output (0) maps to 0.5 and stays released because threshold uses > 0.5.
            gas01 = 0.5 * (float(action_input[0]) + 1.0)
            brake01 = 0.5 * (float(action_input[1]) + 1.0)
            action = np.array(
                [
                    1.0 if gas01 > 0.5 else 0.0,
                    1.0 if brake01 > 0.5 else 0.0,
                    action_input[2],
                ],
                dtype=np.float32,
            )

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
        self.controller.update()
        time.sleep(1)

    def close(self):
        # Make sure the virtual gamepad is left in neutral state when the program exits.
        try:
            self.previous_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            if hasattr(self, "controller") and self.controller is not None:
                self.controller.reset()
                self.controller.update()
                # Send neutral twice to reduce the chance of stale state on fast shutdown.
                time.sleep(0.02)
                self.controller.reset()
                self.controller.update()
        except Exception:
            pass
        try:
            super().close()
        except Exception:
            pass


        
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

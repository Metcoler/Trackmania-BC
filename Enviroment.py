import gymnasium as gym
import numpy as np
import socket
import vgamepad
import time

from Car import Car
from Map import Map
from ObservationEncoder import ObservationEncoder


class RacingGameEnviroment(gym.Env):
    RESET_BUTTON_HOLD_SECONDS = 0.30
    RESET_CONFIRM_TIMEOUT_SECONDS = 0.50
    RESET_RETRY_COOLDOWN_SECONDS = 0.20
    RESET_MAX_ATTEMPTS = 10
    RESET_CONFIRM_MIN_TIME_DROP_SECONDS = 0.25
    RESET_CONFIRM_MAX_START_TIME_SECONDS = 0.75
    RESET_CONFIRM_MAX_START_DISTANCE = 5.0
    RESET_CONFIRM_MAX_START_SPEED = 8.0
    RESET_PACKET_TIMEOUT_SECONDS = 0.10
    FINISH_RESET_SETTLE_SECONDS = 0.5

    def __init__(
        self,
        map_name,
        max_time=60,
        never_quit=False,
        action_mode: str = "delta",
        dt_ref: float = 1.0 / 100.0,
        dt_ratio_clip: float = 3.0,
        vertical_mode: bool = True,
        surface_step_size: float = Car.SURFACE_STEP_SIZE,
        surface_probe_height: float = Car.SURFACE_PROBE_HEIGHT,
        surface_ray_lift: float = Car.SURFACE_RAY_LIFT,
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
        self.vertical_mode = bool(vertical_mode)
        self.obs_encoder = ObservationEncoder(
            dt_ref=dt_ref,
            dt_ratio_clip=dt_ratio_clip,
            vertical_mode=self.vertical_mode,
        )
        self.dt_ref = self.obs_encoder.dt_ref
        self.dt_ratio_clip = self.obs_encoder.dt_ratio_clip
        self.laser_max_distance = self.obs_encoder.laser_max_distance
        action_mode = str(action_mode).strip().lower()
        if action_mode not in {"delta", "target"}:
            raise ValueError("action_mode must be 'delta' or 'target'.")
        self.action_mode = action_mode

        # Observations:
        # [lasers N] + [path instructions M] +
        # [speed, side_speed, segment_heading_error, next_segment_heading_error, dt_ratio,
        #  slip_fl, slip_fr, slip_rl, slip_rr,
        #  surface_instruction_0..4,
        #  height_instruction_0..4,
        #  longitudinal_accel, lateral_accel, yaw_rate,
        #  clearance_rate_sector_0..4]
        # optional vertical block (when vertical_mode=True):
        # [vertical_speed, forward_y, support_normal_y, cross_slope,
        #  surface_elevation_sector_0..4]
        obs_dim = self.obs_encoder.obs_dim
        obs_low, obs_high = self.obs_encoder.get_observation_bounds(action_mode=self.action_mode)
        self.observation_space = gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action semantics:
        # - delta: per-step action delta (legacy behavior)
        # - target: policy output is treated as direct target action
        if self.action_mode == "delta":
            self.action_space = gym.spaces.Box(low=-0.2, high=0.2, shape=(3,), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Box(
                low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                shape=(3,),
                dtype=np.float32,
            )
        #self.action_space = gym.spaces.Discrete(8)
        
        # Create the car and the map
        self.map = Map(map_name)
        self.car = Car(
            self.map,
            vertical_mode=self.vertical_mode,
            surface_step_size=surface_step_size,
            surface_probe_height=surface_probe_height,
            surface_ray_lift=surface_ray_lift,
        )
        
        # Gamepad
        self.controller = vgamepad.VX360Gamepad()
        self.controller.reset()

        self.max_time = max_time
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
        self.last_reset_attempts = 0
        self.last_reset_seconds = 0.0
        self._post_finish_reset_pending = False
        self.current_dt_ratio = 1.0

    def _clear_episode_runtime_state(self) -> None:
        self.current_step = 0
        self.obs_encoder.reset()
        self.current_dt_ratio = 1.0
        self.previous_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.previous_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.previous_observation_info = (
            [self.laser_max_distance for _ in range(Car.NUM_LASERS)],
            [0 for _ in range(Car.SIGHT_TILES)],
            {"time": -1.0, "done": 0.0},
        )
        self.race_terminated = 0
        self.touch_count = 0
        self._laser_touch_latched = False
        self._stall_touch_latched = False
        self._wall_contact_since_time = None
        self._stuck_since_time = None
        self.car.reset()

    def _neutralize_controller(self) -> None:
        self.previous_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.controller.reset()
        self.controller.update()

    def _press_restart_button(self) -> None:
        self._neutralize_controller()
        self.controller.press_button(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_B)
        self.controller.update()
        time.sleep(self.RESET_BUTTON_HOLD_SECONDS)
        self.controller.release_button(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_B)
        self.controller.update()
        self._neutralize_controller()

    def _is_reset_confirmed(
        self,
        info,
        baseline_time: float | None,
        baseline_distance: float | None,
        require_start_state: bool = False,
    ) -> bool:
        current_time = float(info.get("time", 0.0))
        if current_time < 0.0:
            return True
        if float(info.get("done", 0.0)) == 1.0:
            return False

        if (
            not require_start_state
            and np.isfinite(current_time)
            and baseline_time is not None
            and np.isfinite(baseline_time)
        ):
            if current_time < (float(baseline_time) - self.RESET_CONFIRM_MIN_TIME_DROP_SECONDS):
                return True

        current_distance = float(info.get("distance", 0.0))
        current_speed = abs(float(info.get("speed", 0.0)))
        current_progress = float(info.get("total_progress", 0.0))
        near_fresh_start = (
            current_time <= self.RESET_CONFIRM_MAX_START_TIME_SECONDS
            and current_distance <= self.RESET_CONFIRM_MAX_START_DISTANCE
            and current_speed <= self.RESET_CONFIRM_MAX_START_SPEED
            and current_progress <= max(1.0, self.start_idle_progress_epsilon)
        )
        if not near_fresh_start:
            return False

        if baseline_time is not None and np.isfinite(baseline_time):
            if current_time < (float(baseline_time) - self.RESET_CONFIRM_MIN_TIME_DROP_SECONDS):
                return True

        if baseline_distance is not None and np.isfinite(baseline_distance):
            if current_distance < (float(baseline_distance) - 1.0):
                return True

        return False

    def _wait_for_reset_confirmation(
        self,
        baseline_time: float | None,
        baseline_distance: float | None,
        timeout_seconds: float,
        require_start_state: bool = False,
    ):
        deadline = time.monotonic() + float(timeout_seconds)
        last_payload = None
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            payload = self.observation_info(
                timeout_seconds=min(self.RESET_PACKET_TIMEOUT_SECONDS, remaining)
            )
            if payload is None:
                continue
            distances, instructions, info = payload
            last_payload = (distances, instructions, info)
            if self._is_reset_confirmed(
                info=info,
                baseline_time=baseline_time,
                baseline_distance=baseline_distance,
                require_start_state=require_start_state,
            ):
                return distances, instructions, info
        return last_payload

    def _drain_post_finish_packets(self) -> None:
        deadline = time.monotonic() + self.FINISH_RESET_SETTLE_SECONDS
        while time.monotonic() < deadline:
            self._neutralize_controller()
            remaining = max(0.0, deadline - time.monotonic())
            _ = self.observation_info(
                timeout_seconds=min(self.RESET_PACKET_TIMEOUT_SECONDS, remaining)
            )

    def _reset_track_until_confirmed(self, post_finish: bool = False):
        reset_started_at = time.monotonic()
        self.last_reset_attempts = 0
        self.last_reset_seconds = 0.0
        if post_finish:
            self._drain_post_finish_packets()

        baseline_payload = self.observation_info(timeout_seconds=self.RESET_PACKET_TIMEOUT_SECONDS)
        baseline_info = None if baseline_payload is None else baseline_payload[2]
        baseline_time = None if baseline_info is None else float(baseline_info.get("time", 0.0))
        baseline_distance = None if baseline_info is None else float(baseline_info.get("distance", 0.0))
        if baseline_payload is not None and not post_finish:
            baseline_distances, baseline_instructions, baseline_info = baseline_payload
            if self._is_reset_confirmed(
                info=baseline_info,
                baseline_time=None,
                baseline_distance=None,
                require_start_state=False,
            ):
                self.last_reset_seconds = time.monotonic() - reset_started_at
                return baseline_distances, baseline_instructions, baseline_info

        last_info = baseline_info
        for attempt in range(1, self.RESET_MAX_ATTEMPTS + 1):
            self.last_reset_attempts = attempt
            self._press_restart_button()
            payload = self._wait_for_reset_confirmation(
                baseline_time=baseline_time,
                baseline_distance=baseline_distance,
                timeout_seconds=self.RESET_CONFIRM_TIMEOUT_SECONDS,
                require_start_state=post_finish,
            )
            if payload is not None:
                distances, instructions, info = payload
                last_info = info
                if self._is_reset_confirmed(
                    info=info,
                    baseline_time=baseline_time,
                    baseline_distance=baseline_distance,
                    require_start_state=post_finish,
                ):
                    self.last_reset_seconds = time.monotonic() - reset_started_at
                    return distances, instructions, info
                baseline_time = float(info.get("time", baseline_time))
                baseline_distance = float(info.get("distance", baseline_distance))
            if attempt < self.RESET_MAX_ATTEMPTS:
                time.sleep(self.RESET_RETRY_COOLDOWN_SECONDS)

        last_time = None if last_info is None else float(last_info.get("time", float("nan")))
        self.last_reset_seconds = time.monotonic() - reset_started_at
        raise RuntimeError(
            "Failed to confirm Trackmania reset after "
            f"{self.RESET_MAX_ATTEMPTS} B-button attempts; last observed time={last_time!r}."
        )

    
    def reset(self, seed=None):
        super().reset(seed=seed)
        post_finish = bool(self._post_finish_reset_pending or self.race_terminated == 1)
        self._neutralize_controller()
        distances, instructions, info = self._reset_track_until_confirmed(post_finish=post_finish)
        self._clear_episode_runtime_state()
        self._post_finish_reset_pending = False
        self.previous_observation = observation = self.build_observation(
            distances=distances,
            instructions=instructions,
            info=info,
        )
        self.previous_observation_info = (distances, instructions, info)
        return observation, info

    def step(self, action):
        done = False
        truncated = False
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

        reward = 0.0
        info["touch_count"] = int(self.touch_count)
        info["max_touches"] = int(self.max_touches)
        
        if info["done"] == 1.0:
            self.controller.reset()
            self.controller.update()
            self.race_terminated = 1
            self._post_finish_reset_pending = True
        elif (
            not timed_out
            and abs(float(info.get("segment_heading_error", 0.0))) > (2.0 / 3.0)
            and not self.never_quit
        ):
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
                    self.race_terminated = -1
                    reward = 0
                    info["touch_reason"] = "stuck_after_progress"
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

    def observation_info(self, timeout_seconds: float | None = None):
        return self.car.get_data(timeout_seconds=timeout_seconds)

    def build_observation(self, distances, instructions, info):
        observation = self.obs_encoder.build_observation(
            distances=distances,
            instructions=instructions,
            info=info,
        )
        self.current_dt_ratio = self.obs_encoder.current_dt_ratio
        return observation

    def _update_dt_ratio(self, current_time: float) -> float:
        dt_ratio = self.obs_encoder.update_dt_ratio(current_time)
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
            gas01 = 0.5 * (float(action_input[0]) + 1.0)
            brake01 = 0.5 * (float(action_input[1]) + 1.0)
            if 0.0 <= float(action_input[0]) <= 1.0 and 0.0 <= float(action_input[1]) <= 1.0:
                gas01 = float(action_input[0])
                brake01 = float(action_input[1])
            gas01 = 1.0 if gas01 > 0.5 else 0.0
            brake01 = 1.0 if brake01 > 0.5 else 0.0
            action = np.array(
                [
                    float(np.clip(gas01, 0.0, 1.0)),
                    float(np.clip(brake01, 0.0, 1.0)),
                    float(np.clip(action_input[2], -1.0, 1.0)),
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
        post_finish = bool(self._post_finish_reset_pending or self.race_terminated == 1)
        self._neutralize_controller()
        distances, instructions, info = self._reset_track_until_confirmed(post_finish=post_finish)
        self._clear_episode_runtime_state()
        self._post_finish_reset_pending = False
        self.previous_observation_info = (distances, instructions, info)
        self.previous_observation = self.build_observation(
            distances=distances,
            instructions=instructions,
            info=info,
        )

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

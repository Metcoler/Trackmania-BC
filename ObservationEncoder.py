import numpy as np

from Car import Car


class ObservationEncoder:
    def __init__(
        self,
        laser_max_distance: float = 320.0,
        path_instruction_abs_max: float = 4.0,
        speed_abs_max: float = 1000.0,
        side_speed_abs_max: float = 1000.0,
        dt_ref: float = 1.0 / 100.0,
        dt_ratio_clip: float = 3.0,
    ) -> None:
        self.laser_max_distance = float(laser_max_distance)
        self.path_instruction_abs_max = float(path_instruction_abs_max)
        self.speed_abs_max = float(speed_abs_max)
        self.side_speed_abs_max = float(side_speed_abs_max)
        self.dt_ref = float(dt_ref)
        self.dt_ratio_clip = float(dt_ratio_clip)
        if not np.isfinite(self.dt_ref) or self.dt_ref <= 0.0:
            raise ValueError("dt_ref must be a positive finite number.")
        if not np.isfinite(self.dt_ratio_clip) or self.dt_ratio_clip <= 0.0:
            raise ValueError("dt_ratio_clip must be a positive finite number.")
        self.previous_game_time = None
        self.current_dt_ratio = 1.0

    @property
    def obs_dim(self) -> int:
        return Car.NUM_LASERS + Car.SIGHT_TILES + 7

    def reset(self) -> None:
        self.previous_game_time = None
        self.current_dt_ratio = 1.0

    def get_observation_bounds(self, action_mode: str = "delta"):
        mode = str(action_mode).strip().lower()
        if mode == "target":
            prev_low = [0.0, 0.0, -1.0]
            prev_high = [1.0, 1.0, 1.0]
        else:
            prev_low = [-1.0, -1.0, -1.0]
            prev_high = [1.0, 1.0, 1.0]

        obs_low = np.array(
            [0.0] * Car.NUM_LASERS
            + [-1.0] * Car.SIGHT_TILES
            + [-1.0, -1.0, -1.0, 0.0]
            + prev_low,
            dtype=np.float32,
        )
        obs_high = np.array(
            [1.0] * Car.NUM_LASERS
            + [1.0] * Car.SIGHT_TILES
            + [1.0, 1.0, 1.0, self.dt_ratio_clip]
            + prev_high,
            dtype=np.float32,
        )
        return obs_low, obs_high

    @staticmethod
    def fit_vector(values, expected_size: int, pad_value: float) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float32).reshape(-1)
        if vector.size < expected_size:
            vector = np.pad(vector, (0, expected_size - vector.size), constant_values=pad_value)
        elif vector.size > expected_size:
            vector = vector[:expected_size]
        return vector

    def update_dt_ratio(self, current_time: float) -> float:
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

    def build_observation(self, distances, instructions, info, previous_action) -> np.ndarray:
        distances_vec = self.fit_vector(
            values=distances,
            expected_size=Car.NUM_LASERS,
            pad_value=self.laser_max_distance,
        )
        instructions_vec = self.fit_vector(
            values=instructions,
            expected_size=Car.SIGHT_TILES,
            pad_value=0.0,
        )
        previous_action_vec = self.fit_vector(
            values=previous_action,
            expected_size=3,
            pad_value=0.0,
        )

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
        dt_ratio = self.update_dt_ratio(float(info.get("time", 0.0)))
        info["dt_ratio"] = dt_ratio

        return np.concatenate(
            [
                distances_norm,
                instructions_norm,
                np.array(
                    [speed_norm, side_speed_norm, next_point_direction, dt_ratio],
                    dtype=np.float32,
                ),
                previous_action_vec.astype(np.float32, copy=False),
            ]
        ).astype(np.float32, copy=False)

    @staticmethod
    def mirror_observation(obs) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).copy()
        if x.ndim != 1:
            return x

        min_expected = Car.NUM_LASERS + Car.SIGHT_TILES + 7
        if x.shape[0] < min_expected:
            return x

        x[: Car.NUM_LASERS] = x[: Car.NUM_LASERS][::-1]

        instr_slice = slice(Car.NUM_LASERS, Car.NUM_LASERS + Car.SIGHT_TILES)
        x[instr_slice] = -x[instr_slice]

        aux_offset = Car.NUM_LASERS + Car.SIGHT_TILES
        side_speed_idx = aux_offset + 1
        prev_action_steer_idx = aux_offset + 6
        x[side_speed_idx] = -x[side_speed_idx]
        x[prev_action_steer_idx] = -x[prev_action_steer_idx]
        return x

    @staticmethod
    def mirror_action(action) -> np.ndarray:
        mirrored = np.asarray(action, dtype=np.float32).copy()
        if mirrored.ndim == 1 and mirrored.shape[0] >= 3:
            mirrored[2] = -mirrored[2]
        return mirrored

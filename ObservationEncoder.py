import numpy as np

from Car import Car


class ObservationEncoder:
    BASE_FEATURE_NAMES = (
        "speed",
        "side_speed",
        "segment_heading_error",
        "next_segment_heading_error",
        "dt_ratio",
    )
    WHEEL_SLIP_FEATURE_NAMES = (
        "slip_fl",
        "slip_fr",
        "slip_rl",
        "slip_rr",
    )
    TEMPORAL_FEATURE_NAMES = (
        "longitudinal_accel",
        "lateral_accel",
        "yaw_rate",
        "clearance_rate_sector_0",
        "clearance_rate_sector_1",
        "clearance_rate_sector_2",
        "clearance_rate_sector_3",
        "clearance_rate_sector_4",
    )

    def __init__(
        self,
        laser_max_distance: float = 320.0,
        path_instruction_abs_max: float = 4.0,
        speed_abs_max: float = 1000.0,
        side_speed_abs_max: float = 1000.0,
        accel_abs_max: float = 80.0,
        yaw_rate_abs_max: float = 8.0,
        clearance_rate_abs_max: float = 200.0,
        dt_ref: float = 1.0 / 100.0,
        dt_ratio_clip: float = 3.0,
    ) -> None:
        self.laser_max_distance = float(laser_max_distance)
        self.path_instruction_abs_max = float(path_instruction_abs_max)
        self.speed_abs_max = float(speed_abs_max)
        self.side_speed_abs_max = float(side_speed_abs_max)
        self.accel_abs_max = float(accel_abs_max)
        self.yaw_rate_abs_max = float(yaw_rate_abs_max)
        self.clearance_rate_abs_max = float(clearance_rate_abs_max)
        self.dt_ref = float(dt_ref)
        self.dt_ratio_clip = float(dt_ratio_clip)
        if not np.isfinite(self.dt_ref) or self.dt_ref <= 0.0:
            raise ValueError("dt_ref must be a positive finite number.")
        if not np.isfinite(self.dt_ratio_clip) or self.dt_ratio_clip <= 0.0:
            raise ValueError("dt_ratio_clip must be a positive finite number.")
        if not np.isfinite(self.accel_abs_max) or self.accel_abs_max <= 0.0:
            raise ValueError("accel_abs_max must be a positive finite number.")
        if not np.isfinite(self.yaw_rate_abs_max) or self.yaw_rate_abs_max <= 0.0:
            raise ValueError("yaw_rate_abs_max must be a positive finite number.")
        if not np.isfinite(self.clearance_rate_abs_max) or self.clearance_rate_abs_max <= 0.0:
            raise ValueError("clearance_rate_abs_max must be a positive finite number.")
        self.previous_game_time = None
        self.current_dt = self.dt_ref
        self.current_dt_ratio = 1.0
        self.previous_speed = None
        self.previous_side_speed = None
        self.previous_yaw = None
        self.previous_clearance_windows = None

    @property
    def obs_dim(self) -> int:
        return (
            self.base_obs_dim()
            + len(self.WHEEL_SLIP_FEATURE_NAMES)
            + len(self.TEMPORAL_FEATURE_NAMES)
        )

    @classmethod
    def base_obs_dim(cls) -> int:
        return Car.NUM_LASERS + Car.SIGHT_TILES + len(cls.BASE_FEATURE_NAMES)

    @classmethod
    def slip_obs_dim(cls) -> int:
        return cls.base_obs_dim() + len(cls.WHEEL_SLIP_FEATURE_NAMES)

    @classmethod
    def feature_names(cls) -> list[str]:
        return (
            [f"laser_{i}" for i in range(Car.NUM_LASERS)]
            + [f"path_instruction_{i}" for i in range(Car.SIGHT_TILES)]
            + list(cls.BASE_FEATURE_NAMES)
            + list(cls.WHEEL_SLIP_FEATURE_NAMES)
            + list(cls.TEMPORAL_FEATURE_NAMES)
        )

    def reset(self) -> None:
        self.previous_game_time = None
        self.current_dt = self.dt_ref
        self.current_dt_ratio = 1.0
        self.previous_speed = None
        self.previous_side_speed = None
        self.previous_yaw = None
        self.previous_clearance_windows = None

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    @classmethod
    def clearance_window_bounds(cls) -> list[tuple[int, int]]:
        num_lasers = int(Car.NUM_LASERS)
        window_count = len(cls.TEMPORAL_FEATURE_NAMES) - 3
        window_size = max(1, min(num_lasers, int(np.ceil(num_lasers / 3.0))))
        max_start = max(num_lasers - window_size, 0)
        starts = np.rint(np.linspace(0, max_start, num=window_count)).astype(np.int32)
        return [(int(start), int(min(start + window_size, num_lasers))) for start in starts]

    def _extract_yaw(self, info) -> float | None:
        dx = float(info.get("dx", 0.0))
        dz = float(info.get("dz", 0.0))
        norm = float(np.hypot(dx, dz))
        if not np.isfinite(norm) or norm <= 1e-6:
            return None
        return float(np.arctan2(dz, dx))

    def _compute_clearance_windows(self, distances_vec: np.ndarray) -> np.ndarray:
        windows = [
            float(np.mean(distances_vec[start:end]))
            for start, end in self.clearance_window_bounds()
        ]
        return np.asarray(windows, dtype=np.float32)

    def get_observation_bounds(self, action_mode: str = "delta"):
        obs_low = np.array(
            [0.0] * Car.NUM_LASERS
            + [-1.0] * Car.SIGHT_TILES
            + [-1.0, -1.0, -1.0, -1.0, 0.0]
            + [0.0] * len(self.WHEEL_SLIP_FEATURE_NAMES),
            dtype=np.float32,
        )
        temporal_low = np.array(
            [-1.0] * len(self.TEMPORAL_FEATURE_NAMES),
            dtype=np.float32,
        )
        obs_high = np.array(
            [1.0] * Car.NUM_LASERS
            + [1.0] * Car.SIGHT_TILES
            + [1.0, 1.0, 1.0, 1.0, self.dt_ratio_clip]
            + [1.0] * len(self.WHEEL_SLIP_FEATURE_NAMES),
            dtype=np.float32,
        )
        temporal_high = np.array(
            [1.0] * len(self.TEMPORAL_FEATURE_NAMES),
            dtype=np.float32,
        )
        return (
            np.concatenate([obs_low, temporal_low]).astype(np.float32, copy=False),
            np.concatenate([obs_high, temporal_high]).astype(np.float32, copy=False),
        )

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
        else:
            self.previous_game_time = None

        self.current_dt = float(dt)
        dt_ratio = float(np.clip(dt / self.dt_ref, 0.0, self.dt_ratio_clip))
        self.current_dt_ratio = dt_ratio
        return dt_ratio

    def build_observation(self, distances, instructions, info) -> np.ndarray:
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
        segment_heading_error = float(
            np.clip(info.get("segment_heading_error", 0.0), -1.0, 1.0)
        )
        next_segment_heading_error = float(
            np.clip(info.get("next_segment_heading_error", 0.0), -1.0, 1.0)
        )
        current_time = float(info.get("time", 0.0))
        previous_game_time = self.previous_game_time
        reset_like_time = (
            not np.isfinite(current_time)
            or (current_time < 0.0)
            or (
                previous_game_time is not None
                and np.isfinite(previous_game_time)
                and current_time < float(previous_game_time)
            )
        )
        if reset_like_time:
            self.previous_speed = None
            self.previous_side_speed = None
            self.previous_yaw = None
            self.previous_clearance_windows = None

        dt_ratio = self.update_dt_ratio(current_time)
        info["dt_ratio"] = dt_ratio
        wheel_slips = self.fit_vector(
            values=[
                info.get("slip_fl", 0.0),
                info.get("slip_fr", 0.0),
                info.get("slip_rl", 0.0),
                info.get("slip_rr", 0.0),
            ],
            expected_size=len(self.WHEEL_SLIP_FEATURE_NAMES),
            pad_value=0.0,
        )
        wheel_slips = np.clip(wheel_slips, 0.0, 1.0)
        current_yaw = self._extract_yaw(info)
        current_clearance_windows = self._compute_clearance_windows(distances_vec)

        speed = float(info.get("speed", 0.0))
        side_speed = float(info.get("side_speed", 0.0))
        dt = max(1e-6, float(self.current_dt))

        longitudinal_accel = 0.0
        lateral_accel = 0.0
        yaw_rate = 0.0
        clearance_rates = np.zeros(len(self.TEMPORAL_FEATURE_NAMES) - 3, dtype=np.float32)

        if self.previous_speed is not None:
            longitudinal_accel = (speed - float(self.previous_speed)) / dt
        if self.previous_side_speed is not None:
            lateral_accel = (side_speed - float(self.previous_side_speed)) / dt
        if self.previous_yaw is not None and current_yaw is not None:
            yaw_delta = self._wrap_angle(current_yaw - float(self.previous_yaw))
            yaw_rate = yaw_delta / dt
        if self.previous_clearance_windows is not None:
            clearance_rates = (
                current_clearance_windows - np.asarray(self.previous_clearance_windows, dtype=np.float32)
            ) / dt

        info["longitudinal_accel"] = float(longitudinal_accel)
        info["lateral_accel"] = float(lateral_accel)
        info["yaw_rate"] = float(yaw_rate)
        for idx, value in enumerate(clearance_rates):
            info[f"clearance_rate_sector_{idx}"] = float(value)

        temporal_features = np.array(
            [
                np.clip(longitudinal_accel / self.accel_abs_max, -1.0, 1.0),
                np.clip(lateral_accel / self.accel_abs_max, -1.0, 1.0),
                np.clip(yaw_rate / self.yaw_rate_abs_max, -1.0, 1.0),
                *np.clip(clearance_rates / self.clearance_rate_abs_max, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

        self.previous_speed = speed
        self.previous_side_speed = side_speed
        self.previous_yaw = current_yaw
        self.previous_clearance_windows = current_clearance_windows.astype(np.float32, copy=True)

        return np.concatenate(
            [
                distances_norm,
                instructions_norm,
                np.array(
                    [
                        speed_norm,
                        side_speed_norm,
                        segment_heading_error,
                        next_segment_heading_error,
                        dt_ratio,
                    ],
                    dtype=np.float32,
                ),
                wheel_slips.astype(np.float32, copy=False),
                temporal_features,
            ]
        ).astype(np.float32, copy=False)

    @staticmethod
    def mirror_observation(obs) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).copy()
        if x.ndim != 1:
            return x

        min_expected = ObservationEncoder.base_obs_dim() + len(
            ObservationEncoder.WHEEL_SLIP_FEATURE_NAMES
        ) + len(ObservationEncoder.TEMPORAL_FEATURE_NAMES)
        if x.shape[0] < min_expected:
            return x

        x[: Car.NUM_LASERS] = x[: Car.NUM_LASERS][::-1]

        instr_slice = slice(Car.NUM_LASERS, Car.NUM_LASERS + Car.SIGHT_TILES)
        x[instr_slice] = -x[instr_slice]

        aux_offset = Car.NUM_LASERS + Car.SIGHT_TILES
        side_speed_idx = aux_offset + 1
        segment_heading_error_idx = aux_offset + 2
        next_segment_heading_error_idx = aux_offset + 3
        x[side_speed_idx] = -x[side_speed_idx]
        x[segment_heading_error_idx] = -x[segment_heading_error_idx]
        x[next_segment_heading_error_idx] = -x[next_segment_heading_error_idx]
        slip_offset = aux_offset + len(ObservationEncoder.BASE_FEATURE_NAMES)
        x[slip_offset : slip_offset + 2] = x[slip_offset : slip_offset + 2][::-1]
        x[slip_offset + 2 : slip_offset + 4] = x[slip_offset + 2 : slip_offset + 4][::-1]
        temporal_offset = slip_offset + len(ObservationEncoder.WHEEL_SLIP_FEATURE_NAMES)
        lateral_accel_idx = temporal_offset + 1
        yaw_rate_idx = temporal_offset + 2
        x[lateral_accel_idx] = -x[lateral_accel_idx]
        x[yaw_rate_idx] = -x[yaw_rate_idx]
        clearance_slice = slice(temporal_offset + 3, temporal_offset + len(ObservationEncoder.TEMPORAL_FEATURE_NAMES))
        x[clearance_slice] = x[clearance_slice][::-1]
        return x

    @staticmethod
    def mirror_action(action) -> np.ndarray:
        mirrored = np.asarray(action, dtype=np.float32).copy()
        if mirrored.ndim == 1 and mirrored.shape[0] >= 3:
            mirrored[2] = -mirrored[2]
        return mirrored

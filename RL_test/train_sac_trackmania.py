import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch.nn as nn
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from Car import Car
from Enviroment import RacingGameEnviroment
from Individual import Individual


MAP_NAME = "AI Training #5"
EPISODES_TO_RUN = 5000
TOTAL_TIMESTEPS = 20_000_000
MAX_RUNTIME_HOURS = 8.0
ENV_MAX_TIME = 30.0
CHECKPOINT_EVERY_EPISODES = 50

# Hybrid reward: sparse progress deltas during the run plus a dominant terminal
# fitness reward using Individual.compute_scalar_fitness_for.
REWARD_MODE = "hybrid_progress_terminal_fitness"
PROGRESS_REWARD_INTERVAL_STEPS = 100
PROGRESS_DELTA_SCALE = 0.10
TERMINAL_FITNESS_SCALE = 1_000_000.0

ACTION_LAYOUT = "gas_brake_steer"  # gas_steer / gas_brake_steer / throttle_steer / target_3d
VERTICAL_MODE = False
MAX_TOUCHES = 1
START_IDLE_MAX_TIME = 2.0

SAC_LEARNING_RATE = 3e-4
SAC_BUFFER_SIZE = 300_000
SAC_LEARNING_STARTS = 1_000
SAC_BATCH_SIZE = 256
SAC_GAMMA = 0.995
SAC_TAU = 0.005
SAC_GRADIENT_STEPS = 64
SAC_TRAIN_FREQ = (1, "episode")
SAC_ENT_COEF = "auto"
SAC_POLICY_NET_ARCH = [128, 128]
SAC_ACTIVATION_FN = "relu"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_name(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return "".join(ch if ch in allowed else "_" for ch in value)


class ContinuousTargetRacingEnv(RacingGameEnviroment):
    """Racing env variant that keeps analog gas/brake for SB3 continuous actions."""

    def perform_action(self, action_input):
        if self.race_terminated != 0:
            self.controller.reset()
            self.controller.update()
            return

        action_input = np.asarray(action_input, dtype=np.float32)
        if action_input.shape != (3,) or not np.all(np.isfinite(action_input)):
            return

        if self.action_mode != "target":
            return super().perform_action(action_input)

        action = np.array(
            [
                float(np.clip(action_input[0], 0.0, 1.0)),
                float(np.clip(action_input[1], 0.0, 1.0)),
                float(np.clip(action_input[2], -1.0, 1.0)),
            ],
            dtype=np.float32,
        )
        self.previous_action = action
        self.controller.reset()
        self.controller.right_trigger_float(action[0])
        self.controller.left_trigger_float(action[1])
        self.controller.left_joystick_float(action[2], 0.0)
        self.controller.update()


class TrackmaniaSB3Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        map_name: str,
        reward_mode: str,
        action_layout: str,
        env_max_time: float,
        vertical_mode: bool = False,
        max_touches: int = 1,
        start_idle_max_time: float = 2.0,
        terminal_fitness_scale: float = TERMINAL_FITNESS_SCALE,
        progress_reward_interval_steps: int = PROGRESS_REWARD_INTERVAL_STEPS,
        progress_delta_scale: float = PROGRESS_DELTA_SCALE,
    ) -> None:
        super().__init__()
        self.reward_mode = str(reward_mode).strip().lower()
        if self.reward_mode not in {
            "terminal_progress",
            "terminal_fitness",
            "progress_delta",
            "hybrid_progress_terminal_fitness",
        }:
            raise ValueError(
                "reward_mode must be terminal_progress, terminal_fitness, progress_delta, "
                "or hybrid_progress_terminal_fitness."
            )
        self.action_layout = str(action_layout).strip().lower()
        if self.action_layout not in {"gas_steer", "gas_brake_steer", "throttle_steer", "target_3d"}:
            raise ValueError(
                "action_layout must be gas_steer, gas_brake_steer, throttle_steer, or target_3d."
            )

        self.env = ContinuousTargetRacingEnv(
            map_name=map_name,
            never_quit=False,
            action_mode="target",
            dt_ref=1.0 / 100.0,
            dt_ratio_clip=3.0,
            vertical_mode=vertical_mode,
            surface_step_size=Car.SURFACE_STEP_SIZE,
            surface_probe_height=Car.SURFACE_PROBE_HEIGHT,
            surface_ray_lift=Car.SURFACE_RAY_LIFT,
            max_time=env_max_time,
            max_touches=max_touches,
            start_idle_max_time=start_idle_max_time,
        )
        self.observation_space = self.env.observation_space
        if self.action_layout in {"gas_steer", "throttle_steer"}:
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        self.terminal_fitness_scale = float(terminal_fitness_scale)
        self.progress_reward_interval_steps = max(1, int(progress_reward_interval_steps))
        self.progress_delta_scale = float(progress_delta_scale)
        self.prev_progress = 0.0
        self.last_reward_progress = 0.0
        self.steps_since_progress_reward = 0
        self.last_info: Dict[str, Any] = {}
        self.episode_index = 0
        self._defer_next_reset = False
        self._pending_live_reset = False
        self._reset_placeholder_obs: Optional[np.ndarray] = None
        self._reset_placeholder_info: Dict[str, Any] = {}

    def _to_env_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.action_layout == "gas_steer":
            gas = 0.5 * (float(np.clip(action[0], -1.0, 1.0)) + 1.0)
            steer = float(np.clip(action[1], -1.0, 1.0))
            return np.array([gas, 0.0, steer], dtype=np.float32)
        if self.action_layout == "gas_brake_steer":
            return np.array(
                [
                    float(np.clip(action[0], 0.0, 1.0)),
                    float(np.clip(action[1], 0.0, 1.0)),
                    float(np.clip(action[2], -1.0, 1.0)),
                ],
                dtype=np.float32,
            )
        if self.action_layout == "throttle_steer":
            throttle = float(np.clip(action[0], -1.0, 1.0))
            steer = float(np.clip(action[1], -1.0, 1.0))
            gas = max(throttle, 0.0)
            brake = max(-throttle, 0.0)
            return np.array([gas, brake, steer], dtype=np.float32)
        return np.array(
            [
                float(np.clip(action[0], 0.0, 1.0)),
                float(np.clip(action[1], 0.0, 1.0)),
                float(np.clip(action[2], -1.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def _terminal_metrics(self, info: Dict[str, Any]) -> Dict[str, float]:
        progress = float(info.get("total_progress", 0.0))
        time_value = float(info.get("time", 0.0))
        if time_value <= 0.0 or not np.isfinite(time_value):
            time_value = 1e-3
        distance = float(info.get("distance", 0.0))
        term = int(getattr(self.env, "race_terminated", 0))
        if info.get("done", 0.0) == 1.0 and term == 0:
            term = 1
        fitness = Individual.compute_scalar_fitness_for(
            term=term,
            progress=progress,
            time_value=time_value,
            distance=distance,
        )
        return dict(
            term=float(term),
            progress=progress,
            time=time_value,
            distance=distance,
            fitness=float(fitness),
        )

    def _reward(
        self,
        info: Dict[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> tuple[float, Dict[str, float]]:
        metrics = self._terminal_metrics(info)
        progress = float(metrics["progress"])
        progress_delta = max(0.0, progress - self.prev_progress)
        self.prev_progress = max(self.prev_progress, progress)
        self.steps_since_progress_reward += 1

        reward = 0.0
        if self.reward_mode == "progress_delta":
            reward = progress_delta
        elif self.reward_mode == "hybrid_progress_terminal_fitness":
            should_emit_progress_reward = (
                self.steps_since_progress_reward >= self.progress_reward_interval_steps
                or terminated
                or truncated
            )
            if should_emit_progress_reward:
                sparse_progress_delta = max(0.0, progress - self.last_reward_progress)
                reward += sparse_progress_delta * self.progress_delta_scale
                self.last_reward_progress = max(self.last_reward_progress, progress)
                self.steps_since_progress_reward = 0
            if terminated or truncated:
                reward += float(metrics["fitness"]) / self.terminal_fitness_scale
        elif terminated or truncated:
            if self.reward_mode == "terminal_progress":
                reward = progress
            elif self.reward_mode == "terminal_fitness":
                reward = float(metrics["fitness"]) / self.terminal_fitness_scale

        metrics["progress_delta"] = float(progress_delta)
        metrics["last_reward_progress"] = float(self.last_reward_progress)
        metrics["reward"] = float(reward)
        return float(reward), metrics

    def _perform_live_reset(self, seed: Optional[int] = None):
        obs, info = self.env.reset(seed=seed)
        while info.get("done", 0.0) != 0:
            obs, info = self.env.reset(seed=seed)
        self.prev_progress = float(info.get("total_progress", 0.0))
        self.last_reward_progress = self.prev_progress
        self.steps_since_progress_reward = 0
        self.last_info = dict(info)
        self.episode_index += 1
        self._pending_live_reset = False
        self._reset_placeholder_obs = np.asarray(obs, dtype=np.float32).copy()
        self._reset_placeholder_info = dict(info)
        return np.asarray(obs, dtype=np.float32), dict(info)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if self._defer_next_reset and self._reset_placeholder_obs is not None:
            self._defer_next_reset = False
            self._pending_live_reset = True
            info = dict(self._reset_placeholder_info)
            info["deferred_live_reset"] = True
            return self._reset_placeholder_obs.copy(), info
        return self._perform_live_reset(seed=seed)

    def step(self, action):
        if self._pending_live_reset:
            self._perform_live_reset()
        env_action = self._to_env_action(action)
        obs, _, done, truncated, info = self.env.step(env_action)
        race_term = int(getattr(self.env, "race_terminated", 0))
        info_done = info.get("done", 0.0) == 1.0
        terminated = bool(done or info_done or race_term != 0)
        truncated = bool(truncated)
        reward, metrics = self._reward(info, terminated=terminated, truncated=truncated)
        info = dict(info)
        info["sb3_action"] = np.asarray(action, dtype=np.float32).tolist()
        info["env_action"] = env_action.tolist()
        info["episode_metrics"] = metrics
        self.last_info = info
        if terminated or truncated:
            self._defer_next_reset = True
        return np.asarray(obs, dtype=np.float32), reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()
        super().close()


class EpisodeMetricsCallback:
    def __init__(self, run_dir: Path, checkpoint_every_episodes: int = 50, verbose: int = 1) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer: "EpisodeMetricsCallback") -> None:
                super().__init__(verbose=outer.verbose)
                self.outer = outer

            def _on_training_start(self) -> None:
                self.outer.model = self.model

            def _on_step(self) -> bool:
                infos = self.locals.get("infos", [])
                dones = self.locals.get("dones", [])
                for done, info in zip(dones, infos):
                    if bool(done):
                        self.outer.on_episode_end(info)
                return True

        self.run_dir = run_dir
        self.checkpoints_dir = run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = run_dir / "episode_metrics.csv"
        self.checkpoint_every_episodes = max(1, int(checkpoint_every_episodes))
        self.verbose = int(verbose)
        self.episode = 0
        self.best_progress = -float("inf")
        self.best_reward = -float("inf")
        self.model = None
        self.callback = _Callback(self)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers())
            writer.writeheader()

    @staticmethod
    def headers() -> list[str]:
        return [
            "timestamp_utc",
            "episode",
            "term",
            "progress",
            "time",
            "distance",
            "fitness",
            "progress_delta",
            "last_reward_progress",
            "episode_reward",
            "timesteps",
        ]

    def on_episode_end(self, info: Dict[str, Any]) -> None:
        self.episode += 1
        metrics = dict(info.get("episode_metrics", {}))
        monitor_episode = info.get("episode", {})
        episode_reward = float(monitor_episode.get("r", metrics.get("reward", 0.0)))
        progress = float(metrics.get("progress", 0.0))
        row = dict(
            timestamp_utc=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            episode=self.episode,
            term=int(metrics.get("term", 0)),
            progress=progress,
            time=float(metrics.get("time", 0.0)),
            distance=float(metrics.get("distance", 0.0)),
            fitness=float(metrics.get("fitness", 0.0)),
            progress_delta=float(metrics.get("progress_delta", 0.0)),
            last_reward_progress=float(metrics.get("last_reward_progress", 0.0)),
            episode_reward=episode_reward,
            timesteps=int(getattr(self.model, "num_timesteps", 0)) if self.model is not None else 0,
        )
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers())
            writer.writerow(row)

        if self.model is not None:
            self.model.save(self.run_dir / "latest_model")
            if progress > self.best_progress or (
                np.isclose(progress, self.best_progress) and episode_reward > self.best_reward
            ):
                self.best_progress = progress
                self.best_reward = episode_reward
                self.model.save(self.run_dir / "best_model")
            if self.episode % self.checkpoint_every_episodes == 0:
                self.model.save(self.checkpoints_dir / f"sac_episode_{self.episode:05d}")

        if self.verbose:
            print(
                f"[SB3 SAC] ep={self.episode} progress={progress:.2f}% "
                f"reward={episode_reward:.3f} term={int(row['term'])} "
                f"time={row['time']:.2f}s"
            )


class StopTrainingOnWallClock:
    def __init__(self, max_runtime_hours: Optional[float], verbose: int = 1) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer: "StopTrainingOnWallClock") -> None:
                super().__init__(verbose=outer.verbose)
                self.outer = outer

            def _on_training_start(self) -> None:
                self.outer.started_at = time.monotonic()

            def _on_step(self) -> bool:
                if self.outer.max_runtime_seconds is None:
                    return True
                elapsed = time.monotonic() - self.outer.started_at
                if elapsed < self.outer.max_runtime_seconds:
                    return True
                if self.verbose:
                    print(
                        "[SB3 SAC] Max runtime reached: "
                        f"{elapsed / 3600.0:.2f}h >= {self.outer.max_runtime_hours:.2f}h"
                    )
                return False

        self.max_runtime_hours = None if max_runtime_hours is None else float(max_runtime_hours)
        self.max_runtime_seconds = (
            None
            if self.max_runtime_hours is None or self.max_runtime_hours <= 0.0
            else self.max_runtime_hours * 3600.0
        )
        self.verbose = int(verbose)
        self.started_at = time.monotonic()
        self.callback = _Callback(self)


def activation_fn_from_name(name: str):
    value = str(name).strip().lower()
    if value == "relu":
        return nn.ReLU
    if value == "tanh":
        return nn.Tanh
    if value == "elu":
        return nn.ELU
    if value == "leaky_relu":
        return nn.LeakyReLU
    raise ValueError("activation name must be relu, tanh, elu, or leaky_relu.")


def build_run_dir(base_dir: str, map_name: str, reward_mode: str, action_layout: str) -> Path:
    run_name = (
        f"{timestamp()}_sac_map_{map_name}_{reward_mode}_{action_layout}"
    )
    run_dir = Path(base_dir) / sanitize_name(run_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable-Baselines3 SAC Trackmania experiment.")
    parser.add_argument("--episodes", type=int, default=EPISODES_TO_RUN)
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--max-runtime-hours", type=float, default=MAX_RUNTIME_HOURS)
    parser.add_argument(
        "--reward-mode",
        default=REWARD_MODE,
        choices=[
            "terminal_progress",
            "terminal_fitness",
            "progress_delta",
            "hybrid_progress_terminal_fitness",
        ],
    )
    parser.add_argument(
        "--action-layout",
        default=ACTION_LAYOUT,
        choices=["gas_steer", "gas_brake_steer", "throttle_steer", "target_3d"],
    )
    parser.add_argument("--net-arch", default=",".join(str(dim) for dim in SAC_POLICY_NET_ARCH))
    parser.add_argument("--activation-fn", default=SAC_ACTIVATION_FN, choices=["relu", "tanh", "elu", "leaky_relu"])
    parser.add_argument("--env-max-time", type=float, default=ENV_MAX_TIME)
    parser.add_argument("--checkpoint-every-episodes", type=int, default=CHECKPOINT_EVERY_EPISODES)
    parser.add_argument("--progress-reward-interval-steps", type=int, default=PROGRESS_REWARD_INTERVAL_STEPS)
    parser.add_argument("--progress-delta-scale", type=float, default=PROGRESS_DELTA_SCALE)
    parser.add_argument("--terminal-fitness-scale", type=float, default=TERMINAL_FITNESS_SCALE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--import-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CallbackList, StopTrainingOnMaxEpisodes
    from stable_baselines3.common.monitor import Monitor

    if args.import_check:
        print("Stable-Baselines3 import check OK.")
        return

    run_dir = Path(args.run_dir) if args.run_dir else build_run_dir(
        "logs/sb3_runs",
        map_name=MAP_NAME,
        reward_mode=args.reward_mode,
        action_layout=args.action_layout,
    )
    monitor_path = str(run_dir / "monitor.csv")
    raw_env = TrackmaniaSB3Env(
        map_name=MAP_NAME,
        reward_mode=args.reward_mode,
        action_layout=args.action_layout,
        env_max_time=args.env_max_time,
        vertical_mode=VERTICAL_MODE,
        max_touches=MAX_TOUCHES,
        start_idle_max_time=START_IDLE_MAX_TIME,
        terminal_fitness_scale=args.terminal_fitness_scale,
        progress_reward_interval_steps=args.progress_reward_interval_steps,
        progress_delta_scale=args.progress_delta_scale,
    )
    env = Monitor(raw_env, filename=monitor_path, info_keywords=("episode_metrics",))

    config = dict(
        algorithm="SAC",
        map_name=MAP_NAME,
        episodes_to_run=int(args.episodes),
        total_timesteps=int(args.total_timesteps),
        max_runtime_hours=float(args.max_runtime_hours),
        env_max_time=float(args.env_max_time),
        reward_mode=args.reward_mode,
        action_layout=args.action_layout,
        vertical_mode=VERTICAL_MODE,
        progress_reward_interval_steps=int(args.progress_reward_interval_steps),
        progress_delta_scale=float(args.progress_delta_scale),
        terminal_fitness_scale=float(args.terminal_fitness_scale),
        train_freq=list(SAC_TRAIN_FREQ),
        gradient_steps=SAC_GRADIENT_STEPS,
        learning_rate=SAC_LEARNING_RATE,
        buffer_size=SAC_BUFFER_SIZE,
        learning_starts=SAC_LEARNING_STARTS,
        batch_size=SAC_BATCH_SIZE,
        gamma=SAC_GAMMA,
        tau=SAC_TAU,
        ent_coef=SAC_ENT_COEF,
        policy_net_arch=[int(dim) for dim in str(args.net_arch).split(",") if str(dim).strip()],
        activation_fn=args.activation_fn,
        action_space=str(env.action_space),
        observation_space=str(env.observation_space),
        device=args.device,
    )
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=True)

    print("\n[SB3 SAC] Prepared Trackmania SAC experiment")
    print(f"[SB3 SAC] run_dir={run_dir}")
    print(f"[SB3 SAC] reward_mode={args.reward_mode} action_layout={args.action_layout}")
    print(
        f"[SB3 SAC] max_runtime_hours={float(args.max_runtime_hours):.2f} "
        f"episodes_cap={int(args.episodes)} total_timesteps_cap={int(args.total_timesteps)}"
    )
    print(f"[SB3 SAC] train_freq={SAC_TRAIN_FREQ}, gradient_steps={SAC_GRADIENT_STEPS}")
    print(f"[SB3 SAC] action_space={env.action_space}")
    print(f"[SB3 SAC] observation_space={env.observation_space}")

    net_arch = [int(dim) for dim in str(args.net_arch).split(",") if str(dim).strip()]
    policy_kwargs = dict(
        net_arch=net_arch,
        activation_fn=activation_fn_from_name(args.activation_fn),
    )
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=SAC_LEARNING_RATE,
        buffer_size=SAC_BUFFER_SIZE,
        learning_starts=SAC_LEARNING_STARTS,
        batch_size=SAC_BATCH_SIZE,
        tau=SAC_TAU,
        gamma=SAC_GAMMA,
        train_freq=SAC_TRAIN_FREQ,
        gradient_steps=SAC_GRADIENT_STEPS,
        ent_coef=SAC_ENT_COEF,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard"),
        device=args.device,
    )
    episode_callback = EpisodeMetricsCallback(
        run_dir=run_dir,
        checkpoint_every_episodes=args.checkpoint_every_episodes,
        verbose=1,
    )
    wall_clock_callback = StopTrainingOnWallClock(
        max_runtime_hours=args.max_runtime_hours,
        verbose=1,
    )
    callbacks = CallbackList(
        [
            episode_callback.callback,
            StopTrainingOnMaxEpisodes(max_episodes=int(args.episodes), verbose=1),
            wall_clock_callback.callback,
        ]
    )

    try:
        model.learn(
            total_timesteps=int(args.total_timesteps),
            callback=callbacks,
            log_interval=1,
            progress_bar=False,
        )
        model.save(run_dir / "final_model")
        model.save(run_dir / "latest_model")
        print(f"[SB3 SAC] Finished. Artifacts saved in: {run_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

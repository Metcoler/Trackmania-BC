import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.utils

from EvolutionPolicy import EvolutionPolicy, HiddenActivations, HiddenDims
from Individual import Individual
from ObservationEncoder import ObservationEncoder


def choose_device(requested: Optional[str] = None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    return torch.device("cpu")


def sanitize_name(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return "".join(ch if ch in allowed else "_" for ch in value)


def timestamp_utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class SampledAction:
    action: np.ndarray
    log_prob: torch.Tensor
    entropy: torch.Tensor
    gas_prob: float
    brake_prob: float
    steer_mean: float


@dataclass
class EpisodeResult:
    episode: int
    term: int
    progress: float
    time_value: float
    distance: float
    fitness: float
    baseline: float
    return_std: float
    advantage: float
    loss: float
    log_prob_sum: float
    entropy: float
    grad_norm: float
    steps: int
    steer_std: float
    model_path: str


class RunningReturnNormalizer:
    def __init__(
        self,
        baseline_beta: float = 0.9,
        min_std: float = 1.0,
        first_episode_scale: float = 1_000_000.0,
        advantage_clip: Optional[float] = 5.0,
    ) -> None:
        if not 0.0 <= float(baseline_beta) < 1.0:
            raise ValueError("baseline_beta must be in [0, 1).")
        if float(min_std) <= 0.0:
            raise ValueError("min_std must be positive.")
        self.baseline_beta = float(baseline_beta)
        self.min_std = float(min_std)
        self.first_episode_scale = float(max(first_episode_scale, min_std))
        self.advantage_clip = None if advantage_clip is None else float(advantage_clip)
        self.baseline: Optional[float] = None
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def _std(self, value: float, baseline: float) -> float:
        if self.count >= 2:
            variance = self.m2 / float(self.count - 1)
            return float(max(np.sqrt(max(variance, 0.0)), self.min_std))
        return float(max(abs(float(value) - baseline), self.first_episode_scale, self.min_std))

    def normalize_before_update(self, value: float) -> Tuple[float, float, float]:
        value = float(value)
        baseline = 0.0 if self.baseline is None else float(self.baseline)
        std = self._std(value=value, baseline=baseline)
        advantage = (value - baseline) / std
        if self.advantage_clip is not None:
            advantage = float(np.clip(advantage, -self.advantage_clip, self.advantage_clip))
        return baseline, std, float(advantage)

    def update(self, value: float) -> None:
        value = float(value)
        if self.baseline is None:
            self.baseline = value
        else:
            self.baseline = (
                self.baseline_beta * float(self.baseline)
                + (1.0 - self.baseline_beta) * value
            )

        self.count += 1
        delta = value - self.mean
        self.mean += delta / float(self.count)
        delta2 = value - self.mean
        self.m2 += delta * delta2


class RLTrainingLogger:
    EPISODE_HEADERS = [
        "timestamp_utc",
        "episode",
        "term",
        "progress",
        "time",
        "distance",
        "fitness",
        "baseline",
        "return_std",
        "advantage",
        "loss",
        "log_prob_sum",
        "entropy",
        "grad_norm",
        "steps",
        "steer_std",
        "model_path",
    ]

    def __init__(
        self,
        base_dir: str = "logs/rl_runs",
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> None:
        os.makedirs(base_dir, exist_ok=True)
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, sanitize_name(run_name))
        os.makedirs(self.run_dir, exist_ok=False)
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.config_path = os.path.join(self.run_dir, "config.json")
        self.episode_metrics_path = os.path.join(self.run_dir, "episode_metrics.csv")
        self.latest_model_path = os.path.join(self.run_dir, "latest_model.pt")
        self.best_model_path = os.path.join(self.run_dir, "best_model.pt")

        with open(self.episode_metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.EPISODE_HEADERS)
            writer.writeheader()
        if config is not None:
            self.write_config(config)

    def write_config(self, config: Dict) -> None:
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=True)

    def log_episode(self, result: EpisodeResult) -> None:
        row = {
            "timestamp_utc": timestamp_utc(),
            "episode": int(result.episode),
            "term": int(result.term),
            "progress": float(result.progress),
            "time": float(result.time_value),
            "distance": float(result.distance),
            "fitness": float(result.fitness),
            "baseline": float(result.baseline),
            "return_std": float(result.return_std),
            "advantage": float(result.advantage),
            "loss": float(result.loss),
            "log_prob_sum": float(result.log_prob_sum),
            "entropy": float(result.entropy),
            "grad_norm": float(result.grad_norm),
            "steps": int(result.steps),
            "steer_std": float(result.steer_std),
            "model_path": result.model_path,
        }
        with open(self.episode_metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.EPISODE_HEADERS)
            writer.writerow(row)

    def save_model(
        self,
        policy: EvolutionPolicy,
        path: str,
        extra: Optional[Dict] = None,
    ) -> str:
        return policy.save(path, extra=extra)

    def save_checkpoint(
        self,
        policy: EvolutionPolicy,
        episode: int,
        extra: Optional[Dict] = None,
    ) -> str:
        path = os.path.join(self.checkpoints_dir, f"policy_episode_{episode:04d}.pt")
        return self.save_model(policy=policy, path=path, extra=extra)


class ReinforcementTrainer:
    def __init__(
        self,
        env,
        policy: EvolutionPolicy,
        learning_rate: float = 3e-4,
        steer_std: float = 0.35,
        steer_std_decay: float = 1.0,
        steer_std_min: float = 0.08,
        entropy_coef: float = 0.001,
        grad_clip_norm: float = 1.0,
        max_steps: Optional[int] = None,
        baseline_beta: float = 0.9,
        advantage_clip: Optional[float] = 5.0,
        logger: Optional[RLTrainingLogger] = None,
        device: Optional[torch.device] = None,
        debug: bool = False,
        debug_every_steps: int = 100,
        debug_every_seconds: float = 2.0,
    ) -> None:
        if policy.action_mode != "target":
            raise ValueError("ReinforcementTrainer v1 supports only action_mode='target'.")
        if policy.act_dim != 3:
            raise ValueError("ReinforcementTrainer expects act_dim=3.")
        self.env = env
        self.policy = policy
        self.device = choose_device(str(device)) if device is not None else policy.device
        self.policy.to(self.device)
        self.policy.device = self.device
        self.learning_rate = float(learning_rate)
        self.steer_std = float(steer_std)
        self.steer_std_decay = float(steer_std_decay)
        self.steer_std_min = float(steer_std_min)
        self.entropy_coef = float(entropy_coef)
        self.grad_clip_norm = float(grad_clip_norm)
        self.max_steps = None if max_steps is None else int(max_steps)
        self.logger = logger
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.return_normalizer = RunningReturnNormalizer(
            baseline_beta=baseline_beta,
            advantage_clip=advantage_clip,
        )
        self.best_fitness: Optional[float] = None
        self.best_episode: Optional[int] = None
        self.debug = bool(debug)
        self.debug_every_steps = max(1, int(debug_every_steps))
        self.debug_every_seconds = max(0.1, float(debug_every_seconds))

    def _wait_for_positive_game_time(
        self,
        observation: np.ndarray,
        info: Dict,
        timeout_seconds: float = 5.0,
    ) -> Tuple[np.ndarray, Dict]:
        if float(info.get("time", 0.0)) > 0.0:
            return observation, info

        deadline = datetime.now().timestamp() + float(timeout_seconds)
        last_info = dict(info)
        while datetime.now().timestamp() < deadline:
            distances, instructions, info = self.env.observation_info()
            last_info = info
            if float(info.get("time", 0.0)) > 0.0:
                observation = self.env.build_observation(
                    distances=distances,
                    instructions=instructions,
                    info=info,
                )
                self.env.previous_observation_info = (distances, instructions, info)
                self.env.previous_observation = observation
                return observation, info
        return observation, last_info

    def sample_target_action(self, obs: np.ndarray, steer_std: float) -> SampledAction:
        observation = np.asarray(obs, dtype=np.float32).reshape(-1)
        if observation.shape[0] != self.policy.obs_dim:
            raise ValueError(
                f"Obs dim {observation.shape[0]} does not match expected {self.policy.obs_dim}."
            )

        tensor = torch.from_numpy(observation).to(self.device).unsqueeze(0)
        raw = self.policy.raw_forward(tensor).squeeze(0)
        if raw.shape[0] != 3:
            raise ValueError(f"Expected raw action dim 3, got {raw.shape[0]}.")

        gas_dist = torch.distributions.Bernoulli(logits=raw[0])
        brake_dist = torch.distributions.Bernoulli(logits=raw[1])
        steer_mean = torch.tanh(raw[2])
        std = torch.tensor(max(float(steer_std), 1e-6), dtype=torch.float32, device=self.device)
        steer_dist = torch.distributions.Normal(loc=steer_mean, scale=std)

        gas = gas_dist.sample()
        brake = brake_dist.sample()
        steer_unclamped = steer_dist.sample()
        steer = torch.clamp(steer_unclamped, -1.0, 1.0)

        log_prob = (
            gas_dist.log_prob(gas)
            + brake_dist.log_prob(brake)
            + steer_dist.log_prob(steer_unclamped)
        )
        entropy = gas_dist.entropy() + brake_dist.entropy() + steer_dist.entropy()
        action = np.array(
            [
                float(gas.detach().cpu().item()),
                float(brake.detach().cpu().item()),
                float(steer.detach().cpu().item()),
            ],
            dtype=np.float32,
        )
        return SampledAction(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            gas_prob=float(torch.sigmoid(raw[0]).detach().cpu().item()),
            brake_prob=float(torch.sigmoid(raw[1]).detach().cpu().item()),
            steer_mean=float(steer_mean.detach().cpu().item()),
        )

    def _rollout(self, episode: int, steer_std: float) -> Tuple[Dict[str, float], List[torch.Tensor], List[torch.Tensor]]:
        obs, info = self.env.reset()
        while info.get("done", 0.0) != 0:
            obs, info = self.env.reset()

        obs, info = self._wait_for_positive_game_time(obs, info)
        if self.debug:
            print(
                f"\n[RL][episode {episode}] reset confirmed | "
                f"time={float(info.get('time', 0.0)):.3f}s "
                f"progress={float(info.get('total_progress', 0.0)):.2f}% "
                f"obs_dim={np.asarray(obs).reshape(-1).shape[0]} "
                f"steer_std={steer_std:.3f}"
            )
        last_info = info
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        step_count = 0
        next_debug_at = time.monotonic() + self.debug_every_seconds

        self.policy.train()
        while True:
            if self.max_steps is not None and step_count >= self.max_steps:
                break

            sampled = self.sample_target_action(obs=obs, steer_std=steer_std)
            obs, _, done, truncated, info = self.env.step(sampled.action)
            last_info = info
            log_probs.append(sampled.log_prob)
            entropies.append(sampled.entropy)
            step_count += 1

            if self.debug and (
                step_count == 1
                or step_count % self.debug_every_steps == 0
                or time.monotonic() >= next_debug_at
            ):
                print(
                    f"[RL][episode {episode}][step {step_count}] "
                    f"game_time={float(info.get('time', 0.0)):.2f}s "
                    f"progress={float(info.get('total_progress', 0.0)):.2f}% "
                    f"speed={float(info.get('speed', 0.0)):.1f} "
                    f"action=[gas={sampled.action[0]:.0f}, brake={sampled.action[1]:.0f}, steer={sampled.action[2]:+.2f}] "
                    f"policy=[p_gas={sampled.gas_prob:.2f}, p_brake={sampled.brake_prob:.2f}, steer_mu={sampled.steer_mean:+.2f}] "
                    f"term={int(getattr(self.env, 'race_terminated', 0))}"
                )
                next_debug_at = time.monotonic() + self.debug_every_seconds

            race_term = int(getattr(self.env, "race_terminated", 0))
            info_done = info.get("done", 0.0) == 1.0
            if done or truncated or info_done or race_term != 0:
                break

        total_progress = float(last_info.get("total_progress", 0.0))
        time_value = float(last_info.get("time", 0.0))
        if time_value <= 0.0 or not np.isfinite(time_value):
            time_value = 1e-3
        distance = float(last_info.get("distance", 0.0))
        term = int(getattr(self.env, "race_terminated", 0))
        if last_info.get("done", 0.0) == 1.0 and term == 0:
            term = 1
        fitness = Individual.compute_scalar_fitness_for(
            term=term,
            progress=total_progress,
            time_value=time_value,
            distance=distance,
        )
        metrics = dict(
            episode=float(episode),
            term=float(term),
            progress=total_progress,
            time=time_value,
            distance=distance,
            fitness=float(fitness),
            steps=float(step_count),
        )
        return metrics, log_probs, entropies

    def _policy_extra(
        self,
        episode: int,
        metrics: Dict[str, float],
        baseline: float,
        return_std: float,
        advantage: float,
    ) -> Dict:
        vertical_mode = bool(getattr(getattr(self.env, "obs_encoder", None), "vertical_mode", False))
        return dict(
            trainer="reinforce_terminal",
            episode=int(episode),
            term=int(metrics["term"]),
            total_progress=float(metrics["progress"]),
            time=float(metrics["time"]),
            distance=float(metrics["distance"]),
            fitness=float(metrics["fitness"]),
            baseline=float(baseline),
            return_std=float(return_std),
            advantage=float(advantage),
            observation_layout=ObservationEncoder.feature_names(vertical_mode=vertical_mode),
            vertical_mode=vertical_mode,
        )

    def train_episode(self, episode: int, checkpoint_every: int = 1) -> EpisodeResult:
        current_steer_std = max(
            self.steer_std_min,
            self.steer_std * (self.steer_std_decay ** max(0, episode - 1)),
        )
        metrics, log_probs, entropies = self._rollout(
            episode=episode,
            steer_std=current_steer_std,
        )
        fitness = float(metrics["fitness"])
        baseline, return_std, advantage = self.return_normalizer.normalize_before_update(fitness)

        loss_value = 0.0
        log_prob_sum_value = 0.0
        entropy_value = 0.0
        grad_norm_value = 0.0

        if log_probs:
            log_prob_sum = torch.stack(log_probs).sum()
            entropy_sum = torch.stack(entropies).sum() if entropies else torch.tensor(0.0, device=self.device)
            advantage_tensor = torch.tensor(
                float(advantage),
                dtype=torch.float32,
                device=self.device,
            )
            loss = -(advantage_tensor.detach() * log_prob_sum) - (self.entropy_coef * entropy_sum)

            if torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    max_norm=self.grad_clip_norm,
                )
                self.optimizer.step()
                loss_value = float(loss.detach().cpu().item())
                grad_norm_value = float(grad_norm.detach().cpu().item())
            log_prob_sum_value = float(log_prob_sum.detach().cpu().item())
            entropy_value = float(entropy_sum.detach().cpu().item())

        self.return_normalizer.update(fitness)

        model_path = ""
        if self.logger is not None:
            extra = self._policy_extra(
                episode=episode,
                metrics=metrics,
                baseline=baseline,
                return_std=return_std,
                advantage=advantage,
            )
            model_path = self.logger.save_model(
                policy=self.policy,
                path=self.logger.latest_model_path,
                extra=extra,
            )
            if self.best_fitness is None or fitness > float(self.best_fitness):
                self.best_fitness = fitness
                self.best_episode = episode
                self.logger.save_model(
                    policy=self.policy,
                    path=self.logger.best_model_path,
                    extra=extra,
                )
            if checkpoint_every > 0 and episode % int(checkpoint_every) == 0:
                self.logger.save_checkpoint(
                    policy=self.policy,
                    episode=episode,
                    extra=extra,
                )

        result = EpisodeResult(
            episode=episode,
            term=int(metrics["term"]),
            progress=float(metrics["progress"]),
            time_value=float(metrics["time"]),
            distance=float(metrics["distance"]),
            fitness=fitness,
            baseline=baseline,
            return_std=return_std,
            advantage=advantage,
            loss=loss_value,
            log_prob_sum=log_prob_sum_value,
            entropy=entropy_value,
            grad_norm=grad_norm_value,
            steps=int(metrics["steps"]),
            steer_std=float(current_steer_std),
            model_path=model_path,
        )
        if self.logger is not None:
            self.logger.log_episode(result)
        return result

    def run(
        self,
        episodes: int,
        checkpoint_every: int = 1,
        verbose: bool = True,
        max_runtime_minutes: Optional[float] = None,
    ) -> List[EpisodeResult]:
        history: List[EpisodeResult] = []
        started_at = time.monotonic()
        max_runtime_seconds = (
            None
            if max_runtime_minutes is None or float(max_runtime_minutes) <= 0.0
            else max(0.0, float(max_runtime_minutes) * 60.0)
        )
        for episode in range(1, int(episodes) + 1):
            if max_runtime_seconds is not None:
                elapsed = time.monotonic() - started_at
                if elapsed >= max_runtime_seconds:
                    print(
                        f"Stopping RL run before episode {episode}: "
                        f"runtime cap {float(max_runtime_minutes):.2f} minutes reached."
                    )
                    break
            result = self.train_episode(
                episode=episode,
                checkpoint_every=checkpoint_every,
            )
            history.append(result)
            if verbose:
                print(
                    f"{episode}/{episodes} "
                    f"term={result.term} progress={result.progress:.1f}% "
                    f"time={result.time_value:.2f}s score={result.fitness:.2f} "
                    f"adv={result.advantage:.3f} loss={result.loss:.3f} "
                    f"grad_norm={result.grad_norm:.3f} entropy={result.entropy:.2f} "
                    f"steps={result.steps}"
                )
        return history


def validate_loaded_policy(
    policy: EvolutionPolicy,
    obs_dim: int,
    action_mode: str,
) -> None:
    if int(policy.obs_dim) != int(obs_dim):
        raise ValueError(
            f"Initial model obs_dim={policy.obs_dim} does not match env obs_dim={obs_dim}."
        )
    if int(policy.act_dim) != 3:
        raise ValueError(f"Initial model act_dim={policy.act_dim}, expected 3.")
    if str(policy.action_mode) != str(action_mode):
        raise ValueError(
            f"Initial model action_mode={policy.action_mode!r} does not match {action_mode!r}."
        )


def build_policy(
    obs_dim: int,
    hidden_dim: HiddenDims,
    hidden_activation: HiddenActivations,
    action_mode: str,
    device: torch.device,
    initial_model_path: Optional[str] = None,
) -> Tuple[EvolutionPolicy, Dict]:
    if initial_model_path:
        policy, extra = EvolutionPolicy.load(initial_model_path, map_location=device)
        validate_loaded_policy(policy=policy, obs_dim=obs_dim, action_mode=action_mode)
        policy.to(device)
        policy.device = device
        return policy, dict(extra)

    policy = EvolutionPolicy(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        act_dim=3,
        action_scale=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        action_mode=action_mode,
        hidden_activation=hidden_activation,
        device=device,
    )
    return policy, {}


def run_offline_sanity_test() -> None:
    device = choose_device("cpu")
    obs_dim = ObservationEncoder.total_obs_dim(vertical_mode=False)
    policy = EvolutionPolicy(
        obs_dim=obs_dim,
        hidden_dim=[8, 4],
        act_dim=3,
        action_scale=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        action_mode="target",
        hidden_activation=["relu", "tanh"],
        device=device,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    trainer = object.__new__(ReinforcementTrainer)
    trainer.policy = policy
    trainer.device = device

    log_probs: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    for _ in range(8):
        fake_obs = np.random.uniform(-1.0, 1.0, size=(obs_dim,)).astype(np.float32)
        sampled = ReinforcementTrainer.sample_target_action(
            trainer,
            obs=fake_obs,
            steer_std=0.35,
        )
        if sampled.action.shape != (3,):
            raise AssertionError("Sampled action has invalid shape.")
        if not np.all(np.isfinite(sampled.action)):
            raise AssertionError("Sampled action contains non-finite values.")
        log_probs.append(sampled.log_prob)
        entropies.append(sampled.entropy)

    loss = -(torch.stack(log_probs).sum()) - (0.001 * torch.stack(entropies).sum())
    if not torch.isfinite(loss):
        raise AssertionError("Sanity loss is not finite.")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()
    print("Offline RL sanity test passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run terminal-only custom RL training.")
    parser.add_argument("--sanity-test", action="store_true", help="Run offline sampling/backprop sanity test only.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of RL episodes to run.")
    parser.add_argument("--env-max-time", type=float, default=15.0, help="Maximum seconds per episode in Trackmania.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save a checkpoint every N episodes.")
    parser.add_argument("--max-runtime-minutes", type=float, default=10.0, help="Stop before starting a new episode after this many minutes.")
    parser.add_argument("--initial-model-path", type=str, default=None, help="Optional compatible .pt policy to continue from.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run directory name.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, default cpu.")
    parser.add_argument("--no-debug", action="store_true", help="Disable live rollout debug prints.")
    parser.add_argument("--debug-every-steps", type=int, default=100, help="Print rollout debug at least every N policy steps.")
    parser.add_argument("--debug-every-seconds", type=float, default=2.0, help="Print rollout debug at least every N real seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sanity_test:
        run_offline_sanity_test()
        return

    from Car import Car
    from Enviroment import RacingGameEnviroment

    map_name = "AI Training #5"
    hidden_dim: Sequence[int] = [32, 16]
    hidden_activation: Sequence[str] = ["relu", "tanh"]
    action_mode = "target"
    vertical_mode = False
    max_steps = None
    device = choose_device(args.device)

    env = RacingGameEnviroment(
        map_name=map_name,
        never_quit=False,
        action_mode=action_mode,
        dt_ref=1.0 / 100.0,
        dt_ratio_clip=3.0,
        vertical_mode=vertical_mode,
        surface_step_size=Car.SURFACE_STEP_SIZE,
        surface_probe_height=Car.SURFACE_PROBE_HEIGHT,
        surface_ray_lift=Car.SURFACE_RAY_LIFT,
        max_time=args.env_max_time,
        max_touches=1,
        start_idle_max_time=2.0,
    )

    try:
        obs, _ = env.reset()
        obs_dim = int(obs.shape[0])
        policy, initial_model_extra = build_policy(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            hidden_activation=hidden_activation,
            action_mode=action_mode,
            device=device,
            initial_model_path=args.initial_model_path,
        )

        run_name = args.run_name
        if run_name is None:
            run_name = (
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                f"_map_{map_name}_{'v3d' if vertical_mode else 'v2d'}"
                f"_h{'x'.join(str(dim) for dim in policy.hidden_dims)}_rl"
            )
        config = dict(
            trainer="reinforce_terminal",
            map_name=map_name,
            episodes_to_run=int(args.episodes),
            env_max_time=float(args.env_max_time),
            max_steps=max_steps,
            max_runtime_minutes=float(args.max_runtime_minutes),
            checkpoint_every=int(args.checkpoint_every),
            obs_dim=obs_dim,
            action_mode=action_mode,
            vertical_mode=vertical_mode,
            hidden_dim=list(policy.hidden_dims),
            hidden_activation=list(policy.hidden_activations),
            learning_rate=3e-4,
            steer_std=0.35,
            steer_std_decay=1.0,
            steer_std_min=0.08,
            entropy_coef=0.001,
            grad_clip_norm=1.0,
            baseline_beta=0.9,
            advantage_clip=5.0,
            initial_model_path=args.initial_model_path,
            initial_model_extra=initial_model_extra,
            device=str(device),
            debug=not bool(args.no_debug),
            debug_every_steps=int(args.debug_every_steps),
            debug_every_seconds=float(args.debug_every_seconds),
        )
        logger = RLTrainingLogger(
            base_dir="logs/rl_runs",
            run_name=run_name,
            config=config,
        )
        print("\n[RL] Prepared terminal-only REINFORCE trainer")
        print(f"[RL] run_dir={logger.run_dir}")
        print(
            f"[RL] map={map_name!r} obs_dim={obs_dim} action_mode={action_mode} "
            f"vertical_mode={vertical_mode}"
        )
        print(
            f"[RL] network: {obs_dim} -> "
            f"{' -> '.join(str(dim) for dim in policy.hidden_dims)} -> 3 "
            f"activations={list(policy.hidden_activations)}"
        )
        print(
            f"[RL] episodes={int(args.episodes)} env_max_time={float(args.env_max_time):.1f}s "
            f"runtime_cap={float(args.max_runtime_minutes):.1f}min "
            f"lr={config['learning_rate']} steer_std={config['steer_std']} "
            f"entropy_coef={config['entropy_coef']} grad_clip={config['grad_clip_norm']}"
        )
        trainer = ReinforcementTrainer(
            env=env,
            policy=policy,
            learning_rate=config["learning_rate"],
            steer_std=config["steer_std"],
            steer_std_decay=config["steer_std_decay"],
            steer_std_min=config["steer_std_min"],
            entropy_coef=config["entropy_coef"],
            grad_clip_norm=config["grad_clip_norm"],
            max_steps=max_steps,
            baseline_beta=config["baseline_beta"],
            advantage_clip=config["advantage_clip"],
            logger=logger,
            device=device,
            debug=config["debug"],
            debug_every_steps=config["debug_every_steps"],
            debug_every_seconds=config["debug_every_seconds"],
        )
        history = trainer.run(
            episodes=int(args.episodes),
            checkpoint_every=int(args.checkpoint_every),
            verbose=True,
            max_runtime_minutes=float(args.max_runtime_minutes),
        )
        if history:
            best = max(history, key=lambda item: item.fitness)
            print(
                f"Best RL episode: {best.episode}, "
                f"progress={best.progress:.1f}%, time={best.time_value:.2f}s, "
                f"fitness={best.fitness:.2f}"
            )
        print(f"Run artifacts saved in: {logger.run_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

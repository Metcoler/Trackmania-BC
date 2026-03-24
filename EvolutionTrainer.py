import csv
import glob
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from EvolutionPolicy import EvolutionPolicy
from Individual import Individual
from ObservationEncoder import ObservationEncoder


HiddenDims = Union[int, Sequence[int]]


def normalize_hidden_dims(hidden_dim: HiddenDims) -> Tuple[int, ...]:
    if isinstance(hidden_dim, (tuple, list)):
        dims = tuple(int(dim) for dim in hidden_dim)
    else:
        dims = (int(hidden_dim),)
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError("hidden_dim must contain positive integers.")
    return dims


def hidden_dims_tag(hidden_dims: HiddenDims) -> str:
    dims = normalize_hidden_dims(hidden_dims)
    return "x".join(str(dim) for dim in dims)


class TrainingLogger:
    INDIVIDUAL_HEADERS = [
        "timestamp_utc",
        "generation",
        "individual_index",
        "term",
        "is_finish",
        "total_progress",
        "distance",
        "time",
        "fitness",
    ]

    SUMMARY_HEADERS = [
        "timestamp_utc",
        "generation",
        "dist_avg",
        "dist_best_gen",
        "dist_best_global",
        "time_avg",
        "time_best_gen",
        "time_best_global",
        "finish_rate",
        "crash_rate",
        "timeout_rate",
        "best_term",
        "best_progress",
        "best_distance",
        "best_time",
        "best_fitness",
    ]

    def __init__(
        self,
        base_dir: str = "logs/ga_runs",
        run_name: Optional[str] = None,
        run_dir: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        if run_dir is None:
            os.makedirs(base_dir, exist_ok=True)
            if run_name is None:
                run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(base_dir, self._sanitize_name(run_name))
            os.makedirs(run_dir, exist_ok=False)
        else:
            os.makedirs(run_dir, exist_ok=True)

        self.run_dir = run_dir
        self.config_path = os.path.join(run_dir, "config.json")
        self.individual_metrics_path = os.path.join(run_dir, "individual_metrics.csv")
        self.generation_summary_path = os.path.join(run_dir, "generation_summary.csv")
        self.checkpoints_dir = os.path.join(run_dir, "checkpoints")
        self.best_individual_path = os.path.join(run_dir, "best_individual.npz")
        self.best_individual_model_path = os.path.join(run_dir, "best_individual.pt")
        self.global_best_path = os.path.join(run_dir, "global_best.npz")
        self.global_best_model_path = os.path.join(run_dir, "global_best.pt")
        self.final_population_path = os.path.join(run_dir, "final_population.npz")
        self.final_population_model_path = os.path.join(run_dir, "final_population_best.pt")

        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self._init_csv_if_missing(self.individual_metrics_path, self.INDIVIDUAL_HEADERS)
        self._init_csv_if_missing(self.generation_summary_path, self.SUMMARY_HEADERS)

        if config is not None:
            self.write_config(config, merge=True)

    @staticmethod
    def _sanitize_name(value: str) -> str:
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        return "".join(ch if ch in allowed else "_" for ch in value)

    @staticmethod
    def _timestamp_utc() -> str:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _init_csv_if_missing(path: str, headers: List[str]) -> None:
        if os.path.exists(path):
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    def write_config(self, updates: Dict, merge: bool = True) -> None:
        data: Dict = {}
        if merge and os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        data.update(updates)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=True)

    def log_individual_batch(self, rows: List[Dict]) -> None:
        if not rows:
            return
        with open(self.individual_metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.INDIVIDUAL_HEADERS)
            writer.writerows(rows)

    def log_generation_summary(self, row: Dict) -> None:
        with open(self.generation_summary_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.SUMMARY_HEADERS)
            writer.writerow(row)

    def save_population_checkpoint(
        self,
        population: List[Individual],
        generation: int,
        obs_dim: int,
        hidden_dim: HiddenDims,
        act_dim: int,
        best_individual: Optional[Individual] = None,
    ) -> str:
        checkpoint_path = os.path.join(
            self.checkpoints_dir, f"population_gen_{generation:04d}.npz"
        )

        genomes = np.stack([ind.genome for ind in population]).astype(np.float32)
        progresses = np.array(
            [float(ind.total_progress) for ind in population], dtype=np.float32
        )
        times = np.array([float(ind.time) for ind in population], dtype=np.float32)
        terms = np.array([int(ind.term) for ind in population], dtype=np.int32)
        distances = np.array([float(ind.distance) for ind in population], dtype=np.float32)
        fitnesses = np.array(
            [np.nan if ind.fitness is None else float(ind.fitness) for ind in population],
            dtype=np.float32,
        )

        payload = dict(
            generation=np.array([generation], dtype=np.int32),
            genomes=genomes,
            progresses=progresses,
            times=times,
            terms=terms,
            distances=distances,
            fitnesses=fitnesses,
            obs_dim=np.array([obs_dim], dtype=np.int32),
            act_dim=np.array([act_dim], dtype=np.int32),
            hidden_dims=np.array(normalize_hidden_dims(hidden_dim), dtype=np.int32),
        )
        hidden_dims = normalize_hidden_dims(hidden_dim)
        if len(hidden_dims) == 1:
            payload["hidden_dim"] = np.array([hidden_dims[0]], dtype=np.int32)
        if population:
            payload.update(
                action_scale=population[0].policy.action_scale.detach().cpu().numpy().astype(np.float32),
                action_mode=np.array([population[0].policy.action_mode]),
                hidden_activation=np.array([population[0].policy.hidden_activation]),
            )

        if best_individual is not None:
            payload.update(
                best_genome=best_individual.genome.astype(np.float32),
                best_progress=np.array([float(best_individual.total_progress)], dtype=np.float32),
                best_time=np.array([float(best_individual.time)], dtype=np.float32),
                best_term=np.array([int(best_individual.term)], dtype=np.int32),
                best_distance=np.array([float(best_individual.distance)], dtype=np.float32),
                best_fitness=np.array(
                    [np.nan if best_individual.fitness is None else float(best_individual.fitness)],
                    dtype=np.float32,
                ),
            )

        np.savez(checkpoint_path, **payload)
        return checkpoint_path

    def save_final_population(
        self,
        population: List[Individual],
        generation: int,
        obs_dim: int,
        hidden_dim: HiddenDims,
        act_dim: int,
        best_individual: Optional[Individual] = None,
    ) -> str:
        final_path = self.final_population_path
        genomes = np.stack([ind.genome for ind in population]).astype(np.float32)
        progresses = np.array(
            [float(ind.total_progress) for ind in population], dtype=np.float32
        )
        times = np.array([float(ind.time) for ind in population], dtype=np.float32)
        terms = np.array([int(ind.term) for ind in population], dtype=np.int32)
        distances = np.array([float(ind.distance) for ind in population], dtype=np.float32)
        fitnesses = np.array(
            [np.nan if ind.fitness is None else float(ind.fitness) for ind in population],
            dtype=np.float32,
        )

        payload = dict(
            generation=np.array([generation], dtype=np.int32),
            genomes=genomes,
            progresses=progresses,
            times=times,
            terms=terms,
            distances=distances,
            fitnesses=fitnesses,
            obs_dim=np.array([obs_dim], dtype=np.int32),
            act_dim=np.array([act_dim], dtype=np.int32),
            hidden_dims=np.array(normalize_hidden_dims(hidden_dim), dtype=np.int32),
        )
        hidden_dims = normalize_hidden_dims(hidden_dim)
        if len(hidden_dims) == 1:
            payload["hidden_dim"] = np.array([hidden_dims[0]], dtype=np.int32)
        if population:
            payload.update(
                action_scale=population[0].policy.action_scale.detach().cpu().numpy().astype(np.float32),
                action_mode=np.array([population[0].policy.action_mode]),
                hidden_activation=np.array([population[0].policy.hidden_activation]),
            )
        if best_individual is not None:
            payload.update(best_genome=best_individual.genome.astype(np.float32))
        np.savez(final_path, **payload)
        if best_individual is not None:
            best_individual.policy.save(
                self.final_population_model_path,
                extra=self._policy_extra(best_individual, generation),
            )
        return final_path

    def save_best_individual(self, best: Individual, generation: Optional[int] = None) -> str:
        payload = dict(
            genome=best.genome.astype(np.float32),
            total_progress=float(best.total_progress),
            time=float(best.time),
            term=int(best.term),
            distance=float(best.distance),
            fitness=np.nan if best.fitness is None else float(best.fitness),
        )
        if generation is not None:
            payload["generation"] = np.array([int(generation)], dtype=np.int32)
        payload["saved_utc"] = np.array([self._timestamp_utc()])

        # Primary continuously-updated artifact requested for long-running training.
        np.savez(self.global_best_path, **payload)
        # Backward compatibility for older tooling.
        np.savez(self.best_individual_path, **payload)
        extra = self._policy_extra(best, generation)
        best.policy.save(self.global_best_model_path, extra=extra)
        best.policy.save(self.best_individual_model_path, extra=extra)
        return self.global_best_path

    @staticmethod
    def _policy_extra(best: Individual, generation: Optional[int]) -> Dict:
        extra: Dict = dict(
            total_progress=float(best.total_progress),
            time=float(best.time),
            term=int(best.term),
            distance=float(best.distance),
            fitness=np.nan if best.fitness is None else float(best.fitness),
        )
        if generation is not None:
            extra["generation"] = int(generation)
        return extra


class EvolutionTrainer:
    def __init__(
        self,
        env,
        obs_dim: int,
        hidden_dim: HiddenDims = 16,
        act_dim: int = 3,
        pop_size: int = 16,
        max_steps: Optional[int] = 2000,
        policy_action_scale: Optional[np.ndarray] = None,
        policy_action_mode: str = "delta",
        hidden_activation: str = "tanh",
        target_steer_deadzone: float = 0.0,
        logger: Optional[TrainingLogger] = None,
    ) -> None:
        self.env = env
        self.obs_dim = obs_dim
        self.hidden_dims = normalize_hidden_dims(hidden_dim)
        self.hidden_dim = self.hidden_dims[0] if len(self.hidden_dims) == 1 else self.hidden_dims
        self.act_dim = act_dim
        self.pop_size = pop_size
        self.max_steps = None if max_steps is None else int(max_steps)
        self.policy_action_scale = None if policy_action_scale is None else np.asarray(policy_action_scale, dtype=np.float32)
        self.policy_action_mode = str(policy_action_mode).strip().lower()
        self.hidden_activation = str(hidden_activation).strip().lower()
        self.target_steer_deadzone = float(target_steer_deadzone)
        self.logger = logger

        self.population: List[Individual] = [
            Individual(
                obs_dim,
                self.hidden_dims,
                act_dim,
                action_scale=self.policy_action_scale,
                action_mode=self.policy_action_mode,
                hidden_activation=self.hidden_activation,
            )
            for _ in range(pop_size)
        ]
        self.best_individual: Optional[Individual] = None

        # Počet už vyhodnotených generácií.
        self.generation: int = 0

        # Ak načítame checkpoint vyhodnotenej generácie, prvý krok run() má vytvoriť ďalšiu.
        self._loaded_checkpoint_evaluated: bool = False
        self._pending_initial_downselect: bool = False

    @staticmethod
    def _term_status_text(term: int) -> str:
        if term == 1:
            return "FINISH"
        if term == 0:
            return "TIMEOUT"
        if term < 0:
            return f"CRASHx{abs(int(term))}"
        return str(term)

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

    def evaluate_individual(
        self,
        individual: Individual,
        index: Optional[int] = None,
        total: Optional[int] = None,
        verbose: bool = False,
        mirrored: bool = False,
    ) -> float:
        if verbose and index is not None and total is not None:
            print(f"{index + 1}/{total} Evaluating individual...", end="\r")

        obs, info = self.env.reset()
        while info["done"] != 0:
            obs, info = self.env.reset()
        if self.policy_action_mode == "target":
            obs, info = self._wait_for_positive_game_time(obs, info)

        last_info = info

        step_count = 0
        while True:
            if self.max_steps is not None and step_count >= self.max_steps:
                break
            policy_obs = self._mirror_observation(obs) if mirrored else obs
            action = individual.act(policy_obs)
            if mirrored:
                action = self._mirror_action_delta(action)
            action = self._apply_target_steer_deadzone(action)
            obs, reward, done, truncated, info = self.env.step(action)
            last_info = info
            step_count += 1

            race_term = getattr(self.env, "race_terminated", 0)
            info_done = info.get("done", 0.0) == 1.0

            terminated = done or truncated or info_done or (race_term != 0)
            if terminated:
                break

        total_progress = float(last_info.get("total_progress", 0.0))
        t = float(last_info.get("time", 0.0))
        if t <= 0:
            t = 1e-3

        distance = float(last_info.get("distance", 0.0))
        term = int(getattr(self.env, "race_terminated", 0))
        info_done = last_info.get("done", 0.0) == 1.0
        if info_done and term == 0:
            term = 1

        individual.total_progress = total_progress
        individual.time = t
        individual.term = term
        individual.distance = distance

        scalar = individual.compute_scalar_fitness()
        individual.fitness = scalar

        if verbose and index is not None and total is not None:
            status = self._term_status_text(term)
            mirror_tag = " [MIRROR]" if mirrored else ""
            print(
                f"{index + 1}/{total} "
                f"{status} | progress={total_progress:.1f}% | "
                f"time={t:.2f}s | score={scalar:.2f}{mirror_tag}"
            )

        return scalar

    def _mirror_observation(self, obs: np.ndarray) -> np.ndarray:
        return ObservationEncoder.mirror_observation(obs)

    @staticmethod
    def _mirror_action_delta(action: np.ndarray) -> np.ndarray:
        return ObservationEncoder.mirror_action(action)

    def _apply_target_steer_deadzone(self, action: np.ndarray) -> np.ndarray:
        if self.policy_action_mode != "target" or self.target_steer_deadzone <= 0.0:
            return action
        adjusted = np.asarray(action, dtype=np.float32).copy()
        if adjusted.shape == (3,) and abs(float(adjusted[2])) < self.target_steer_deadzone:
            adjusted[2] = 0.0
        return adjusted

    @staticmethod
    def _sample_mirror_flags(count: int, mirror_episode_prob: float) -> np.ndarray:
        if count <= 0 or mirror_episode_prob <= 0.0:
            return np.zeros(max(count, 0), dtype=bool)
        if mirror_episode_prob >= 1.0:
            return np.ones(count, dtype=bool)
        return (np.random.rand(count) < mirror_episode_prob)

    def evaluate_population(
        self,
        verbose: bool = False,
        mirror_flags: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = len(self.population)
        fitnesses = np.zeros(n, dtype=np.float32)
        if mirror_flags is None:
            mirror_flags = np.zeros(n, dtype=bool)
        elif len(mirror_flags) != n:
            raise ValueError(
                f"mirror_flags length {len(mirror_flags)} does not match population size {n}."
            )
        for i, ind in enumerate(self.population):
            fitnesses[i] = self.evaluate_individual(
                individual=ind,
                index=i,
                total=n,
                verbose=verbose,
                mirrored=bool(mirror_flags[i]),
            )
        return fitnesses

    def next_generation(
        self,
        elite_fraction: float = 0.2,
        mutation_prob: float = 0.1,
        mutation_sigma: float = 0.1,
    ) -> None:
        self.population.sort(reverse=True)

        elite_count = max(1, int(self.pop_size * elite_fraction))
        parent_pool_size = max(2, self.pop_size // 2)
        parents = self.population[:parent_pool_size]

        new_population: List[Individual] = [
            ind.copy() for ind in self.population[:elite_count]
        ]

        parent_indices = np.arange(parent_pool_size)
        while len(new_population) < self.pop_size:
            i1, i2 = np.random.choice(parent_indices, size=2, replace=False)
            p1 = parents[int(i1)]
            p2 = parents[int(i2)]
            child = p1.crossover(p2)
            child.mutate(mutation_prob=mutation_prob, sigma=mutation_sigma)
            new_population.append(child)

        self.population = new_population

    def seed_population_from_model(
        self,
        model_path: str,
        exact_copies: int = 1,
        mutation_probs: Sequence[float] = (0.02, 0.05, 0.08),
        mutation_sigmas: Sequence[float] = (0.01, 0.03, 0.05),
        tier_counts: Optional[Sequence[int]] = None,
    ) -> Dict[str, object]:
        if len(mutation_probs) != len(mutation_sigmas):
            raise ValueError("mutation_probs and mutation_sigmas must have the same length.")

        loaded_policy, extra = EvolutionPolicy.load(model_path, map_location="cpu")
        loaded_hidden_dims = tuple(int(dim) for dim in loaded_policy.hidden_dims)
        if loaded_policy.obs_dim != self.obs_dim:
            raise ValueError(
                f"Model obs_dim={loaded_policy.obs_dim} does not match trainer obs_dim={self.obs_dim}."
            )
        if loaded_policy.act_dim != self.act_dim:
            raise ValueError(
                f"Model act_dim={loaded_policy.act_dim} does not match trainer act_dim={self.act_dim}."
            )
        if loaded_hidden_dims != self.hidden_dims:
            raise ValueError(
                f"Model hidden_dims={loaded_hidden_dims} do not match trainer hidden_dims={self.hidden_dims}."
            )
        if loaded_policy.action_mode != self.policy_action_mode:
            raise ValueError(
                f"Model action_mode='{loaded_policy.action_mode}' does not match "
                f"trainer action_mode='{self.policy_action_mode}'."
            )
        if loaded_policy.hidden_activation != self.hidden_activation:
            raise ValueError(
                f"Model hidden_activation='{loaded_policy.hidden_activation}' does not match "
                f"trainer hidden_activation='{self.hidden_activation}'."
            )

        loaded_scale = loaded_policy.action_scale.detach().cpu().numpy().astype(np.float32)
        if self.policy_action_scale is None:
            self.policy_action_scale = loaded_scale.copy()
        elif not np.allclose(self.policy_action_scale, loaded_scale):
            raise ValueError(
                f"Model action_scale={loaded_scale.tolist()} does not match trainer "
                f"action_scale={self.policy_action_scale.tolist()}."
            )

        base_individual = Individual(
            obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dims,
            act_dim=self.act_dim,
            genome=loaded_policy.genome.copy(),
            action_scale=self.policy_action_scale.copy(),
            action_mode=self.policy_action_mode,
            hidden_activation=self.hidden_activation,
        )

        exact_copies = int(max(0, exact_copies))
        if exact_copies > self.pop_size:
            raise ValueError(
                f"exact_copies={exact_copies} exceeds population size {self.pop_size}."
            )

        remaining = self.pop_size - exact_copies
        num_tiers = len(mutation_probs)
        if tier_counts is None:
            tier_counts = [remaining // num_tiers for _ in range(num_tiers)]
            for i in range(remaining % num_tiers):
                tier_counts[i] += 1
        else:
            tier_counts = [int(count) for count in tier_counts]
            if len(tier_counts) != num_tiers:
                raise ValueError("tier_counts length must match number of mutation tiers.")
            if sum(tier_counts) != remaining:
                raise ValueError(
                    f"tier_counts sum to {sum(tier_counts)}, expected remaining population size {remaining}."
                )

        seeded_population: List[Individual] = [base_individual.copy() for _ in range(exact_copies)]
        tier_summaries = []
        for count, prob, sigma in zip(tier_counts, mutation_probs, mutation_sigmas):
            count = int(count)
            if count <= 0:
                continue
            for _ in range(count):
                child = base_individual.copy()
                child.mutate(mutation_prob=float(prob), sigma=float(sigma))
                seeded_population.append(child)
            tier_summaries.append(
                dict(
                    count=count,
                    mutation_prob=float(prob),
                    mutation_sigma=float(sigma),
                )
            )

        if len(seeded_population) != self.pop_size:
            raise RuntimeError(
                f"Seeded population has size {len(seeded_population)}, expected {self.pop_size}."
            )

        self.population = seeded_population
        self.best_individual = None
        self.generation = 0
        self._loaded_checkpoint_evaluated = False
        self._pending_initial_downselect = False

        return dict(
            source_model=model_path,
            exact_copies=exact_copies,
            tiers=tier_summaries,
            model_extra=extra,
            hidden_dims=list(self.hidden_dims),
            action_mode=self.policy_action_mode,
        )

    def _log_generation(
        self,
        generation: int,
        best_gen: Individual,
        best_so_far: Individual,
        dnf_time_for_plot: float,
        history: Dict[str, List[float]],
    ) -> None:
        if self.logger is None:
            return

        timestamp = TrainingLogger._timestamp_utc()

        individual_rows: List[Dict] = []
        for idx, ind in enumerate(self.population):
            individual_rows.append(
                dict(
                    timestamp_utc=timestamp,
                    generation=generation,
                    individual_index=idx,
                    term=int(ind.term),
                    is_finish=1 if int(ind.term) == 1 else 0,
                    total_progress=float(ind.total_progress),
                    distance=float(ind.distance),
                    time=float(ind.time),
                    fitness=np.nan if ind.fitness is None else float(ind.fitness),
                )
            )
        self.logger.log_individual_batch(individual_rows)

        terms = np.array([int(ind.term) for ind in self.population], dtype=np.int32)
        finish_rate = float((terms == 1).mean())
        crash_rate = float((terms < 0).mean())
        timeout_rate = float((terms == 0).mean())

        summary_row = dict(
            timestamp_utc=timestamp,
            generation=generation,
            dist_avg=history["dist_avg"][-1],
            dist_best_gen=history["dist_best_gen"][-1],
            dist_best_global=history["dist_best_global"][-1],
            time_avg=history["time_avg"][-1],
            time_best_gen=history["time_best_gen"][-1],
            time_best_global=history["time_best_global"][-1],
            finish_rate=finish_rate,
            crash_rate=crash_rate,
            timeout_rate=timeout_rate,
            best_term=int(best_gen.term),
            best_progress=float(best_gen.total_progress),
            best_distance=float(best_gen.distance),
            best_time=float(best_gen.time),
            best_fitness=np.nan if best_gen.fitness is None else float(best_gen.fitness),
        )
        self.logger.log_generation_summary(summary_row)

    def run(
        self,
        generations: int,
        elite_fraction: float = 0.2,
        mutation_prob: float = 0.1,
        mutation_sigma: float = 0.1,
        mutation_prob_decay: float = 1.0,
        mutation_prob_min: float = 0.0,
        mutation_sigma_decay: float = 1.0,
        mutation_sigma_min: float = 0.0,
        mirror_episode_prob: float = 0.0,
        verbose: bool = True,
        dnf_time_for_plot: float = 30.0,
        checkpoint_every: int = 10,
        training_config: Optional[dict] = None,
    ) -> Dict[str, List[float]]:
        history = {
            "dist_avg": [],
            "dist_best_gen": [],
            "dist_best_global": [],
            "time_avg": [],
            "time_best_gen": [],
            "time_best_global": [],
        }

        best_so_far: Optional[Individual] = (
            None if self.best_individual is None else self.best_individual.copy()
        )

        if self.logger is not None:
            cfg = dict(
                obs_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                act_dim=self.act_dim,
                pop_size=self.pop_size,
                max_steps=self.max_steps,
                policy_action_mode=self.policy_action_mode,
                hidden_activation=self.hidden_activation,
                run_started_utc=TrainingLogger._timestamp_utc(),
                start_generation=self.generation,
                generations_requested=generations,
                elite_fraction=elite_fraction,
                mutation_prob=mutation_prob,
                mutation_sigma=mutation_sigma,
                mutation_prob_decay=mutation_prob_decay,
                mutation_prob_min=mutation_prob_min,
                mutation_sigma_decay=mutation_sigma_decay,
                mutation_sigma_min=mutation_sigma_min,
                mirror_episode_prob=mirror_episode_prob,
                checkpoint_every=checkpoint_every,
                dnf_time_for_plot=dnf_time_for_plot,
            )
            if training_config is not None:
                cfg.update(training_config)
            self.logger.write_config(cfg, merge=True)

        current_mutation_prob = float(mutation_prob)
        current_mutation_sigma = float(mutation_sigma)
        mutation_prob_decay = float(mutation_prob_decay)
        mutation_prob_min = float(mutation_prob_min)
        mutation_sigma_decay = float(mutation_sigma_decay)
        mutation_sigma_min = float(mutation_sigma_min)

        if self._loaded_checkpoint_evaluated:
            if verbose:
                print(
                    f"Loaded evaluated generation {self.generation}. "
                    "Creating next generation before continuing..."
                )
            self.next_generation(
                elite_fraction=elite_fraction,
                mutation_prob=current_mutation_prob,
                mutation_sigma=current_mutation_sigma,
            )
            self._loaded_checkpoint_evaluated = False

        if self._pending_initial_downselect:
            loaded_count = len(self.population)
            if loaded_count > self.pop_size:
                if verbose:
                    print("\n" + "=" * 20)
                    print("Initial TM Screening (Generation 0)")
                    print("=" * 20)
                    print(
                        f"Evaluating all loaded candidates: {loaded_count} -> "
                        f"select top {self.pop_size} for TM generation 1"
                    )

                screening_mirror_flags = self._sample_mirror_flags(
                    loaded_count,
                    mirror_episode_prob=mirror_episode_prob,
                )
                if verbose and screening_mirror_flags.any():
                    print(
                        f"Screening mirror flags: "
                        f"{int(screening_mirror_flags.sum())}/{loaded_count}"
                    )
                _ = self.evaluate_population(
                    verbose=verbose,
                    mirror_flags=screening_mirror_flags,
                )
                self.population.sort(reverse=True)
                screened_best = self.population[0].copy()
                best_so_far = screened_best
                self.population = self.population[: self.pop_size]

                if verbose:
                    status = self._term_status_text(int(screened_best.term))
                    print(
                        f"Screening best: {status} | "
                        f"progress={screened_best.total_progress:.1f}% | "
                        f"time={screened_best.time:.2f}s"
                    )
                    print(
                        f"Downselected to {len(self.population)} individuals for TM training."
                    )
            self._pending_initial_downselect = False

        # Ensure global best artifact exists from the start (useful when resuming).
        if self.logger is not None and best_so_far is not None:
            self.logger.save_best_individual(best_so_far, generation=self.generation)

        for local_gen in range(generations):
            current_generation = self.generation + 1
            if verbose:
                print("\n" + "=" * 20)
                print(
                    f"Generation {current_generation} | "
                    f"mut_p={current_mutation_prob:.4f}, sigma={current_mutation_sigma:.4f}"
                )
                print("=" * 20)

            gen_mirror_flags = self._sample_mirror_flags(
                len(self.population),
                mirror_episode_prob=mirror_episode_prob,
            )
            if verbose and gen_mirror_flags.any():
                print(
                    f"Mirror flags this generation: "
                    f"{int(gen_mirror_flags.sum())}/{len(self.population)}"
                )

            _ = self.evaluate_population(
                verbose=verbose,
                mirror_flags=gen_mirror_flags,
            )

            progresses = np.array(
                [float(ind.total_progress) for ind in self.population],
                dtype=np.float32,
            )
            times_plot = np.array(
                [
                    float(ind.time) if ind.term == 1 else float(dnf_time_for_plot)
                    for ind in self.population
                ],
                dtype=np.float32,
            )

            dist_avg = float(progresses.mean())
            time_avg = float(times_plot.mean())

            best_gen = max(self.population)
            dist_best_gen = float(best_gen.total_progress)
            time_best_gen = float(
                best_gen.time if best_gen.term == 1 else dnf_time_for_plot
            )

            best_improved = best_so_far is None or best_gen > best_so_far
            if best_improved:
                best_so_far = best_gen.copy()

            dist_best_global = float(best_so_far.total_progress)
            time_best_global = float(
                best_so_far.time if best_so_far.term == 1 else dnf_time_for_plot
            )

            history["dist_avg"].append(dist_avg)
            history["dist_best_gen"].append(dist_best_gen)
            history["dist_best_global"].append(dist_best_global)
            history["time_avg"].append(time_avg)
            history["time_best_gen"].append(time_best_gen)
            history["time_best_global"].append(time_best_global)

            if verbose:
                from_status = self._term_status_text(int(best_gen.term))
                print(
                    f"Best of generation: {from_status} | "
                    f"progress={dist_best_gen:.1f}% | time={best_gen.time:.2f}s"
                )

            self.generation = current_generation
            self._log_generation(
                generation=current_generation,
                best_gen=best_gen,
                best_so_far=best_so_far,
                dnf_time_for_plot=dnf_time_for_plot,
                history=history,
            )

            if self.logger is not None and best_improved and best_so_far is not None:
                global_best_path = self.logger.save_best_individual(
                    best_so_far,
                    generation=current_generation,
                )
                if verbose:
                    print(f"Global best updated: {global_best_path}")

            should_checkpoint = (
                self.logger is not None
                and checkpoint_every > 0
                and (
                    current_generation % checkpoint_every == 0
                    or local_gen == generations - 1
                )
            )
            if should_checkpoint:
                checkpoint_path = self.logger.save_population_checkpoint(
                    population=self.population,
                    generation=current_generation,
                    obs_dim=self.obs_dim,
                    hidden_dim=self.hidden_dim,
                    act_dim=self.act_dim,
                    best_individual=best_so_far,
                )
                if verbose:
                    print(f"Checkpoint saved: {checkpoint_path}")

            if local_gen < generations - 1:
                self.next_generation(
                    elite_fraction=elite_fraction,
                    mutation_prob=current_mutation_prob,
                    mutation_sigma=current_mutation_sigma,
                )
                current_mutation_prob = max(
                    mutation_prob_min,
                    current_mutation_prob * mutation_prob_decay,
                )
                current_mutation_sigma = max(
                    mutation_sigma_min,
                    current_mutation_sigma * mutation_sigma_decay,
                )

        self.best_individual = best_so_far

        if self.logger is not None and self.best_individual is not None:
            self.logger.save_best_individual(self.best_individual, generation=self.generation)
            self.logger.save_final_population(
                population=self.population,
                generation=self.generation,
                obs_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                act_dim=self.act_dim,
                best_individual=self.best_individual,
            )

        return history

    def load_population_checkpoint(
        self, checkpoint_path: str, assume_evaluated_generation: bool = True
    ) -> int:
        data = np.load(checkpoint_path)
        if "genomes" not in data.files:
            raise ValueError(f"Checkpoint {checkpoint_path} does not contain 'genomes'.")

        genomes = data["genomes"].astype(np.float32)
        if genomes.ndim != 2:
            raise ValueError(f"Expected 2D genomes array, got shape {genomes.shape}.")
        loaded_pop_size = int(genomes.shape[0])
        if loaded_pop_size < self.pop_size:
            raise ValueError(
                f"Checkpoint pop_size={loaded_pop_size}, expected at least {self.pop_size}."
            )
        if loaded_pop_size > self.pop_size and assume_evaluated_generation:
            raise ValueError(
                "Checkpoint population is larger than trainer pop_size and is marked as "
                "already evaluated. Use assume_evaluated_generation=False for initial "
                "TM screening + downselect, or match pop_size exactly."
            )

        expected_genome_size = self.population[0].genome.shape[0]
        if genomes.shape[1] != expected_genome_size:
            raise ValueError(
                f"Checkpoint genome_size={genomes.shape[1]}, expected {expected_genome_size}."
            )

        checkpoint_hidden_dims: Optional[Tuple[int, ...]] = None
        if "hidden_dims" in data.files:
            checkpoint_hidden_dims = tuple(
                int(value) for value in np.asarray(data["hidden_dims"]).reshape(-1)
            )
        elif "hidden_dim" in data.files:
            checkpoint_hidden_dims = (
                int(np.asarray(data["hidden_dim"]).reshape(-1)[0]),
            )
        if checkpoint_hidden_dims is not None and checkpoint_hidden_dims != self.hidden_dims:
            raise ValueError(
                f"Checkpoint hidden_dims={checkpoint_hidden_dims} do not match "
                f"trainer hidden_dims={self.hidden_dims}."
            )

        progresses = data["progresses"] if "progresses" in data.files else None
        times = data["times"] if "times" in data.files else None
        terms = data["terms"] if "terms" in data.files else None
        distances = data["distances"] if "distances" in data.files else None
        fitnesses = data["fitnesses"] if "fitnesses" in data.files else None

        restored_population: List[Individual] = []
        for i in range(loaded_pop_size):
            ind = Individual(
                obs_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                act_dim=self.act_dim,
                genome=genomes[i],
                action_scale=self.policy_action_scale,
                action_mode=self.policy_action_mode,
                hidden_activation=self.hidden_activation,
            )
            if progresses is not None:
                ind.total_progress = float(progresses[i])
            if times is not None:
                ind.time = float(times[i])
            if terms is not None:
                ind.term = int(terms[i])
            if distances is not None:
                ind.distance = float(distances[i])
            if fitnesses is not None:
                val = float(fitnesses[i])
                ind.fitness = None if np.isnan(val) else val
            restored_population.append(ind)

        self.population = restored_population

        generation = 0
        if "generation" in data.files:
            generation = int(np.asarray(data["generation"]).reshape(-1)[0])
        self.generation = generation
        self._loaded_checkpoint_evaluated = bool(assume_evaluated_generation)
        self._pending_initial_downselect = (
            (loaded_pop_size > self.pop_size) and (not assume_evaluated_generation)
        )

        if "best_genome" in data.files:
            best = Individual(
                obs_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                act_dim=self.act_dim,
                genome=np.asarray(data["best_genome"], dtype=np.float32),
                action_scale=self.policy_action_scale,
                action_mode=self.policy_action_mode,
                hidden_activation=self.hidden_activation,
            )
            if "best_progress" in data.files:
                best.total_progress = float(np.asarray(data["best_progress"]).reshape(-1)[0])
            if "best_time" in data.files:
                best.time = float(np.asarray(data["best_time"]).reshape(-1)[0])
            if "best_term" in data.files:
                best.term = int(np.asarray(data["best_term"]).reshape(-1)[0])
            if "best_distance" in data.files:
                best.distance = float(np.asarray(data["best_distance"]).reshape(-1)[0])
            if "best_fitness" in data.files:
                bf = float(np.asarray(data["best_fitness"]).reshape(-1)[0])
                best.fitness = None if np.isnan(bf) else bf
            self.best_individual = best
        else:
            self.best_individual = max(self.population).copy() if self.population else None

        return self.generation

    @staticmethod
    def find_latest_checkpoint(base_dir: str = "logs/ga_runs") -> str:
        pattern = os.path.join(base_dir, "**", "checkpoints", "population_gen_*.npz")
        files = glob.glob(pattern, recursive=True)
        if not files:
            raise FileNotFoundError(f"No checkpoints found in {base_dir}.")
        return max(files, key=os.path.getmtime)

    @staticmethod
    def find_latest_supervised_model(base_dir: str = "logs/supervised_runs") -> str:
        pattern = os.path.join(base_dir, "**", "best_model.pt")
        files = glob.glob(pattern, recursive=True)
        if not files:
            raise FileNotFoundError(f"No supervised models found in {base_dir}.")
        return max(files, key=os.path.getmtime)


if __name__ == "__main__":
    from Enviroment import RacingGameEnviroment

    map_name = "small_map"
    hidden_dim = 32
    act_dim = 3
    pop_size = 64
    max_steps = None
    env_max_time = 60
    env_dt_ref = 1.0 / 100.0
    env_dt_ratio_clip = 3.0
    action_mode = "target"
    policy_action_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    hidden_activation = "relu"
    target_steer_deadzone = 0.05
    generations_to_run = 100
    checkpoint_every = 10
    mirror_episode_prob = 0.0
    max_touches = 1
    start_idle_max_time = 3.0
    # Baseline run from scratch: stronger exploration first, then gradual annealing.
    mutation_prob = 0.20
    mutation_prob_decay = 0.997
    mutation_prob_min = 0.03
    mutation_sigma = 0.5
    mutation_sigma_decay = 0.99
    mutation_sigma_min = 0.04
    initial_population_source: Optional[str] = None
    # initial_population_source = r"logs/supervised_runs\20260317_123456_target_supervised\best_model.pt"
    # initial_population_source = (
    #     r"Cars Evolution Training Project\logs\mini_pretrain_runs\20260224_232445"
    #     r"\checkpoints\population_gen_0060.npz"
    # )
    seed_model_exact_copies = 2
    seed_model_mutation_probs = (0.015, 0.04, 0.08)
    seed_model_mutation_sigmas = (0.008, 0.02, 0.04)
    # True = TM checkpoint already evaluated in TM -> continue from next generation.
    # False = mini pretrain checkpoint -> evaluate loaded population in TM first.
    resume_assume_evaluated_generation = False

    resume_checkpoint: Optional[str] = None
    seed_model_path: Optional[str] = None
    initial_population_source_kind = "random"
    if initial_population_source:
        source_ext = os.path.splitext(initial_population_source)[1].lower()
        if source_ext == ".pt":
            seed_model_path = initial_population_source
            initial_population_source_kind = "model_seed"
        elif source_ext == ".npz":
            resume_checkpoint = initial_population_source
            initial_population_source_kind = "population_checkpoint"
        else:
            raise ValueError(
                "initial_population_source must point to a .pt model or .npz population checkpoint."
            )

    if seed_model_path:
        seed_policy, _ = EvolutionPolicy.load(seed_model_path, map_location="cpu")
        hidden_dim = seed_policy.hidden_dims
        act_dim = seed_policy.act_dim
        action_mode = seed_policy.action_mode
        hidden_activation = seed_policy.hidden_activation
        policy_action_scale = (
            seed_policy.action_scale.detach().cpu().numpy().astype(np.float32)
        )

    env = RacingGameEnviroment(
        map_name=map_name,
        never_quit=False,
        action_mode=action_mode,
        dt_ref=env_dt_ref,
        dt_ratio_clip=env_dt_ratio_clip,
        max_time=env_max_time,
        max_touches=max_touches,
        start_idle_max_time=start_idle_max_time,
    )
    obs, info = env.reset()
    obs_dim = obs.shape[0]

    logger: Optional[TrainingLogger]
    if resume_checkpoint:
        if resume_assume_evaluated_generation:
            checkpoint_dir = os.path.dirname(resume_checkpoint)
            run_dir = os.path.dirname(checkpoint_dir)
            logger = TrainingLogger(run_dir=run_dir)
        else:
            source_checkpoint_name = os.path.splitext(os.path.basename(resume_checkpoint))[0]
            source_run_name = os.path.basename(os.path.dirname(os.path.dirname(resume_checkpoint)))
            run_name = (
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                f"_tm_finetune_map_{map_name}_h{hidden_dims_tag(hidden_dim)}_p{pop_size}"
                f"_src_{source_run_name}_{source_checkpoint_name}"
            )
            logger = TrainingLogger(base_dir="logs/tm_finetune_runs", run_name=run_name)
    elif seed_model_path:
        source_model_name = os.path.splitext(os.path.basename(seed_model_path))[0]
        run_name = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            f"_tm_seed_map_{map_name}_h{hidden_dims_tag(hidden_dim)}_p{pop_size}"
            f"_src_{source_model_name}"
        )
        logger = TrainingLogger(base_dir="logs/tm_finetune_runs", run_name=run_name)
    else:
        run_name = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            f"_map_{map_name}_h{hidden_dims_tag(hidden_dim)}_p{pop_size}"
        )
        logger = TrainingLogger(run_name=run_name)

    trainer = EvolutionTrainer(
        env=env,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        act_dim=act_dim,
        pop_size=pop_size,
        max_steps=max_steps,
        policy_action_scale=policy_action_scale,
        policy_action_mode=action_mode,
        hidden_activation=hidden_activation,
        target_steer_deadzone=target_steer_deadzone,
        logger=logger,
    )

    try:
        if resume_checkpoint:
            loaded_generation = trainer.load_population_checkpoint(
                resume_checkpoint,
                assume_evaluated_generation=resume_assume_evaluated_generation,
            )
            print(f"Loaded checkpoint from generation {loaded_generation}: {resume_checkpoint}")
            if not resume_assume_evaluated_generation:
                trainer.generation = 0
                print("Reset TM generation counter to 0 for fine-tuning.")
        elif seed_model_path:
            seed_summary = trainer.seed_population_from_model(
                model_path=seed_model_path,
                exact_copies=seed_model_exact_copies,
                mutation_probs=seed_model_mutation_probs,
                mutation_sigmas=seed_model_mutation_sigmas,
            )
            print(f"Seeded population from model: {seed_model_path}")
            print(f"Seed summary: {seed_summary}")

        history = trainer.run(
            generations=generations_to_run,
            elite_fraction=0.25,
            # Exploratory start + annealing toward fine-tuning.
            mutation_prob=mutation_prob,
            mutation_sigma=mutation_sigma,
            mutation_prob_decay=mutation_prob_decay,
            mutation_prob_min=mutation_prob_min,
            mutation_sigma_decay=mutation_sigma_decay,
            mutation_sigma_min=mutation_sigma_min,
            mirror_episode_prob=mirror_episode_prob,
            verbose=True,
            dnf_time_for_plot=60.0,
            checkpoint_every=checkpoint_every,
            training_config=dict(
                map_name=map_name,
                max_steps=max_steps,
                env_max_time=env_max_time,
                env_dt_ref=env_dt_ref,
                env_dt_ratio_clip=env_dt_ratio_clip,
                action_mode=action_mode,
                policy_action_scale=policy_action_scale.tolist(),
                hidden_activation=hidden_activation,
                target_steer_deadzone=target_steer_deadzone,
                finetune_from_checkpoint=resume_checkpoint,
                initial_population_source=initial_population_source,
                initial_population_source_kind=initial_population_source_kind,
                seed_model_path=seed_model_path,
                seed_model_exact_copies=seed_model_exact_copies,
                seed_model_mutation_probs=list(seed_model_mutation_probs),
                seed_model_mutation_sigmas=list(seed_model_mutation_sigmas),
                mirror_episode_prob=mirror_episode_prob,
                max_touches=max_touches,
                start_idle_max_time=start_idle_max_time,
                mutation_prob_decay=mutation_prob_decay,
                mutation_prob_min=mutation_prob_min,
                mutation_sigma_decay=mutation_sigma_decay,
                mutation_sigma_min=mutation_sigma_min,
            ),
        )

        if trainer.best_individual is not None:
            print(
                f"\nBest overall: term={trainer.best_individual.term}, "
                f"progress={trainer.best_individual.total_progress:.1f}%, "
                f"time={trainer.best_individual.time:.2f}s"
            )
        if logger is not None:
            print(f"Run artifacts saved in: {logger.run_dir}")

    finally:
        env.close()

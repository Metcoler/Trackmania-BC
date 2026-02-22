import csv
import glob
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from Individual import Individual


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
        self.global_best_path = os.path.join(run_dir, "global_best.npz")
        self.final_population_path = os.path.join(run_dir, "final_population.npz")

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
        hidden_dim: int,
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
            hidden_dim=np.array([hidden_dim], dtype=np.int32),
            act_dim=np.array([act_dim], dtype=np.int32),
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
        hidden_dim: int,
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
            hidden_dim=np.array([hidden_dim], dtype=np.int32),
            act_dim=np.array([act_dim], dtype=np.int32),
        )
        if best_individual is not None:
            payload.update(best_genome=best_individual.genome.astype(np.float32))
        np.savez(final_path, **payload)
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
        return self.global_best_path


class EvolutionTrainer:
    def __init__(
        self,
        env,
        obs_dim: int,
        hidden_dim: int = 16,
        act_dim: int = 3,
        pop_size: int = 16,
        max_steps: int = 2000,
        logger: Optional[TrainingLogger] = None,
    ) -> None:
        self.env = env
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.act_dim = act_dim
        self.pop_size = pop_size
        self.max_steps = max_steps
        self.logger = logger

        self.population: List[Individual] = [
            Individual(obs_dim, hidden_dim, act_dim) for _ in range(pop_size)
        ]
        self.best_individual: Optional[Individual] = None

        # Počet už vyhodnotených generácií.
        self.generation: int = 0

        # Ak načítame checkpoint vyhodnotenej generácie, prvý krok run() má vytvoriť ďalšiu.
        self._loaded_checkpoint_evaluated: bool = False

    def evaluate_individual(
        self,
        individual: Individual,
        index: Optional[int] = None,
        total: Optional[int] = None,
        verbose: bool = False,
    ) -> float:
        if verbose and index is not None and total is not None:
            print(f"{index + 1}/{total} Evaluating individual...", end="\r")

        obs, info = self.env.reset()
        while info["done"] != 0:
            obs, info = self.env.reset()

        last_info = info

        for _ in range(self.max_steps):
            action = individual.act(obs)
            obs, reward, done, truncated, info = self.env.step(action)
            last_info = info

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
            status = {1: "FINISH", 0: "TIMEOUT", -1: "CRASH"}.get(term, str(term))
            print(
                f"{index + 1}/{total} "
                f"{status} | progress={total_progress:.1f}% | "
                f"time={t:.2f}s | score={scalar:.2f}"
            )

        return scalar

    def evaluate_population(self, verbose: bool = False) -> np.ndarray:
        fitnesses = np.zeros(self.pop_size, dtype=np.float32)
        for i, ind in enumerate(self.population):
            fitnesses[i] = self.evaluate_individual(
                individual=ind,
                index=i,
                total=self.pop_size,
                verbose=verbose,
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
        crash_rate = float((terms == -1).mean())
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
                run_started_utc=TrainingLogger._timestamp_utc(),
                start_generation=self.generation,
                generations_requested=generations,
                elite_fraction=elite_fraction,
                mutation_prob=mutation_prob,
                mutation_sigma=mutation_sigma,
                checkpoint_every=checkpoint_every,
                dnf_time_for_plot=dnf_time_for_plot,
            )
            if training_config is not None:
                cfg.update(training_config)
            self.logger.write_config(cfg, merge=True)

        if self._loaded_checkpoint_evaluated:
            if verbose:
                print(
                    f"Loaded evaluated generation {self.generation}. "
                    "Creating next generation before continuing..."
                )
            self.next_generation(
                elite_fraction=elite_fraction,
                mutation_prob=mutation_prob,
                mutation_sigma=mutation_sigma,
            )
            self._loaded_checkpoint_evaluated = False

        # Ensure global best artifact exists from the start (useful when resuming).
        if self.logger is not None and best_so_far is not None:
            self.logger.save_best_individual(best_so_far, generation=self.generation)

        for local_gen in range(generations):
            current_generation = self.generation + 1
            if verbose:
                print("\n" + "=" * 20)
                print(f"Generation {current_generation}")
                print("=" * 20)

            _ = self.evaluate_population(verbose=verbose)

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
                from_status = {1: "FINISH", 0: "TIMEOUT", -1: "CRASH"}.get(
                    best_gen.term, str(best_gen.term)
                )
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
                    mutation_prob=mutation_prob,
                    mutation_sigma=mutation_sigma,
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
        if genomes.shape[0] != self.pop_size:
            raise ValueError(
                f"Checkpoint pop_size={genomes.shape[0]}, expected {self.pop_size}."
            )

        expected_genome_size = self.population[0].genome.shape[0]
        if genomes.shape[1] != expected_genome_size:
            raise ValueError(
                f"Checkpoint genome_size={genomes.shape[1]}, expected {expected_genome_size}."
            )

        progresses = data["progresses"] if "progresses" in data.files else None
        times = data["times"] if "times" in data.files else None
        terms = data["terms"] if "terms" in data.files else None
        distances = data["distances"] if "distances" in data.files else None
        fitnesses = data["fitnesses"] if "fitnesses" in data.files else None

        restored_population: List[Individual] = []
        for i in range(self.pop_size):
            ind = Individual(
                obs_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                act_dim=self.act_dim,
                genome=genomes[i],
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

        if "best_genome" in data.files:
            best = Individual(
                obs_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                act_dim=self.act_dim,
                genome=np.asarray(data["best_genome"], dtype=np.float32),
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


if __name__ == "__main__":
    from Enviroment import RacingGameEnviroment

    map_name = "small_map"
    hidden_dim = 32
    act_dim = 3
    pop_size = 16
    max_steps = RacingGameEnviroment.STEPS
    generations_to_run = 50
    checkpoint_every = 10

    # Ak chceš pokračovať z checkpointu, nastav konkrétnu cestu.
    resume_checkpoint: Optional[str] = None
    # resume_checkpoint = EvolutionTrainer.find_latest_checkpoint()
    # True = checkpoint je už "vyhodnotená generácia" (TM -> pokračovanie od ďalšej generácie)
    # False = checkpoint ber ako počiatočnú populáciu (mini pretrain -> najprv vyhodnotiť v TM)
    resume_assume_evaluated_generation = True

    env = RacingGameEnviroment(map_name=map_name, never_quit=False)
    obs, info = env.reset()
    obs_dim = obs.shape[0]

    logger: Optional[TrainingLogger]
    if resume_checkpoint:
        checkpoint_dir = os.path.dirname(resume_checkpoint)
        run_dir = os.path.dirname(checkpoint_dir)
        logger = TrainingLogger(run_dir=run_dir)
    else:
        run_name = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            f"_map_{map_name}_h{hidden_dim}_p{pop_size}"
        )
        logger = TrainingLogger(run_name=run_name)

    trainer = EvolutionTrainer(
        env=env,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        act_dim=act_dim,
        pop_size=pop_size,
        max_steps=max_steps,
        logger=logger,
    )

    try:
        if resume_checkpoint:
            loaded_generation = trainer.load_population_checkpoint(
                resume_checkpoint,
                assume_evaluated_generation=resume_assume_evaluated_generation,
            )
            print(f"Loaded checkpoint from generation {loaded_generation}: {resume_checkpoint}")

        history = trainer.run(
            generations=generations_to_run,
            elite_fraction=0.25,
            mutation_prob=0.1,
            mutation_sigma=0.05,
            verbose=True,
            dnf_time_for_plot=60.0,
            checkpoint_every=checkpoint_every,
            training_config=dict(map_name=map_name),
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

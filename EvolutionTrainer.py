import numpy as np
from typing import Optional, List, Tuple
from Individual import Individual
import matplotlib.pyplot as plt


class EvolutionTrainer:
    def __init__(
        self,
        env,
        obs_dim: int,
        hidden_dim: int = 16,
        act_dim: int = 3,
        pop_size: int = 16,
        max_steps: int = 2000,
    ) -> None:
        self.env = env
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.act_dim = act_dim
        self.pop_size = pop_size
        self.max_steps = max_steps

        # inicializácia populácie
        self.population: List[Individual] = [
            Individual(obs_dim, hidden_dim, act_dim) for _ in range(pop_size)
        ]

        self.best_individual: Optional[Individual] = None
        self.generation: int = 0

    # ------------------------------------------------------------------
    # Vyhodnotenie jedného jedinca
    # ------------------------------------------------------------------
    def evaluate_individual(
        self,
        individual: Individual,
        index: Optional[int] = None,
        total: Optional[int] = None,
        verbose: bool = False,
        ) -> float:
        """
        Spustí jednu epizódu v prostredí pre daného jedinca.

        Na konci:
        - nastaví individual.total_progress (0..100),
        - individual.time (sekundy),
        - individual.term (-1 crash, 0 timeout, 1 finish),
        - spočíta scalar fitness pomocou Individual.compute_scalar_fitness().
        """
        if verbose and index is not None and total is not None:
            print(f"{index + 1}/{total} Evaluating individual...", end="\r")

        
        obs, info = self.env.reset()
        while info["done"] != 0:
            obs, info = self.env.reset()
        last_info = info
        steps_taken = 0

        for step in range(self.max_steps):
            action = individual.act(obs)
            obs, reward, done, truncated, info = self.env.step(action)
            last_info = info
            steps_taken = step + 1

            race_term = getattr(self.env, "race_terminated", 0)
            info_done = info.get("done", 0.0) == 1.0

            # koniec epizódy:
            #  - done z env (limit krokov)
            #  - truncated
            #  - info['done'] z OpenPlanet (dokončenie mapy)
            #  - race_terminated (crash / zlý smer / finish)
            terminated = (
                done
                or truncated
                or info_done
                or (race_term != 0)
            )
            if terminated:
                break

        # --------- multi-kritériá z last_info / env ---------

        total_progress = float(last_info.get("total_progress", 0.0))  # 0..100
        t = float(last_info.get("time", 0.0))
        if t <= 0:
            t = 1e-3  # aby sme sa vyhli deleniu nulou / -inf

        # celková prejdená vzdialenosť z OpenPlanet (Car.get_data -> data["distance"])
        distance = float(last_info.get("distance", 0.0))

        term = int(getattr(self.env, "race_terminated", 0))
        info_done = last_info.get("done", 0.0) == 1.0
        # keby náhodou OpenPlanet nahlásil finish, ale race_terminated zostalo 0
        if info_done and term == 0:
            term = 1

        individual.total_progress = total_progress
        individual.time = t
        individual.term = term
        individual.distance = distance


        # scalar fitness len ako monotónne číslo k ranking_key
        scalar = individual.compute_scalar_fitness()
        individual.fitness = scalar

        if verbose and index is not None and total is not None:
            status = {1: "FINISH", 0: "TIMEOUT", -1: "CRASH"}.get(term, str(term))
            print(
                f"{index + 1}/{total} "
                f"{status} | progress={total_progress:.1f}% | time={t:.2f}s | score={scalar:.2f}"
            )

        return scalar


    # ------------------------------------------------------------------
    # Vyhodnotenie celej populácie
    # ------------------------------------------------------------------
    def evaluate_population(self, verbose: bool = False) -> np.ndarray:
        """
        Vyhodnotí všetkých jedincov v populácii a vráti vektor fitness hodnôt.
        """
        fitnesses = np.zeros(self.pop_size, dtype=np.float32)
        for i, ind in enumerate(self.population):
            fitnesses[i] = self.evaluate_individual(
                individual=ind,
                index=i,
                total=self.pop_size,
                verbose=verbose,
            )
        return fitnesses

    # ------------------------------------------------------------------
    # Vytvorenie novej generácie
    # ------------------------------------------------------------------
    def next_generation(
        self,
        elite_fraction: float = 0.2,
        mutation_prob: float = 0.1,
        mutation_sigma: float = 0.1,
    ) -> None:
        """
        Z aktuálnej populácie vytvorí novú:
          - zachová top elite_fraction ako elitu (bez zmeny)
          - z top polovice populácie náhodne vyberá rodičov, robí crossover + mutate
        """
        # zoradíme populáciu podľa fitness (najhorší -> najlepší),
        # reverse=True => najlepší bude na indexe 0
        self.population.sort(reverse=True)

        elite_count = max(1, int(self.pop_size * elite_fraction))
        parent_pool_size = max(2, self.pop_size // 2)

        parents = self.population[:parent_pool_size]

        # elita prechádza nezmenená
        new_population: List[Individual] = [
            ind.copy() for ind in self.population[:elite_count]
        ]

        # zvyšok doplníme potomkami
        parent_indices = np.arange(parent_pool_size)
        while len(new_population) < self.pop_size:
            i1, i2 = np.random.choice(parent_indices, size=2, replace=False)
            p1 = parents[int(i1)]
            p2 = parents[int(i2)]
            child = p1.crossover(p2)
            child.mutate(mutation_prob=mutation_prob, sigma=mutation_sigma)
            new_population.append(child)

        self.population = new_population
        self.generation += 1

    # ------------------------------------------------------------------
    # Hlavný tréningový loop
    # ------------------------------------------------------------------
    def run(
        self,
        generations: int,
        elite_fraction: float = 0.2,
        mutation_prob: float = 0.1,
        mutation_sigma: float = 0.1,
        verbose: bool = True,
        dnf_time_for_plot: float = 30.0,
    ):
        """
        Spustí evolučný tréning na daný počet generácií.

        Namiesto jednej "magickej" fitness teraz sledujeme:

          - vzdialenosť (total_progress v %)
          - čas (time v sekundách, pri DNF používame dnf_time_for_plot)

        A pre obe veličiny si vedieme 3 krivky:
          - priemer populácie
          - najlepší jedinec v aktuálnej generácii
          - globálne najlepší jedinec (od začiatku tréningu)

        Vracia slovník history, z ktorého vieme urobiť grafy.
        """

        history = {
            "dist_avg": [],
            "dist_best_gen": [],
            "dist_best_global": [],
            "time_avg": [],
            "time_best_gen": [],
            "time_best_global": [],
        }

        best_so_far: Optional[Individual] = None

        for gen in range(generations):
            # 1) pekný header generácie
            if verbose:
                print("\n" + "=" * 20)
                print(f"Generation {gen + 1}/{generations}")
                print("=" * 20)

            # 2) vyhodnotíme populáciu (nastaví term/progress/time + scalar fitness)
            _ = self.evaluate_population(verbose=verbose)

            # 3) štatistiky generácie
            progresses = np.array(
                [float(ind.total_progress) for ind in self.population],
                dtype=np.float32,
            )  # 0..100

            # čas pre graf: finisher = skutočný čas, inak konštanta
            times_plot = np.array(
                [
                    float(ind.time) if ind.term == 1 else float(dnf_time_for_plot)
                    for ind in self.population
                ],
                dtype=np.float32,
            )

            dist_avg = float(progresses.mean())
            time_avg = float(times_plot.mean())

            # najlepší v generácii podľa ranking_key (__lt__)
            best_gen = max(self.population)

            dist_best_gen = float(best_gen.total_progress)
            time_best_gen = float(
                best_gen.time if best_gen.term == 1 else dnf_time_for_plot
            )

            # globálne najlepší (podľa ranking_key)
            if best_so_far is None or best_gen > best_so_far:
                best_so_far = best_gen.copy()

            dist_best_global = float(best_so_far.total_progress)
            time_best_global = float(
                best_so_far.time if best_so_far.term == 1 else dnf_time_for_plot
            )

            # uložíme do histórie
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
                    f"progress={dist_best_gen:.1f}% | "
                    f"time={best_gen.time:.2f}s"
                )

            # 4) ak nie sme v poslednej generácii, vytvoríme novú
            if gen < generations - 1:
                self.next_generation(
                    elite_fraction=elite_fraction,
                    mutation_prob=mutation_prob,
                    mutation_sigma=mutation_sigma,
                )

        # uložíme si aj globálneho najlepšieho jedinca
        self.best_individual = best_so_far

        return history




def plot_training_curves(history, dnf_time_for_plot: float = 60.0, prefix: str = "ga"):
    """
    Vykreslí a uloží dva grafy:

      1) vzdialenosť (progress v %)
      2) čas (finish time, DNF = dnf_time_for_plot)

    V oboch grafoch sú tri krivky:
      - Generation average
      - Generation best
      - Global best individual
    """

    gens = np.arange(1, len(history["dist_avg"]) + 1)

    # --- Graf 1: Distance traveled [%] ---
    plt.figure(figsize=(8, 4))
    plt.plot(gens, history["dist_avg"], label="Generation average", linewidth=2)
    plt.plot(gens, history["dist_best_gen"], label="Generation best", linewidth=2)
    plt.plot(
        gens, history["dist_best_global"], label="Global best individual", linewidth=2
    )
    plt.xlabel("Generation")
    plt.ylabel("Distance traveled [%]")
    plt.ylim(0, 101)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_distance.png", dpi=200)
    plt.close()

    # --- Graf 2: Finish time [s] ---
    plt.figure(figsize=(8, 4))
    plt.plot(gens, history["time_avg"], label="Generation average", linewidth=2)
    plt.plot(gens, history["time_best_gen"], label="Generation best", linewidth=2)
    plt.plot(
        gens, history["time_best_global"], label="Global best individual", linewidth=2
    )
    plt.xlabel("Generation")
    plt.ylabel(f"Finish time [s] (DNF = {dnf_time_for_plot})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_time.png", dpi=200)
    plt.close()




if __name__ == "__main__":
    from Enviroment import RacingGameEnviroment
    import csv
    from datetime import datetime

    map_name = "small_map"  # prípadne "AI Training #3" a pod.
    env = RacingGameEnviroment(map_name=map_name, never_quit=False)

    # zistíme dimenziu observácie
    obs, info = env.reset()
    obs_dim = obs.shape[0]

    hidden_dim = 32
    act_dim = 3
    pop_size = 16
    max_steps = RacingGameEnviroment.STEPS

    trainer = EvolutionTrainer(
        env=env,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        act_dim=act_dim,
        pop_size=pop_size,
        max_steps=max_steps,
    )

    history = None
    # spoločný timestamp pre všetky výstupy z tohto behu
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        generations = 50
        dnf_time_for_plot = 60.0

        # hlavný GA tréning
        history = trainer.run(
            generations=generations,
            elite_fraction=0.25,
            mutation_prob=0.1,
            mutation_sigma=0.05,
            verbose=True,
            dnf_time_for_plot=dnf_time_for_plot,
        )

        # grafy s timestampom v názve
        plot_training_curves(
            history,
            dnf_time_for_plot=dnf_time_for_plot,
            prefix=f"ga_training_{timestamp}",
        )

        if trainer.best_individual is not None:
            print(
                f"\nBest overall: term={trainer.best_individual.term}, "
                f"progress={trainer.best_individual.total_progress:.1f}%, "
                f"time={trainer.best_individual.time:.2f}s"
            )

    finally:
        # --- uloženie history do CSV ---
        if history is not None:
            history_filename = f"ga_history_{timestamp}.csv"
            with open(history_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "generation",
                        "dist_avg",
                        "dist_best_gen",
                        "dist_best_global",
                        "time_avg",
                        "time_best_gen",
                        "time_best_global",
                    ]
                )
                num_gens = len(history["dist_avg"])
                for gen_idx in range(num_gens):
                    writer.writerow(
                        [
                            gen_idx + 1,
                            history["dist_avg"][gen_idx],
                            history["dist_best_gen"][gen_idx],
                            history["dist_best_global"][gen_idx],
                            history["time_avg"][gen_idx],
                            history["time_best_gen"][gen_idx],
                            history["time_best_global"][gen_idx],
                        ]
                    )

            print(f"\nHistory saved to {history_filename}")

        # --- uloženie poslednej generácie (aktuálnej populácie) ---
        if trainer is not None and trainer.population:
            # matica genómov: (pop_size, genome_size)
            genomes = np.stack([ind.genome for ind in trainer.population])

            progresses = np.array(
                [float(ind.total_progress) for ind in trainer.population],
                dtype=np.float32,
            )
            times = np.array(
                [float(ind.time) for ind in trainer.population],
                dtype=np.float32,
            )
            terms = np.array(
                [int(ind.term) for ind in trainer.population],
                dtype=np.int32,
            )
            distances = np.array(
                [float(ind.distance) for ind in trainer.population],
                dtype=np.float32,
            )
            fitnesses = np.array(
                [
                    np.nan if ind.fitness is None else float(ind.fitness)
                    for ind in trainer.population
                ],
                dtype=np.float32,
            )

            pop_filename = f"ga_last_population_{timestamp}.npz"
            np.savez(
                pop_filename,
                genomes=genomes,
                progresses=progresses,
                times=times,
                terms=terms,
                distances=distances,
                fitnesses=fitnesses,
            )

            print(f"Last population saved to {pop_filename}")

        # --- uloženie globálne najlepšieho jedinca ---
        if trainer is not None and trainer.best_individual is not None:
            best = trainer.best_individual
            best_filename = f"ga_best_individual_{timestamp}.npz"
            np.savez(
                best_filename,
                genome=best.genome,
                total_progress=float(best.total_progress),
                time=float(best.time),
                term=int(best.term),
                distance=float(best.distance),
                fitness=(
                    np.nan if best.fitness is None else float(best.fitness)
                ),
            )
            print(f"Best individual saved to {best_filename}")

        env.close()

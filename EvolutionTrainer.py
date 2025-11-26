import numpy as np
from typing import Optional, List, Tuple
from Individual import Individual


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
        early_penalty_per_step: float = -0.5,
    ) -> float:
        """
        Spustí jednu epizódu v prostredí pre daného jedinca a nastaví mu fitness.

        Ak epizóda skončí skôr ako max_steps a NEJDE o úspešné dokončenie mapy
        (info["done"] != 1.0), za každý nevyužitý krok sa pridá penalizácia
        early_penalty_per_step (typicky -2.0).
        """
        if verbose and index is not None and total is not None:
            print(f"{index + 1}/{total} Evaluating individual...", end="\r")

        obs, info = self.env.reset()
        total_reward = 0.0
        steps_taken = 0

        last_info = info  # keby sme náhodou potrebovali na konci

        for step in range(self.max_steps):
            action = individual.act(obs)
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += float(reward)
            steps_taken = step + 1
            last_info = info

            race_term = getattr(self.env, "race_terminated", 0)
            info_done = info.get("done", 0.0) == 1.0

            # koniec epizódy:
            # - done z env (limit krokov)
            # - truncated
            # - info['done'] z OpenPlanet (dokončenie mapy)
            # - race_terminated (náraz / zlyhanie trate / dojazd)
            terminated = (
                done
                or truncated
                or info_done
                or (race_term != 0)
            )
            if terminated:
                break

        # penalizácia za nevyužité kroky (iba ak epizóda skončila predčasne a nešlo o success)
        remaining_steps = self.max_steps - steps_taken
        if remaining_steps > 0 and remaining_steps > 0:
            race_term = getattr(self.env, "race_terminated", 0)
            info_done = last_info.get("done", 0.0) == 1.0

            # epizóda skončila "zle", ak:
            # - nebola dokončená mapa (info_done == False)
            # (race_term < 0 v budúcnosti môžeš použiť na crash / zlý smer)
            bad_end = not info_done

            if bad_end and early_penalty_per_step is not None:
                total_reward += early_penalty_per_step * float(remaining_steps)

        individual.fitness = total_reward

        if verbose and index is not None and total is not None:
            print(f"{index + 1}/{total} Fitness: {individual.fitness:.3f}")

        return total_reward

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
    ) -> List[Tuple[float, float]]:
        """
        Spustí evolučný tréning na daný počet generácií.
        Vracia zoznam (best_fitness, avg_fitness) za každú generáciu.
        """
        history: List[Tuple[float, float]] = []

        best_so_far = float("-inf")
        self.best_individual = None

        for gen in range(generations):
            # 1) pekný header generácie
            if verbose:
                print("\n" + "=" * 20)
                print(f"Generation {gen + 1}/{generations}")
                print("=" * 20)

            # 2) vyhodnotíme aktuálnu populáciu
            fitnesses = self.evaluate_population(verbose=verbose)
            best = float(np.max(fitnesses))
            avg = float(np.mean(fitnesses))
            history.append((best, avg))

            # 3) uložíme si globálne najlepšieho jedinca
            best_idx = int(np.argmax(fitnesses))
            if best > best_so_far:
                best_so_far = best
                self.best_individual = self.population[best_idx].copy()

            if verbose:
                print(
                    f"\nSummary Gen {gen + 1}: "
                    f"best_fitness = {best:.3f}, avg_fitness = {avg:.3f}"
                )

            # 4) ak nie sme v poslednej generácii, vytvoríme novú
            if gen < generations - 1:
                self.next_generation(
                    elite_fraction=elite_fraction,
                    mutation_prob=mutation_prob,
                    mutation_sigma=mutation_sigma,
                )

        return history


if __name__ == "__main__":
    # Jednoduchý tréningový beh GA
    from Enviroment import RacingGameEnviroment
    import csv
    from datetime import datetime

    map_name = "AI Training #3"  # uprav podľa vlastnej mapy
    env = RacingGameEnviroment(map_name=map_name, never_quit=False)

    # zistíme dimenziu observácie
    obs, info = env.reset()
    obs_dim = obs.shape[0]

    hidden_dim = 32
    act_dim = 3
    pop_size = 64

    # použijeme rovnaký horizon, ako má env (RacingGameEnviroment.STEPS ~ 8192)
    max_steps = 4096*2

    trainer = EvolutionTrainer(
        env=env,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        act_dim=act_dim,
        pop_size=pop_size,
        max_steps=max_steps,
    )

    history = None 

    try:
        generations = 100  
        history = trainer.run(
            generations=generations,
            elite_fraction=0.25,
            mutation_prob=0.1,
            mutation_sigma=0.05,
            verbose=True,
        )

        if trainer.best_individual is not None:
            print(f"\nBest overall fitness: {trainer.best_individual.fitness:.3f}")
    finally:
        # spoločný timestamp pre súbory z tohto behu
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # --- uloženie history do CSV ---
        if history is not None:
            history_filename = f"ga_history_{timestamp}.csv"
            with open(history_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["generation", "best_fitness", "avg_fitness"])
                for gen_idx, (best, avg) in enumerate(history, start=1):
                    writer.writerow([gen_idx, best, avg])

            print(f"\nHistory saved to {history_filename}")

        # --- uloženie poslednej generácie jedincov ---
        if trainer is not None and trainer.population:
            # matica genómov: (pop_size, genome_size)
            genomes = np.stack([ind.genome for ind in trainer.population])

            # vektor fitness hodnôt (None -> NaN, keby náhodou niečo nebolo spočítané)
            fitnesses = np.array(
                [
                    np.nan if ind.fitness is None else float(ind.fitness)
                    for ind in trainer.population
                ],
                dtype=np.float32,
            )

            pop_filename = f"ga_last_population_{timestamp}.npz"
            np.savez(pop_filename, genomes=genomes, fitnesses=fitnesses)

            print(f"Last population saved to {pop_filename}")

        env.close()

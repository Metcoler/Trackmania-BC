import os
import glob
import time
from typing import Optional, Tuple

import numpy as np

from Enviroment import RacingGameEnviroment
from Individual import Individual


def find_latest_population(pattern: str = "ga_last_population_*.npz") -> str:
    """Nájde najnovší .npz súbor s populáciou podľa mtime."""
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"Nenašiel som žiadny súbor podľa patternu '{pattern}'. "
            "Spusť najprv EvolutionTrainer, aby vytvoril ga_last_population_*.npz."
        )
    latest = max(files, key=os.path.getmtime)
    return latest


def infer_hidden_dim(genome_size: int, obs_dim: int, act_dim: int) -> int:
    """
    Z dĺžky genómu dopočíta hidden_dim pre sieť:

        genome_size = H*(obs_dim + 1) + act_dim*(H + 1)
                    = H*(obs_dim + 1 + act_dim) + act_dim

    => H = (genome_size - act_dim) / (obs_dim + 1 + act_dim)
    """
    denom = obs_dim + 1 + act_dim
    num = genome_size - act_dim
    if denom <= 0:
        raise ValueError("Zlé rozmery pri infer_hidden_dim.")

    if num % denom != 0:
        raise ValueError(
            f"Genome size {genome_size} nie je kompatibilný s "
            f"obs_dim={obs_dim}, act_dim={act_dim}."
        )

    hidden_dim = num // denom
    if hidden_dim <= 0:
        raise ValueError(
            f"Vyšlo hidden_dim={hidden_dim}, čo nedáva zmysel. "
            "Skontroluj architektúru siete."
        )
    return hidden_dim


def load_population(filename: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Načíta genómy a fitnessy z .npz súboru."""
    data = np.load(filename)
    genomes = data["genomes"]
    fitnesses = data.get("fitnesses", None)
    return genomes, fitnesses


def replay_population(
    map_name: str = "small_map",
    population_file: Optional[str] = None,
    episodes_per_individual: int = 1,
    max_steps: Optional[int] = None,
    pause_between: bool = True,
) -> None:
    """
    Načíta poslednú populáciu a pustí každého jedinca ako "drivera" v hre.
    """

    # 1) nájdeme a načítame populáciu
    if population_file is None:
        population_file = find_latest_population()

    print(f"Načítavam populáciu zo súboru: {population_file}")
    genomes, fitnesses = load_population(population_file)
    pop_size, genome_size = genomes.shape

    # 2) vytvoríme env a zistíme rozmery observation/action
    env = RacingGameEnviroment(map_name=map_name, never_quit=True)
    obs, info = env.reset()
    obs_dim = obs.shape[0]

    try:
        act_dim = env.action_space.shape[0]
    except Exception:
        act_dim = 3  # fallback, ak by action_space nemal shape

    # 3) ak netreba inak, ako max_steps použijeme (rozumný) limit
    if max_steps is None:
        # pre replay nemusíme používať extrémne dlhý horizon
        max_steps = min(RacingGameEnviroment.STEPS, 4000)

    # 4) z dĺžky genómu dopočítame hidden_dim
    hidden_dim = infer_hidden_dim(genome_size, obs_dim, act_dim)
    print(
        f"Populácia: {pop_size} jedincov, genome_size = {genome_size}, "
        f"obs_dim = {obs_dim}, act_dim = {act_dim}, hidden_dim = {hidden_dim}"
    )

    # 5) zoradíme jedincov podľa uloženého fitness (ak je k dispozícii)
    indices = np.arange(pop_size)
    if fitnesses is not None:
        fitnesses_safe = np.array(
            [(-np.inf if np.isnan(f) else float(f)) for f in fitnesses],
            dtype=np.float32,
        )
        indices = np.argsort(-fitnesses_safe)  # od najlepšieho

    # 6) replay každého jedinca
    try:
        for rank, idx in enumerate(indices, start=1):
            genome = genomes[idx]
            saved_fitness = None
            if fitnesses is not None:
                saved_fitness = fitnesses[idx]

            print("\n" + "=" * 30)
            print(f"Jedinec {rank}/{pop_size} (index v populácii: {idx})")
            if saved_fitness is not None:
                print(f"Uložená fitness z tréningu: {saved_fitness:.3f}")
            print("=" * 30)

            individual = Individual(
                obs_dim=obs_dim,
                hidden_dim=hidden_dim,
                act_dim=act_dim,
                genome=genome,
            )

            for ep in range(episodes_per_individual):
                obs, info = env.reset()
                total_reward = 0.0

                for step in range(max_steps):
                    action = individual.act(obs)
                    obs, reward, done, truncated, info = env.step(action)
                    total_reward += float(reward)

                    race_term = getattr(env, "race_terminated", 0)
                    info_done = info.get("done", 0.0) == 1.0

                    terminated = (
                        done
                        or truncated
                        or info_done
                        or (race_term != 0)
                    )
                    if terminated:
                        break

                print(
                    f"  Epizóda {ep + 1}/{episodes_per_individual} "
                    f"- total_reward = {total_reward:.3f}"
                )

            if pause_between and rank < pop_size:
                input("Stlač Enter pre ďalšieho jedinca...")

    finally:
        env.close()
        print("Enviroment zatvorený.")


if __name__ == "__main__":
    MAP_NAME = "small_map"          # názov mapy
    EPISODES_PER_INDIVIDUAL = 1     # koľkokrát pustiť každého jedinca
    MAX_STEPS = 2000                # None -> použije min(STEPS, 4000)
    PAUSE_BETWEEN = False           # pauza (Enter) medzi jedincami

    replay_population(
        map_name=MAP_NAME,
        population_file=None,       # None -> automaticky nájde najnovší .npz
        episodes_per_individual=EPISODES_PER_INDIVIDUAL,
        max_steps=MAX_STEPS,
        pause_between=PAUSE_BETWEEN,
    )

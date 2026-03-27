import glob
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Enviroment import RacingGameEnviroment
from EvolutionPolicy import EvolutionPolicy
from Individual import Individual


def find_latest_population(patterns: Optional[List[str]] = None) -> str:
    """Find newest population .npz checkpoint (mini pretrain or TM GA run)."""
    if patterns is None:
        patterns = [
            "Cars Evolution Training Project/logs/mini_pretrain_runs/**/checkpoints/population_gen_*.npz",
            "Cars Evolution Training Project/logs/mini_pretrain_runs/**/final_population.npz",
            "logs/mini_pretrain_runs/**/checkpoints/population_gen_*.npz",
            "logs/mini_pretrain_runs/**/final_population.npz",
            "logs/ga_runs/**/checkpoints/population_gen_*.npz",
            "logs/ga_runs/**/final_population.npz",
            "logs/ga_last_population_*.npz",  # legacy
        ]

    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))

    if not files:
        raise FileNotFoundError(
            "No population .npz found. Check logs/mini_pretrain_runs or logs/ga_runs."
        )

    return max(files, key=os.path.getmtime)


def find_latest_supervised_model(
    pattern: str = "logs/supervised_runs/**/best_model.pt",
) -> str:
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(
            "No supervised model found. Check logs/supervised_runs for best_model.pt."
        )
    return max(files, key=os.path.getmtime)


def infer_hidden_dim(genome_size: int, obs_dim: int, act_dim: int) -> int:
    """
    Infer hidden_dim from flattened MLP genome:

        genome_size = H*(obs_dim + 1) + act_dim*(H + 1)
                    = H*(obs_dim + 1 + act_dim) + act_dim
    """
    denom = obs_dim + 1 + act_dim
    num = genome_size - act_dim
    if denom <= 0:
        raise ValueError("Invalid dimensions for infer_hidden_dim.")

    if num % denom != 0:
        raise ValueError(
            f"Genome size {genome_size} is not compatible with "
            f"obs_dim={obs_dim}, act_dim={act_dim}."
        )

    hidden_dim = num // denom
    if hidden_dim <= 0:
        raise ValueError(
            f"Inferred hidden_dim={hidden_dim}, which is invalid. Check network architecture."
        )
    return hidden_dim


def _read_optional_array(data, key: str) -> Optional[np.ndarray]:
    return np.asarray(data[key]) if key in data.files else None


def _read_optional_scalar_int(data, key: str) -> Optional[int]:
    if key not in data.files:
        return None
    arr = np.asarray(data[key]).reshape(-1)
    if arr.size == 0:
        return None
    return int(arr[0])


def _read_optional_int_tuple(data, key: str) -> Optional[Tuple[int, ...]]:
    if key not in data.files:
        return None
    arr = np.asarray(data[key]).reshape(-1)
    if arr.size == 0:
        return None
    return tuple(int(value) for value in arr)


def _read_optional_str_tuple(data, key: str) -> Optional[Tuple[str, ...]]:
    if key not in data.files:
        return None
    arr = np.asarray(data[key]).reshape(-1)
    if arr.size == 0:
        return None
    return tuple(str(value) for value in arr)


def load_population(
    filename: str,
) -> Tuple[np.ndarray, Dict[str, Optional[np.ndarray]], Dict[str, Any]]:
    """
    Load genomes + optional metrics/meta from population checkpoint.

    Supported formats:
    - mini tkinter pretrain checkpoints (population_gen_XXXX.npz)
    - Trackmania GA checkpoints/final_population.npz
    - legacy ga_last_population_*.npz (if it contains genomes)
    """
    with np.load(filename) as data:
        if "genomes" not in data.files:
            raise ValueError(f"File '{filename}' does not contain 'genomes'.")

        genomes = np.asarray(data["genomes"], dtype=np.float32)
        if genomes.ndim != 2:
            raise ValueError(f"Expected 2D 'genomes' array, got shape {genomes.shape}.")

        metrics: Dict[str, Optional[np.ndarray]] = {
            "fitnesses": _read_optional_array(data, "fitnesses"),
            "progresses": _read_optional_array(data, "progresses"),
            "times": _read_optional_array(data, "times"),
            "terms": _read_optional_array(data, "terms"),
            "distances": _read_optional_array(data, "distances"),
        }
        meta: Dict[str, Any] = {
            "generation": _read_optional_scalar_int(data, "generation"),
            "obs_dim": _read_optional_scalar_int(data, "obs_dim"),
            "hidden_dim": _read_optional_scalar_int(data, "hidden_dim"),
            "act_dim": _read_optional_scalar_int(data, "act_dim"),
        }
        hidden_dims = _read_optional_int_tuple(data, "hidden_dims")
        if hidden_dims is not None:
            meta["hidden_dims"] = hidden_dims
        hidden_activations = _read_optional_str_tuple(data, "hidden_activations")
        if hidden_activations is None:
            hidden_activations = _read_optional_str_tuple(data, "hidden_activation")
        if hidden_activations is not None:
            meta["hidden_activations"] = hidden_activations

    return genomes, metrics, meta


def _build_replay_indices(
    pop_size: int,
    fitnesses: Optional[np.ndarray],
    sort_by_fitness: bool,
    rank_start: int,
    rank_end: Optional[int],
    exact_indices: Optional[List[int]],
) -> np.ndarray:
    if exact_indices is not None and len(exact_indices) > 0:
        selected: List[int] = []
        for idx in exact_indices:
            i = int(idx)
            if i < 0:
                i = pop_size + i
            if i < 0 or i >= pop_size:
                raise IndexError(f"Index {idx} is out of range 0..{pop_size-1}.")
            selected.append(i)
        return np.asarray(selected, dtype=np.int32)

    indices = np.arange(pop_size, dtype=np.int32)
    if sort_by_fitness and fitnesses is not None:
        fitnesses_safe = np.array(
            [(-np.inf if np.isnan(f) else float(f)) for f in fitnesses],
            dtype=np.float32,
        )
        indices = np.argsort(-fitnesses_safe).astype(np.int32)  # descending

    # 1-based rank slice for convenience when replaying "top N"
    start_zero = max(0, int(rank_start) - 1)
    end_zero = None if rank_end is None else int(rank_end)
    return indices[start_zero:end_zero]


def _wait_for_positive_game_time(
    env: RacingGameEnviroment,
    timeout_seconds: float = 5.0,
):
    observation = getattr(env, "previous_observation", None)
    distances, instructions, info = getattr(env, "previous_observation_info", (None, None, {}))
    if float(info.get("time", 0.0)) > 0.0 and observation is not None:
        return observation, info

    deadline = time.perf_counter() + float(timeout_seconds)
    last_info = dict(info)
    while time.perf_counter() < deadline:
        distances, instructions, info = env.observation_info()
        last_info = info
        if float(info.get("time", 0.0)) > 0.0:
            observation = env.build_observation(
                distances=distances,
                instructions=instructions,
                info=info,
            )
            env.previous_observation_info = (distances, instructions, info)
            env.previous_observation = observation
            return observation, info
    return observation, last_info


def _apply_target_steer_deadzone(action: np.ndarray, steer_deadzone: float) -> np.ndarray:
    if steer_deadzone <= 0.0:
        return action
    adjusted = np.asarray(action, dtype=np.float32).copy()
    if adjusted.shape == (3,) and abs(float(adjusted[2])) < float(steer_deadzone):
        adjusted[2] = 0.0
    return adjusted


def replay_population(
    map_name: str = "small_map",
    population_file: Optional[str] = None,
    episodes_per_individual: int = 1,
    max_steps: Optional[int] = None,
    env_max_time: float = 60.0,
    max_touches: int = 1,
    never_quit: bool = True,
    action_mode: str = "delta",
    pause_between: bool = True,
    sort_by_fitness: bool = True,
    rank_start: int = 1,
    rank_end: Optional[int] = None,
    exact_indices: Optional[List[int]] = None,
    target_steer_deadzone: float = 0.0,
) -> None:
    """
    Replay a whole population or selected subset in Trackmania.

    Selection modes:
    - exact_indices=[...] : replay specific indices from saved population
    - otherwise ranks <rank_start, rank_end> after optional fitness sorting
    """
    if population_file is None:
        population_file = find_latest_population()

    print(f"Loading population from: {population_file}")
    genomes, metrics, meta = load_population(population_file)
    fitnesses = metrics.get("fitnesses")
    pop_size, genome_size = genomes.shape

    print(f"Loaded {pop_size} individuals, genome_size={genome_size}")
    if meta.get("generation") is not None:
        print(f"Checkpoint generation: {meta['generation']}")

    env = RacingGameEnviroment(
        map_name=map_name,
        never_quit=never_quit,
        action_mode=action_mode,
        max_time=env_max_time,
        max_touches=max_touches,
    )
    obs, info = env.reset()
    obs_dim = obs.shape[0]

    try:
        try:
            act_dim = int(env.action_space.shape[0])
        except Exception:
            act_dim = 3

        file_obs_dim = meta.get("obs_dim")
        file_hidden_dims = meta.get("hidden_dims")
        file_hidden_dim = meta.get("hidden_dim")
        file_hidden_activations = meta.get("hidden_activations")
        file_act_dim = meta.get("act_dim")

        if file_obs_dim is not None and file_obs_dim != obs_dim:
            print(
                f"WARNING: checkpoint obs_dim={file_obs_dim} does not match env obs_dim={obs_dim}"
            )
        if file_act_dim is not None and file_act_dim != act_dim:
            print(
                f"WARNING: checkpoint act_dim={file_act_dim} does not match env act_dim={act_dim}"
            )

        if file_hidden_dims is not None:
            hidden_dim = tuple(int(dim) for dim in file_hidden_dims)
        elif file_hidden_dim is not None and file_hidden_dim > 0:
            hidden_dim = int(file_hidden_dim)
        else:
            hidden_dim = infer_hidden_dim(genome_size, obs_dim, act_dim)

        hidden_activation: Any = "tanh"
        if file_hidden_activations is not None:
            file_hidden_activations = tuple(str(value) for value in file_hidden_activations)
            hidden_activation = (
                file_hidden_activations[0]
                if len(file_hidden_activations) == 1
                else list(file_hidden_activations)
            )

        print(
            f"Architecture: obs_dim={obs_dim}, hidden_dim={hidden_dim}, "
            f"hidden_activation={hidden_activation}, act_dim={act_dim}"
        )

        indices = _build_replay_indices(
            pop_size=pop_size,
            fitnesses=fitnesses,
            sort_by_fitness=sort_by_fitness,
            rank_start=rank_start,
            rank_end=rank_end,
            exact_indices=exact_indices,
        )

        if indices.size == 0:
            print("Selection is empty, nothing to replay.")
            return

        if exact_indices:
            print(f"Replaying exact indices: {list(map(int, indices))}")
        else:
            sort_txt = "sorted by saved fitness" if (sort_by_fitness and fitnesses is not None) else "saved order"
            print(
                f"Replaying {indices.size} individuals ({sort_txt}), "
                f"ranks {rank_start}..{rank_end or pop_size}"
            )

        m_progress = metrics.get("progresses")
        m_times = metrics.get("times")
        m_terms = metrics.get("terms")
        m_distances = metrics.get("distances")

        for rank_in_selection, idx in enumerate(indices, start=1):
            idx = int(idx)
            genome = genomes[idx]

            print("\n" + "=" * 40)
            print(f"Replay {rank_in_selection}/{indices.size} | population index={idx}")
            if fitnesses is not None:
                fval = float(fitnesses[idx])
                print(f"Saved fitness: {fval:.3f}")
            if m_progress is not None:
                parts = [f"saved progress={float(m_progress[idx]):.2f}%"]
                if m_times is not None:
                    parts.append(f"time={float(m_times[idx]):.2f}")
                if m_distances is not None:
                    parts.append(f"distance={float(m_distances[idx]):.2f}")
                if m_terms is not None:
                    parts.append(f"term={int(m_terms[idx])}")
                print("Saved metrics: " + ", ".join(parts))
            print("=" * 40)

            individual = Individual(
                obs_dim=obs_dim,
                hidden_dim=hidden_dim,
                act_dim=act_dim,
                genome=genome,
                action_scale=np.ones(act_dim, dtype=np.float32) if str(action_mode).strip().lower() == "target" else None,
                action_mode=action_mode,
                hidden_activation=hidden_activation,
            )

            for ep in range(episodes_per_individual):
                obs, info = env.reset()
                if str(action_mode).strip().lower() == "target":
                    obs, info = _wait_for_positive_game_time(env)
                total_reward = 0.0

                step_count = 0
                while True:
                    if max_steps is not None and step_count >= max_steps:
                        break
                    action = individual.act(obs)
                    if str(action_mode).strip().lower() == "target":
                        action = _apply_target_steer_deadzone(action, target_steer_deadzone)
                    obs, reward, done, truncated, info = env.step(action)
                    total_reward += float(reward)
                    step_count += 1

                    race_term = getattr(env, "race_terminated", 0)
                    info_done = info.get("done", 0.0) == 1.0
                    terminated = done or truncated or info_done or (race_term != 0)
                    if terminated:
                        break

                print(
                    f"  Episode {ep + 1}/{episodes_per_individual} | "
                    f"reward={total_reward:.3f} | "
                    f"term={getattr(env, 'race_terminated', 0)} | "
                    f"progress={float(info.get('total_progress', 0.0)):.2f}% | "
                    f"time={float(info.get('time', 0.0)):.2f}s | "
                    f"distance={float(info.get('distance', 0.0)):.2f}"
                )

            if pause_between and rank_in_selection < indices.size:
                input("Press Enter for next individual...")

    finally:
        env.close()
        print("Environment closed.")


def drive_model(
    map_name: str,
    model_file: str,
    max_steps: Optional[int] = None,
    env_max_time: float = 60.0,
    max_touches: int = 1,
    never_quit: bool = True,
    action_mode: str = "target",
    target_steer_deadzone: float = 0.0,
) -> None:
    policy, extra = EvolutionPolicy.load(model_file, map_location="cpu")
    print(f"Loaded model from: {model_file}")
    print(f"Model config: {policy.get_config()}")
    if extra:
        print(f"Model extra: {extra}")

    env = RacingGameEnviroment(
        map_name=map_name,
        never_quit=never_quit,
        action_mode=action_mode,
        max_time=env_max_time,
        max_touches=max_touches,
    )
    obs, info = env.reset()
    if str(action_mode).strip().lower() == "target":
        obs, info = _wait_for_positive_game_time(env)
    if obs.shape[0] != policy.obs_dim:
        raise ValueError(
            f"Model obs_dim={policy.obs_dim} does not match env obs_dim={obs.shape[0]}."
        )
    try:
        total_reward = 0.0
        step_count = 0
        while True:
            if max_steps is not None and step_count >= max_steps:
                break
            action = policy.act(obs)
            if str(action_mode).strip().lower() == "target":
                action = _apply_target_steer_deadzone(action, target_steer_deadzone)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            step_count += 1
            race_term = getattr(env, "race_terminated", 0)
            info_done = info.get("done", 0.0) == 1.0
            if done or truncated or info_done or (race_term != 0):
                break

        print(
            f"Replay finished | reward={total_reward:.3f} | "
            f"term={getattr(env, 'race_terminated', 0)} | "
            f"progress={float(info.get('total_progress', 0.0)):.2f}% | "
            f"time={float(info.get('time', 0.0)):.2f}s | "
            f"distance={float(info.get('distance', 0.0)):.2f}"
        )
    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    MAP_NAME = "AI Training #2"
    MODEL_FILE = find_latest_supervised_model()
    POPULATION_FILE = None
    # POPULATION_FILE = None  # Auto-pick latest supported population checkpoint.
    # Example TM checkpoint:
    # POPULATION_FILE = (
    #     r"logs/tm_finetune_runs/20260222_213116_tm_finetune_map_small_map_h32_p32"
    #     r"_src_20260221_235425_population_gen_0100/checkpoints/population_gen_0100.npz"
    # )

    EPISODES_PER_INDIVIDUAL = 1
    MAX_STEPS = None # None -> run until time/termination guards stop the episode
    ENV_MAX_TIME = 60.0
    MAX_TOUCHES = 3
    NEVER_QUIT = True
    ACTION_MODE = "target"
    TARGET_STEER_DEADZONE = 0.05
    PAUSE_BETWEEN = False

    # Selection (choose one style):
    SORT_BY_FITNESS = True
    RANK_START = 1
    RANK_END = 32  # full mini population (32); set None for all selected from RANK_START onward
    EXACT_INDICES = None  # e.g. [0, 17, 42] (population indices from file)

    if MODEL_FILE:
        drive_model(
            map_name=MAP_NAME,
            model_file=MODEL_FILE,
            max_steps=MAX_STEPS,
            env_max_time=ENV_MAX_TIME,
            max_touches=MAX_TOUCHES,
            never_quit=NEVER_QUIT,
            action_mode=ACTION_MODE,
            target_steer_deadzone=TARGET_STEER_DEADZONE,
        )
    else:
        replay_population(
            map_name=MAP_NAME,
            population_file=POPULATION_FILE,
            episodes_per_individual=EPISODES_PER_INDIVIDUAL,
            max_steps=MAX_STEPS,
            env_max_time=ENV_MAX_TIME,
            max_touches=MAX_TOUCHES,
            never_quit=NEVER_QUIT,
            action_mode=ACTION_MODE,
            pause_between=PAUSE_BETWEEN,
            sort_by_fitness=SORT_BY_FITNESS,
            rank_start=RANK_START,
            rank_end=RANK_END,
            exact_indices=EXACT_INDICES,
            target_steer_deadzone=TARGET_STEER_DEADZONE,
        )

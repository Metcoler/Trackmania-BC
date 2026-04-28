import glob
import os
from datetime import datetime
from typing import Optional, Sequence

from Car import Car
from Enviroment import RacingGameEnviroment
from ReinforcementTrainer import (
    RLTrainingLogger,
    ReinforcementTrainer,
    build_policy,
    choose_device,
)


MAP_NAME = "AI Training #5"
EPISODES_TO_RUN = 5000
ENV_MAX_TIME = 30.0
CHECKPOINT_EVERY = 50

HIDDEN_DIM: Sequence[int] = [32, 16]
HIDDEN_ACTIVATION: Sequence[str] = ["relu", "tanh"]
ACTION_MODE = "target"
VERTICAL_MODE = False

LEARNING_RATE = 3e-4
STEER_STD = 0.35
STEER_STD_DECAY = 1.0
STEER_STD_MIN = 0.08
ENTROPY_COEF = 0.001
GRAD_CLIP_NORM = 1.0
BASELINE_BETA = 0.9
ADVANTAGE_CLIP = 5.0

MAX_STEPS = None
MAX_RUNTIME_MINUTES = None
DEVICE = "cpu"

MAX_TOUCHES = 1
START_IDLE_MAX_TIME = 2.0

# If a short RL smoke test exists, continue from its newest best model.
# Set to False if you want a fully random overnight run.
PREFER_LATEST_RL_BEST_MODEL = False
INITIAL_MODEL_PATH: Optional[str] = None


def find_latest_rl_best_model(base_dir: str = "logs/rl_runs") -> Optional[str]:
    files = glob.glob(f"{base_dir}/**/best_model.pt", recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def main() -> None:
    device = choose_device(DEVICE)
    initial_model_path = INITIAL_MODEL_PATH
    if initial_model_path is None and PREFER_LATEST_RL_BEST_MODEL:
        initial_model_path = find_latest_rl_best_model()

    env = RacingGameEnviroment(
        map_name=MAP_NAME,
        never_quit=False,
        action_mode=ACTION_MODE,
        dt_ref=1.0 / 100.0,
        dt_ratio_clip=3.0,
        vertical_mode=VERTICAL_MODE,
        surface_step_size=Car.SURFACE_STEP_SIZE,
        surface_probe_height=Car.SURFACE_PROBE_HEIGHT,
        surface_ray_lift=Car.SURFACE_RAY_LIFT,
        max_time=ENV_MAX_TIME,
        max_touches=MAX_TOUCHES,
        start_idle_max_time=START_IDLE_MAX_TIME,
    )

    try:
        obs, _ = env.reset()
        obs_dim = int(obs.shape[0])
        policy, initial_model_extra = build_policy(
            obs_dim=obs_dim,
            hidden_dim=HIDDEN_DIM,
            hidden_activation=HIDDEN_ACTIVATION,
            action_mode=ACTION_MODE,
            device=device,
            initial_model_path=initial_model_path,
        )

        run_name = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            f"_overnight_map_{MAP_NAME}_{'v3d' if VERTICAL_MODE else 'v2d'}"
            f"_h{'x'.join(str(dim) for dim in policy.hidden_dims)}_rl"
        )
        config = dict(
            trainer="reinforce_terminal_overnight",
            map_name=MAP_NAME,
            episodes_to_run=EPISODES_TO_RUN,
            env_max_time=ENV_MAX_TIME,
            max_steps=MAX_STEPS,
            max_runtime_minutes=MAX_RUNTIME_MINUTES,
            checkpoint_every=CHECKPOINT_EVERY,
            obs_dim=obs_dim,
            action_mode=ACTION_MODE,
            vertical_mode=VERTICAL_MODE,
            hidden_dim=list(policy.hidden_dims),
            hidden_activation=list(policy.hidden_activations),
            learning_rate=LEARNING_RATE,
            steer_std=STEER_STD,
            steer_std_decay=STEER_STD_DECAY,
            steer_std_min=STEER_STD_MIN,
            entropy_coef=ENTROPY_COEF,
            grad_clip_norm=GRAD_CLIP_NORM,
            baseline_beta=BASELINE_BETA,
            advantage_clip=ADVANTAGE_CLIP,
            max_touches=MAX_TOUCHES,
            start_idle_max_time=START_IDLE_MAX_TIME,
            prefer_latest_rl_best_model=PREFER_LATEST_RL_BEST_MODEL,
            initial_model_path=initial_model_path,
            initial_model_extra=initial_model_extra,
            device=str(device),
            debug=False,
        )
        logger = RLTrainingLogger(
            base_dir="logs/rl_runs",
            run_name=run_name,
            config=config,
        )

        print("\n[RL OVERNIGHT] Prepared terminal-only REINFORCE trainer")
        print(f"[RL OVERNIGHT] run_dir={logger.run_dir}")
        print(
            f"[RL OVERNIGHT] map={MAP_NAME!r} obs_dim={obs_dim} "
            f"action_mode={ACTION_MODE} vertical_mode={VERTICAL_MODE}"
        )
        print(
            f"[RL OVERNIGHT] network: {obs_dim} -> "
            f"{' -> '.join(str(dim) for dim in policy.hidden_dims)} -> 3 "
            f"activations={list(policy.hidden_activations)}"
        )
        print(
            f"[RL OVERNIGHT] episodes={EPISODES_TO_RUN} "
            f"env_max_time={ENV_MAX_TIME:.1f}s checkpoint_every={CHECKPOINT_EVERY}"
        )
        print(
            f"[RL OVERNIGHT] lr={LEARNING_RATE} steer_std={STEER_STD} "
            f"entropy_coef={ENTROPY_COEF} grad_clip={GRAD_CLIP_NORM}"
        )
        if initial_model_path:
            print(f"[RL OVERNIGHT] initial_model_path={initial_model_path}")
        else:
            print("[RL OVERNIGHT] initial_model_path=None, starting from random weights")

        trainer = ReinforcementTrainer(
            env=env,
            policy=policy,
            learning_rate=LEARNING_RATE,
            steer_std=STEER_STD,
            steer_std_decay=STEER_STD_DECAY,
            steer_std_min=STEER_STD_MIN,
            entropy_coef=ENTROPY_COEF,
            grad_clip_norm=GRAD_CLIP_NORM,
            max_steps=MAX_STEPS,
            baseline_beta=BASELINE_BETA,
            advantage_clip=ADVANTAGE_CLIP,
            logger=logger,
            device=device,
            debug=False,
        )
        history = trainer.run(
            episodes=EPISODES_TO_RUN,
            checkpoint_every=CHECKPOINT_EVERY,
            verbose=True,
            max_runtime_minutes=MAX_RUNTIME_MINUTES,
        )
        if history:
            best = max(history, key=lambda item: item.fitness)
            print(
                f"Best RL episode: {best.episode}, "
                f"progress={best.progress:.1f}%, time={best.time_value:.2f}s, "
                f"fitness={best.fitness:.2f}"
            )
        print(f"Run artifacts saved in: {logger.run_dir}")
    except KeyboardInterrupt:
        print("\n[RL OVERNIGHT] Interrupted by user. Latest and best models remain saved.")
    finally:
        env.close()


if __name__ == "__main__":
    main()

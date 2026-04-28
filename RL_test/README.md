# Stable-Baselines3 RL Test

This folder is an isolated experiment that uses Stable-Baselines3 without touching the existing GA or custom RL trainers.

Default long training run (8 hours):

```powershell
python RL_test/train_sac_trackmania.py
```

Short smoke test:

```powershell
python RL_test/train_sac_trackmania.py --max-runtime-hours 0.08 --episodes 50 --total-timesteps 300000 --checkpoint-every-episodes 10
```

Plot the per-episode metrics after a run:

```powershell
python RL_test/plot_sb3_training.py --run-dir "logs/sb3_runs/<run_name>"
```

## Why SAC?

Trackmania control is continuous in steering and can be continuous for gas/brake when using analog triggers. SAC is an off-policy algorithm for continuous `Box` action spaces and is usually much more sample-efficient than plain terminal-only policy gradient.

The script configures SAC with:

```python
train_freq=(1, "episode")
```

This means gradient updates are scheduled after complete episodes/runs, not after every environment step.

## Reward Modes

`REWARD_MODE = "hybrid_progress_terminal_fitness"` is the default experiment. It gives sparse progress-delta rewards during the run and a dominant terminal reward based on `Individual.compute_scalar_fitness_for`, scaled down for SAC.

Available modes:

```python
"terminal_progress"  # only final progress at crash/timeout/finish
"terminal_fitness"   # current Individual.compute_scalar_fitness_for, scaled down
"progress_delta"     # per-step positive progress delta, no extra hand-tuned constants
"hybrid_progress_terminal_fitness"  # sparse progress deltas + dominant terminal fitness
```

The hybrid reward keeps time handling mostly in the terminal `Individual` score. The progress delta is emitted only every `PROGRESS_REWARD_INTERVAL_STEPS` steps and scaled by `PROGRESS_DELTA_SCALE`, so it guides exploration but does not dominate the final episode ranking.

## Real-Time Reset Note

Stable-Baselines3 normally resets the environment immediately after an episode. In Trackmania that is dangerous because the game keeps running while SAC performs post-episode gradient updates. The wrapper therefore defers the real Trackmania reset until the next `step()` call, right before the next action is applied.

## Network

Default SAC MLP:

```text
Actor:  obs_dim -> 128 ReLU -> 128 ReLU -> action_dim
Critic: obs_dim + action_dim -> 128 ReLU -> 128 ReLU -> Q
```

You can change this without editing code:

```powershell
python RL_test/train_sac_trackmania.py --net-arch 64,64 --activation-fn tanh
```

## Action Layout

The default `ACTION_LAYOUT = "gas_brake_steer"` uses three continuous actions:

```text
action[0] -> gas in [0, 1]
action[1] -> brake in [0, 1]
action[2] -> steer in [-1, 1]
```

`gas_steer` is also available and uses two continuous actions:

```text
action[0] -> gas in [0, 1], brake fixed to 0
action[1] -> steer in [-1, 1]
```

`gas_steer` can avoid the early SAC random policy spending half of startup exploration braking, but `gas_brake_steer` is the more complete control setup.

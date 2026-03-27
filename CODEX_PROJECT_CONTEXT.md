## Purpose

This file is a handoff/context document for another Codex/GPT-5.4 instance working on this repository.
It summarizes:

- what the application does
- how data flows through the system
- what each important file is responsible for
- which files should be indexed first
- how the project evolved since December 2025
- which experiments were tried and what conclusions were reached
- what the current baseline state is


## Project Summary

This repository implements an autonomous driving agent for Trackmania.

There are currently two main learning paths in the codebase:

- a Genetic Algorithm / neuroevolution path for live training in Trackmania
- a supervised learning path that records player driving data and trains a torch policy offline

The historical RL path is no longer active in the current workflow. Legacy RL reward code has been removed from `Enviroment.py`; the GA selection logic uses lexicographic metrics instead.

The project also contains Trackmania map extraction assets and an OpenPlanet plugin that streams game state over TCP to Python.


## Current High-Level Architecture

### Runtime dataflow

1. Trackmania runs with the OpenPlanet plugin in `Plugins/get_data_driver/main.as`.
2. The plugin streams one fixed-size packet per game frame over TCP on `127.0.0.1:9002`.
3. `Car.py` connects to the socket, reads packets in a background thread, keeps only the latest decoded packet, and exposes it to the rest of the Python app.
4. `Map.py` loads track geometry and logical path data from `Maps/ExportedBlocks/*.txt` and `Meshes/*.obj`.
5. `Car.py` combines the live packet with map/path state and lidar-style raycasts to produce:
   - laser distances
   - upcoming path instructions
   - progress info
   - direction alignment
6. `ObservationEncoder.py` standardizes those values into the neural-network observation vector.
7. A policy from `EvolutionPolicy.py` maps observation -> action.
8. `Enviroment.py` applies that action through `vgamepad` to Trackmania and enforces training guards such as timeout, touches, idle detection, and wall-ride detection.

### GA training dataflow

1. `EvolutionTrainer.py` initializes a population of `Individual` objects.
2. Each `Individual` contains an `EvolutionPolicy` and a flattened genome view over model parameters.
3. `EvolutionTrainer.py` evaluates each individual sequentially in Trackmania through `RacingGameEnviroment`.
4. The environment returns terminal status and telemetry.
5. The individual is ranked by:
   - `term`
   - `progress`
   - `time_bucket`
   - `distance`
6. The GA applies elitism, selection, crossover, mutation, and annealed mutation schedules.
7. Training logs and checkpoints are stored under `logs/ga_runs/...`.

### Supervised dataflow

1. `Actor.py` reads both:
   - Trackmania state through `Car.py`
   - real Xbox controller state through `XboxController.py`
2. While the human player is driving, it records attempts into `logs/supervised_data/...`.
3. `SupervisedTraining.py` loads all saved attempts, preprocesses them, applies mirror augmentation, and trains a torch MLP policy in target-action mode.
4. Trained models are stored under `logs/supervised_runs/.../best_model.pt`.
5. `Driver.py` can load the latest supervised model and replay it in Trackmania.
6. `EvolutionTrainer.py` can also seed a GA population from a `.pt` supervised model.


## Important Current Semantics

### Observation

The current observation is built in `ObservationEncoder.py`.

Current observation layout:

- `15` laser distances
- `10` path instructions
- `speed`
- `side_speed`
- `next_point_direction`
- `dt_ratio`

Current observation dimension:

- `15 + 10 + 4 = 29`

Important history:

- `previous_action` used to be part of the observation.
- It was removed from the supervised-target pipeline because it created strong label leakage:
  the network learned to repeat the previous action instead of reacting to state, especially failing at the very first frame after race start.

### Action modes

Two action semantics exist in the project:

- `delta`
  - policy outputs a delta action
  - environment integrates it into the previous applied action
  - `dt_ratio` is used to scale the delta
- `target`
  - policy outputs the target action directly
  - this is the active direction for supervised learning

### Current target-action semantics

In target mode:

- `gas` is a sigmoid output in `[0, 1]`
- `brake` is a sigmoid output in `[0, 1]`
- `steer` is a tanh output in `[-1, 1]`

At environment/controller application time:

- `gas` is thresholded at `0.5`
- `brake` is thresholded at `0.5`
- both may be active simultaneously
- `steer` remains analog in `[-1, 1]`

The same binary pedal semantics are used when collecting supervised data in `Actor.py`.


## Core Files To Index First

If another Codex instance needs to understand the project efficiently, index files in this order:

1. `CODEX_PROJECT_CONTEXT.md`
2. `ObservationEncoder.py`
3. `Car.py`
4. `Map.py`
5. `Enviroment.py`
6. `EvolutionPolicy.py`
7. `Individual.py`
8. `EvolutionTrainer.py`
9. `Driver.py`
10. `Actor.py`
11. `SupervisedTraining.py`
12. `XboxController.py`
13. `Plugins/get_data_driver/main.as`
14. `README.md`

Secondary files:

- `GraphView.py`
- `Vizualizer.py`
- `installation.txt`
- `Backup/numpy_logic_20260317_133951/*`


## File Responsibilities

### `Plugins/get_data_driver/main.as`

OpenPlanet plugin.

Responsibilities:

- opens TCP server on `127.0.0.1:9002`
- streams 16 floats every Trackmania frame
- includes:
  - speed
  - side speed
  - distance
  - position
  - steer/gas/brake inputs
  - finish flag
  - gear / rpm
  - direction vector
  - game time

This is the root of the live runtime data stream.

### `Car.py`

Bridge between Trackmania packets and Python-side state.

Responsibilities:

- connect to the OpenPlanet TCP stream
- keep latest packet only
- derive map/path progress
- derive future path instructions
- compute `next_point_direction`
- compute lidar-style laser distances against map walls

Important implementation detail:

- the reader thread stores only the latest decoded packet, so the system does not intentionally process an old packet backlog frame by frame

### `Map.py`

Map geometry and logical path representation.

Responsibilities:

- parse exported block files from `Maps/ExportedBlocks/*.txt`
- instantiate mesh blocks from `Meshes/*.obj`
- construct the logical path from start to finish
- provide road mesh and wall mesh for geometry queries

### `ObservationEncoder.py`

Canonical observation builder.

Responsibilities:

- standardize distances and motion values
- compute `dt_ratio = dt / dt_ref`
- expose observation bounds
- provide mirror helpers for observations and actions

This file should be treated as the single source of truth for observation format.

### `Enviroment.py`

Trackmania environment wrapper.

Responsibilities:

- hold the `Map`, `Car`, and `vgamepad` controller
- reset the game state
- build observations through `ObservationEncoder`
- apply actions in delta or target mode
- enforce termination/truncation conditions

Important guards currently implemented:

- `max_time`
- `wrong-way`
- `start_idle`
- `stuck_after_progress`
- `max_touches`
- `wall_ride`

Important note:

- old RL reward logic was removed
- current reward returned by `step()` is neutral (`0.0`)
- GA optimization does not use per-step reward

### `EvolutionPolicy.py`

Torch policy network.

Responsibilities:

- define the MLP policy
- support one or more hidden layers with per-layer activations
- support `delta` and `target` action modes
- expose flattened genome view for GA
- save/load `.pt` policy files

Important:

- this is now the canonical model implementation
- older numpy policy versions are preserved in `Backup/numpy_logic_20260317_133951`

### `Individual.py`

GA individual wrapper around the policy.

Responsibilities:

- hold evaluation metrics
- expose ranking key
- provide mutation and crossover
- provide scalar fitness only as a log-friendly numeric proxy

Current ranking logic:

- `term < 0`
  - crash-like failure
  - lower values are worse, e.g. `-3` is worse than `-1`
- `term = 0`
  - timeout / truncation
- `term = 1`
  - reached finish

Current ranking policy in `Individual.ranking_key()`:

- for unfinished runs (`term <= 0`):
  - rank by `term`, then `progress`, then exact `time`
  - `distance` is ignored
- for finished runs (`term > 0`):
  - rank by `term`, then `progress`, then `time_bucket`, then `distance`

Reason for this design:

- unfinished agents were finding a local minimum where they drove into the first wall with a short traveled distance
- to avoid rewarding that behavior, `distance` is only minimized among finished runs

### `EvolutionTrainer.py`

Main GA trainer.

Responsibilities:

- population initialization
- optional seeding from supervised `.pt` model
- optional resume from population `.npz`
- individual evaluation in Trackmania
- logging/checkpointing
- mutation annealing

Current baseline default in `__main__`:

- map: `AI Training #3`
- hidden dim: `32`
- population: `64`
- generations: `100`
- action mode: `target`
- no supervised pretraining
- no mirroring
- `max_touches = 1`
- `env_max_time = 60`
- mutation starts exploratory and anneals down

### `Driver.py`

Evaluation and replay tool.

Responsibilities:

- drive a single `.pt` supervised model
- or replay individuals from a population `.npz`
- auto-pick latest supervised model if configured

Useful for sanity checks before starting long GA runs.

### `Actor.py`

Supervised data collection tool.

Responsibilities:

- read Trackmania state
- read physical Xbox controller state
- record attempts into `.npz`

Attempt workflow:

- recording starts when game time becomes `> 0`
- pressing `B` during a run discards the attempt
- after finish:
  - `A` saves the attempt
  - `B` discards it

### `SupervisedTraining.py`

Offline torch training script for imitation learning.

Responsibilities:

- load all attempts from `logs/supervised_data`
- preprocess frames
- optionally filter boring frames
- mirror-augment the dataset
- train a target-action MLP
- save `best_model.pt`

Current simplification trend:

- one hidden layer with `16` neurons
- no validation split
- all frames pooled together and shuffled

### `XboxController.py`

Dedicated Xbox controller reader using `inputs`.

Responsibilities:

- read gas / brake / steer
- read `A` and `B`
- apply steer deadzone

### `Vizualizer.py`

Legacy/auxiliary visualization script for scene inspection and debugging.

### `GraphView.py`

Plotting and post-run analysis separated from the GA trainer.


## Historical Evolution Since December 2025

### December 2025 baseline

Relevant commit:

- `d282cc2` - training history and graph plot

Project state around this period:

- simpler GA pipeline
- no torch policy
- no supervised learning pipeline
- no advanced guard logic
- no resume/checkpoint system comparable to current state

This period is important because the user reported that a simpler earlier trainer could sometimes train a finisher more reliably than the later experimental versions.

### February 2026: GA infrastructure expansion

Relevant commits:

- `606e559`
- `c459f3a`

Main additions:

- separated graphing from trainer via `GraphView`
- added per-run logging
- added resumable population checkpoints
- added persistent `global_best`
- added more robust experiment management

### February 2026: runtime/control experiments

Relevant commit:

- `b3102f8`

Main additions:

- normalized observation
- added `dt_ratio`
- introduced dt-aware control semantics
- added Xbox controller debug reader

### March 2026: torch migration and supervised pipeline

Relevant commit:

- `9ffd709`

Main additions:

- replaced numpy policy with torch-based policy
- introduced shared policy representation for supervised, GA, and driver
- added supervised attempt collection
- added supervised training script
- added seeding GA population from a `.pt` supervised model
- backed up the old numpy logic under `Backup/numpy_logic_20260317_133951`

### March 2026: supervised target pipeline refinement

Relevant commit:

- `f062277`

Main additions:

- refined supervised target-action workflow
- aligned Actor, Environment, Driver, and GA around target semantics
- several iterations on pedal thresholding and model structure

### March 2026: current baseline cleanup

Relevant commits:

- `61e28f2`
- `5742638`
- `04bf526`

Main changes:

- baseline training defaults for cleaner GA runs
- old reward function removed from environment
- focus shifted from feature accumulation to establishing a stable baseline again


## Important Experiments Already Tried

This section is critical. Another Codex instance should not rediscover these from scratch.

### 1. Mirror augmentation

Tried in both the mini project and Trackmania GA.

Goal:

- reduce one-sided overfitting
- teach left/right symmetry

Status:

- mechanism exists
- currently disabled in the baseline trainer because the user wants a simpler baseline first

### 2. Multi-touch instead of instant crash

Goal:

- allow a few small contacts before terminating

Implementation:

- `max_touches`
- touch debounce
- wall-ride guard

Status:

- still implemented
- baseline currently uses `max_touches = 1`

### 3. Target vs delta action

This has been one of the biggest experimental branches.

Observations:

- delta mode historically felt more stable in some GA runs
- target mode is more natural for supervised imitation learning
- target mode initially failed badly due to action semantics inconsistencies and overly strong shortcuts

Current status:

- supervised path is target-mode oriented
- baseline GA currently also defaults to target mode
- this is still an area of uncertainty and comparison

### 4. Previous action in the observation

Originally added to provide temporal context.

Result:

- in supervised target training it became a harmful shortcut
- the network learned to copy previous action instead of initiating correct start behavior
- especially bad at race start: no gas/brake on first frame

Decision:

- removed from the observation
- observation dimension reduced to `29`

### 5. dt_ratio input

Added because Trackmania/OpenPlanet produces variable frame timing.

Current belief:

- keeping `dt_ratio` in the observation is reasonable
- in delta mode it should scale the delta action
- in target mode it is still useful context but not used to scale outputs

### 6. Supervised validation split

A validation split existed previously.

Issues encountered:

- validation could become misleading
- real usefulness is determined by Driver replay in Trackmania, not by abstract validation loss
- map/run-based splitting created confusion in interpreting generalization

Current state:

- validation split was removed
- training uses all pooled frames

### 7. Large supervised model vs small supervised model

A larger model was tried first.

Current simplification:

- reduced to a much smaller MLP: one hidden layer, `16` neurons

Reason:

- simplify the hypothesis space
- test whether the pipeline works before increasing model capacity

### 8. Mini 2D pretraining project

Historically, a separate lightweight mini project existed and was used heavily for:

- cheap pretraining
- mirror experiments
- exporting TM-compatible checkpoints

Current repo state:

- the mini-project source is not present in the current top-level tracked files
- references to mini-project population checkpoints still exist in loaders and historical workflow discussions

Important:

- treat mini-project pretraining as a historical branch of experimentation, not as the current core runtime in this checkout


## Current Known Problems / Open Questions

These are active research/debug topics, not solved truths.

- The user reports difficulty training a reliable finisher despite adding many improvements.
- The simpler historical trainer sometimes seemed to work better.
- It is unclear whether target mode is truly better for GA than delta mode in Trackmania runtime.
- Supervised policies have sometimes:
  - failed to start properly
  - turned too weakly
  - behaved conservatively
- The exact amount of steer needed in Trackmania relative to the learned policy remains a practical issue.
- The influence of observation design vs policy architecture is still unresolved.
- The project has accumulated many safety/guard mechanisms; some may help, some may distort selection pressure.


## Current Recommended Debugging Order

If continuing experimentation, do not start by adding more complexity.

Recommended order:

1. Verify raw runtime data is sane:
   - plugin packets
   - `Car.py` derived values
   - observation ranges
2. Verify `Driver.py` behavior of the latest supervised model.
3. Verify action semantics end to end:
   - policy output
   - environment thresholding/clipping
   - vgamepad behavior in Trackmania
4. Only after runtime sanity is confirmed, launch GA runs.
5. Compare baseline `target` vs `delta` mode cleanly rather than mixing many new features at once.


## Current Practical Entry Points

### GA baseline

Run:

```powershell
python EvolutionTrainer.py
```

This currently uses the baseline config from `EvolutionTrainer.py`.

### Driver replay

Run:

```powershell
python Driver.py
```

By default, this auto-loads the latest supervised model.

### Supervised data collection

Run:

```powershell
python Actor.py
```

### Supervised training

Run:

```powershell
python SupervisedTraining.py
```


## Current Logs/Artifacts Layout

- `logs/ga_runs/...`
  - GA runs
  - summaries
  - population checkpoints
- `logs/supervised_data/...`
  - recorded human driving attempts
- `logs/supervised_runs/...`
  - trained supervised torch models
- `Backup/numpy_logic_20260317_133951/...`
  - backup of pre-torch numpy logic


## Environment / Dependencies

See `installation.txt`.

Important runtime dependencies:

- `torch`
- `numpy`
- `gymnasium`
- `trimesh`
- `vgamepad`
- `inputs`
- OpenPlanet plugin in Trackmania
- ViGEmBus driver for virtual gamepad


## Guidance For Another Codex Instance

When opening this project on another machine:

1. Read this file first.
2. Index the files listed in the "Core Files To Index First" section.
3. Assume the current priority is baseline reliability, not novelty.
4. Do not remove existing experimental features unless explicitly asked.
5. Treat supervised and GA as two connected but not yet fully stabilized pipelines.
6. Prefer simple A/B experiments over stacking multiple new ideas at once.


## Suggested Handoff Prompt

If another Codex/GPT-5.4 instance needs a starting prompt, use something like:

> Read `CODEX_PROJECT_CONTEXT.md` first, then index `ObservationEncoder.py`, `Car.py`, `Map.py`, `Enviroment.py`, `EvolutionPolicy.py`, `Individual.py`, `EvolutionTrainer.py`, `Driver.py`, `Actor.py`, and `SupervisedTraining.py`. This repository is a Trackmania autonomous driving project with a live GA/neuroevolution pipeline and a newer supervised-learning pipeline. The current priority is to restore a reliable baseline training workflow, not to add new complex features. Preserve existing experimental mechanisms, but reason from the current baseline defaults and from the historical experiments summarized in `CODEX_PROJECT_CONTEXT.md`.

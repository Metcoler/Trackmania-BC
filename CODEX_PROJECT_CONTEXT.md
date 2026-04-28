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
   - signed near/far heading alignment against upcoming path segments
6. `ObservationEncoder.py` standardizes those values into the neural-network observation vector.
7. A policy from `EvolutionPolicy.py` maps observation -> action.
8. `Enviroment.py` applies that action through `vgamepad` to Trackmania and enforces training guards such as timeout, touches, idle detection, and wall-ride detection.
9. `Enviroment.reset()` now performs a confirmed track restart handshake:
   - press `B` on the virtual gamepad
   - poll live telemetry until a negative `time` is observed
   - retry the `B` press several times if negative time is not seen
   This prevents the next individual from inheriting stale state from the previous attempt when Trackmania does not restart on the first button press.

### GA training dataflow

1. `EvolutionTrainer.py` initializes a population of `Individual` objects.
2. Each `Individual` contains an `EvolutionPolicy` and a flattened genome view over model parameters.
3. `EvolutionTrainer.py` evaluates each individual sequentially in Trackmania through `RacingGameEnviroment`.
4. The environment returns terminal status and telemetry.
5. The individual is ranked by:
   - averaged rollout fitness across normal and mirrored evaluation
   - telemetry summaries still keep representative aggregate metrics for logging
6. The GA applies elitism, selection from the top half, arithmetic-mean crossover, mutation, and annealed mutation schedules.
   Parent pairing now happens without repetition inside each pairing round; once the round is exhausted,
   the top-half parent pool is reshuffled and paired again if more children are still needed.
7. Training logs and checkpoints are stored under `logs/ga_runs/...`.
8. Population checkpoints now also store `current_mutation_prob` and `current_mutation_sigma`
   so resume can continue the annealing schedule from the actual checkpoint state instead of
   reusing only the values from the trainer script.

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
- `5` path instructions
- `speed`
- `side_speed`
- `segment_heading_error`
- `next_segment_heading_error`
- `dt_ratio`
- `FL/FR/RL/RR slip coefficients`
- `5` `surface_instruction_*` traction estimates aligned with the path lookahead
- `5` `height_instruction_*` values aligned with the path lookahead
  - `0.0` means same height
  - `+0.5` / `-0.5` means one logical height step up/down
  - `+1.0` / `-1.0` means two or more logical height steps up/down
- `longitudinal_accel`
- `lateral_accel`
- `yaw_rate`
- `5` overlapping laser clearance-rate sector averages

Current observation dimension:

- `15 + 5 + 27 = 47`

Current short-horizon settings:

- path lookahead horizon is `5` tiles
- lidar max range is `160` world units

Optional vertical-mode extension:

- `vertical_mode=False`
  - uses the current 2D observation at `47`
  - uses the legacy flat wall-only lidar
- `vertical_mode=True`
  - keeps the same leading `47` features
  - appends:
    - `vertical_speed`
    - `forward_y`
    - `support_normal_y`
    - `cross_slope`
    - `5` overlapping `surface_elevation_sector_*` features
  - current vertical-mode observation dimension is `56`

Current vertical-mode sensor semantics:

- `Car.py` can run in a simplified block-grid surface-following lidar mode
- the sensor first picks the current `MapBlock` from the car's X/Z grid cell and nearest fitted road plane height
- if the upcoming `SIGHT_TILES + 1` path tiles have no height change and the current support plane is flat,
  `vertical_mode=True` automatically uses the fast legacy flat wall raycast for that frame
- each `MapBlock` fits its road surface as a simple plane `y = ax + bz + c`
  - flat blocks become horizontal planes
  - slope blocks become one sloped plate
- on sloped blocks, laser directions are generated around the fitted road-plane normal and projected through the block-grid traversal
- each laser walks from grid cell to grid cell instead of triangle to triangle
  - within a cell, the ray follows the current block's fitted road plane at `surface_ray_lift`
  - wall checks are performed only against that block's `sensor_walls_mesh`
  - `sensor_walls_mesh` keeps the original wall mesh intact
  - slope blocks add simple vertical side-curtain polygons along road boundary edges
  - the curtains leave slope entry/exit open and only make the side catch surface taller
  - transitions to another block are allowed only when those blocks are connected in the logical path
  - if the ray exits the known block grid without a wall hit, the result is treated as `grid_gap`
  - if the ray reaches `LASER_MAX_DISTANCE`, the result is `grid_open`
  - if it hits padded block walls, the result is `grid_wall`
  - if it tries to enter a non-connected neighboring block, the result is `grid_blocked_transition`
- when `vertical_mode=False`, the old flat `walls_mesh`-only raycast remains active
- `surface_step_size` is kept only for backward config compatibility; the active vertical sensor no longer uses fixed marching steps

Current phased observation roadmap:

- current 2D upgrade
  - add per-wheel slip coefficients
  - add `5` compact future surface traction instructions:
    - `RoadTech` / asphalt: `1.00`
    - `PlatformGrass`: `0.70`
    - `RoadDirt` / `PlatformDirt`: `0.50`
    - `PlatformPlastic`: `0.75`
    - `PlatformIce`: `0.05`
    - `PlatformSnow`: `0.15` initial estimate
  - add compact temporal summary:
    - `longitudinal_accel`
    - `lateral_accel`
    - `yaw_rate`
    - `5` overlapping clearance-rate sectors derived from the lidar fan
  - add compact future height instructions:
    - `height_instruction_0..4`
    - `+0.5/-0.5` for one logical level step
    - `+1.0/-1.0` for two or more logical level steps
- next 2D/surface-aware upgrade
  - add gear
  - add rpm
- later 3D upgrade
  - current first 3D step
    - add toggleable `vertical_mode`
    - add block-grid surface-following lidar distances
    - add compact vertical block:
      - `vertical_speed`
      - `forward_y`
      - `support_normal_y`
      - `cross_slope`
      - `surface_elevation_sector_0..4`
  - later 3D expansion
    - add airborne/contact metrics
    - add richer orientation metrics
    - add more advanced surface/material state when needed

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
- streams 37 floats every Trackmania frame
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
  - FL/FR/RL/RR slip coefficients
  - FL/FR/RL/RR ground material diagnostics
  - FL/FR/RL/RR icing and dirt diagnostics
  - wetness

This is the root of the live runtime data stream.

### `Car.py`

Bridge between Trackmania packets and Python-side state.

Responsibilities:

- connect to the OpenPlanet TCP stream
- keep latest packet only
- derive map/path progress
- derive future path instructions
- derive future surface traction instructions
- derive future height-change instructions
- compute signed heading errors for the current and next future path segment
- compute lidar-style laser distances against map walls
- in `vertical_mode`, compute block-grid surface-following laser distances over fitted block road planes
- expose support-normal / slope debug data for the observation encoder and vizualizer

Important implementation detail:

- the reader thread stores only the latest decoded packet, so the system does not intentionally process an old packet backlog frame by frame

### `Map.py`

Map geometry and logical path representation.

Responsibilities:

- parse exported block files from `Maps/ExportedBlocks/*.txt`
- instantiate mesh blocks from `Meshes/*.obj`
- construct the logical path from start to finish
- expand block-level turn semantics into tile-aligned `path_instructions`
- expand block-level surface semantics into tile-aligned `path_surface_instructions`
- expand tile-level height deltas into tile-aligned `path_height_instructions`
- provide road mesh and wall mesh for geometry queries
- provide fitted road planes, X/Z block-grid lookup, logical path block transitions, and per-block sensor walls/side curtains for 3D laser walking

### `ObservationEncoder.py`

Canonical observation builder.

Responsibilities:

- standardize distances and motion values
- standardize per-wheel slip coefficients
- standardize future surface traction instructions
- standardize future height-change instructions
- derive compact temporal motion features from previous vs current frame
- optionally append compact vertical terrain features in `vertical_mode`
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
- The steering cue in `Car.py` now uses signed current/next segment heading errors instead of a pure dot-product alignment scalar.
  This should preserve left/right information, but it is still worth verifying on live maps that the sign convention matches intuitive steering direction.
- A previous bug came from `path_instructions` being stored at block granularity while `path_tile_index` advanced at tile granularity.
  `Map.py` now expands instructions to tile-aligned entries so the lookahead slice in `Car.py` stays synchronized through multi-tile corners.


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

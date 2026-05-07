# dd_mpc

A Model Predictive Controller for differential-drive robots, written in C++20.
It tracks arbitrary 2-D polyline paths at high speed using a receding-horizon
QP formulation, smoothed arc-based path representation, adaptive corridor
constraints, and an OSQP back-end with warm-starting and incremental updates.
A Python visualiser provides real-time animated playback of simulation runs.

---

## Table of Contents

<details><summary>Expand</summary>

- [Overview](#overview)
- [Architecture](#architecture)
- [Dependencies](#dependencies)
- [Building](#building)
- [Running the Simulator](#running-the-simulator)
  - [Standard Mode](#standard-mode)
  - [Random Staged Mode](#random-staged-mode)
  - [All CLI Flags](#all-cli-flags)
- [Configuration File](#configuration-file)
- [Python Visualiser](#python-visualiser)
  - [Panels](#panels)
  - [Visualiser CLI Flags](#visualiser-cli-flags)
  - [Keyboard Controls](#keyboard-controls)
- [Project Structure](#project-structure)
- [Parameter Reference](#parameter-reference)

</details>

---

## Overview

`dd_mpc` solves the path-tracking problem for a unicycle (differential-drive)
robot. At every timestep it:

1. Projects the robot onto a smoothed version of the input polyline path.
2. Builds a linearised discrete-time model around the current operating point.
3. Generates a horizon-length reference trajectory with an adaptive speed
   profile that slows for tight corners.
4. Assembles and solves a constrained QP (via OSQP) that minimises tracking
   error subject to velocity, acceleration, and corridor constraints.
5. Applies the first control output and warm-starts the next solve from the
   shifted solution.

Robustness features include latency compensation, reference blending on path
updates, a widening feasibility funnel when recovering from large cross-track
errors, Stanley heading correction at low speeds, and graceful fallback
(velocity decay) on solver failure.

---

## Architecture

<details><summary>Expand</summary>
```
MPCController::update()
│
├── latencyCompensate()          Forward-integrate current state by
│                                feedback_delay_s to compensate for
│                                actuator/communication lag.
│
├── ReferenceGenerator::generate()
│   ├── PathSmoother::smooth()   Convert polyline to line + circular-arc
│   │                            segments. Arc radii are kinematically
│   │                            constrained (r ≤ v_max / omega_max) and
│   │                            geometrically fitted to prevent overlap.
│   │                            Result is cached and reused while the
│   │                            path hash is unchanged.
│   └── sample horizon           Walk the smooth path forward N steps,
│                                sampling position, heading, left-normal,
│                                and arc speed limit at each knot.
│
├── blend()                      Blend new reference with previous when
│                                the path changes, avoiding step changes
│                                in the QP objective.
│
├── Linearizer::linearize()      First-order Taylor expansion of the
│                                unicycle model around each reference
│                                point to get affine A_k, B_k, c_k.
│
├── QPBuilder::build()           Assemble sparse P, q, A, l, u matrices.
│   │                            Decision vector:
│   │                              z = [x_0..x_N | u_0..u_{N-1} | ε_0..ε_N]
│   │                                   states       controls      slack
│   ├── Objective                Quadratic tracking (Q_xy, Q_theta) +
│   │                            control rate penalty (R_rate_v/omega) +
│   │                            slack penalty (w_slack).
│   └── Constraints              Dynamics equality, velocity/accel box,
│                                corridor inequality with slack variable,
│                                optional terminal stop (near goal).
│
└── Solver::update()             OSQP wrapper.
    ├── Full setup               Only on first call or horizon change.
    ├── Incremental update       osqp_update_data_mat / _vec on subsequent
    │                            calls — skips KKT re-factorisation when
    │                            only the linear/bound terms change.
    └── Warm start               Previous solution shifted by one step.
```
</details>

---

## Dependencies

| Dependency | Version | Notes |
|---|---|---|
| C++ compiler | C++20 | GCC ≥ 11 or Clang ≥ 14 |
| CMake | ≥ 3.16 | |
| Eigen3 | ≥ 3.4 | Linear algebra |
| OSQP | any recent | Included as a git submodule, built statically |
| Python | ≥ 3.9 | Visualiser only |
| NumPy | any | Visualiser only |
| Matplotlib | ≥ 3.5 | Visualiser only |

OSQP is expected as a git submodule in the `osqp/` directory. To initialise it:

```bash
git submodule update --init --recursive
```

---

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

The build produces a single executable: `build/sim_test`.

---

## Running the Simulator

<details><summary>Expand</summary>

The simulator runs a full closed-loop MPC control loop, writing one JSON frame
per timestep to stdout (and to `sim_data.jsonl`). Status messages go to stderr.

### Standard Mode

```bash
# Built-in straight path with mid-run update
./build/sim_test

# Named scenario
./build/sim_test --scenario 1   # sharp L-turn
./build/sim_test --scenario 2   # figure-8
./build/sim_test --scenario 3   # straight, robot starts facing backwards

# Custom path (whitespace-delimited "x y" file, one point per line)
./build/sim_test --path my_path.txt

# Load MPC parameters from a config file
./build/sim_test --config config/mpc_default.cfg

# With sensor noise and path pruning
./build/sim_test --noisy --prune
```

### Random Staged Mode

In random mode the robot starts at a random pose and follows a randomly
generated path. As it approaches the end of each path a new one is generated
seamlessly from the current pose and heading, testing the MPC's dynamic
path-update handling under realistic conditions.

```bash
# 5 stages at default complexity (2)
./build/sim_test --random

# 10 stages at high complexity
./build/sim_test --random --stages 10 --complexity 5

# Reproducible run
./build/sim_test --random --seed 42

# All options combined
./build/sim_test --random --complexity 3 --stages 8 --seed 1234 \
                 --noisy --prune --config config/mpc_default.cfg
```

**Complexity** controls path geometry on a 1–5 scale:

| Complexity | Segments | Segment length | In-segment drift | Boundary kink |
|---|---|---|---|---|
| 1 | 4 | ≈ 2.8 m | ±0.24 rad | ±0.40 rad |
| 2 | 6 | ≈ 3.6 m | ±0.34 rad | ±0.55 rad |
| 3 | 8 | ≈ 4.4 m | ±0.43 rad | ±0.70 rad |
| 4 | 10 | ≈ 5.2 m | ±0.53 rad | ±0.85 rad |
| 5 | 12 | ≈ 6.0 m | ±0.62 rad | ±1.00 rad |

Segment lengths have an additional ±40% random jitter. Boundary kinks are
instantaneous heading jumps applied at segment transitions, producing the
sharp dynamic events the MPC must react to.

### All CLI Flags

| Flag | Argument | Description |
|---|---|---|
| `--scenario` | `1`–`3` | Named built-in scenario |
| `--path` | file | Custom `x y` path file |
| `--config` | file | MPC parameter config file (see below) |
| `--noisy` | — | Gaussian sensor noise on position and heading |
| `--prune` | — | Remove traversed path segments each step |
| `--random` | — | Random staged mode |
| `--complexity` | `1`–`5` | Path difficulty in random mode (default: 2) |
| `--stages` | N | Number of random stages (default: 5) |
| `--seed` | N | RNG seed for reproducible random runs |

</details>

---

## Configuration File

<details><summary>Expand</summary>

All `MPCParams` fields can be overridden at runtime via a plain-text config
file. A fully-documented default file is provided at `config/mpc_default.cfg`.

Format rules:
- One `key = value` pair per line.
- Lines beginning with `#` are comments. Inline `# comments` are also stripped.
- Keys match `MPCParams` field names exactly (case-sensitive).
- Values must be numeric. Unknown keys produce a warning and are ignored.
- Any omitted key retains its compiled-in default.

```ini
# Example: tighten the corridor and increase tracking aggressiveness
d_hard   = 0.03
Q_xy     = 40.0
Q_theta  = 30.0
v_max    = 2.0
```

</details>

---

## Python Visualiser

<details><summary>Expand</summary>

`tools/plot_sim.py` runs the C++ binary (or replays a saved `.jsonl` file) and
renders a live animated dashboard.

```bash
# Run binary and visualise immediately
python tools/plot_sim.py --binary ./build/sim_test

# Replay a saved run
python tools/plot_sim.py --file sim_data.jsonl

# Random staged run at complexity 4
python tools/plot_sim.py --binary ./build/sim_test --random --complexity 4

# Save to MP4 or GIF
python tools/plot_sim.py --binary ./build/sim_test --save run.mp4

# Print text statistics only
python tools/plot_sim.py --file sim_data.jsonl --stats

# Draw a custom path interactively, then run it
python tools/plot_sim.py --binary ./build/sim_test --draw
```

### Visualiser CLI Flags

All binary flags (`--noisy`, `--prune`, `--random`, `--complexity`, `--stages`,
`--seed`, `--config`, `--path`, `--scenario`) are forwarded directly to the
binary when using `--binary`. Additional visualiser-only flags:

| Flag | Argument | Description |
|---|---|---|
| `--binary` | path | C++ binary to run |
| `--file` | path | Replay a saved `.jsonl` file |
| `--data` | path | Also save binary output to this file (default: `sim_data.jsonl`) |
| `--fps` | N | Animation frame rate (default: 20) |
| `--save` | file | Save animation to MP4 or GIF instead of displaying |
| `--stats` | — | Print text statistics summary and exit |
| `--draw` | — | Interactively draw a custom path before running |

### Keyboard Controls

| Key | Action |
|---|---|
| `Space` | Pause / resume |
| `←` / `→` | Step one frame |
| `Home` / `End` | Jump to start / end |
| Slider | Scrub to any frame |

</details>

---

## Project Structure

<details><summary>Expand</summary>
```
dd_mpc/
├── CMakeLists.txt
├── config/
│   └── mpc_default.cfg          All MPCParams with documented defaults
├── include/
│   └── mpc/
│       ├── config_loader.hpp    Load MPCParams from a config file
│       ├── linearization.hpp    Unicycle first-order Taylor linearisation
│       ├── mpc_controller.hpp   Top-level controller interface
│       ├── params.hpp           MPCParams struct and defaults
│       ├── path.hpp             Path / PathPoint types
│       ├── path_smoother.hpp    Polyline → line + arc smoothing
│       ├── projection.hpp       2-D point projection onto a path
│       ├── qp_builder.hpp       Sparse QP assembly
│       ├── reference.hpp        Horizon-length reference generation
│       ├── solver.hpp           OSQP wrapper with warm-start
│       └── types.hpp            State, Control structs
├── src/
│   ├── linearization.cpp
│   ├── mpc_controller.cpp
│   ├── path_smoother.cpp
│   ├── projection.cpp
│   ├── qp_builder.cpp
│   ├── reference.cpp
│   └── solver.cpp
├── test/
│   ├── sim_main.cpp             Simulation entry point and CLI
│   ├── sim_utils.cpp            Plant model, path factories, logging
│   └── sim_utils.hpp
├── tools/
│   └── plot_sim.py              Python animated visualiser
└── osqp/                        Git submodule
```
</details>

---

## Parameter Reference

<details><summary>Expand</summary>

All parameters live in `include/mpc/params.hpp` and can be overridden via a
config file (see above).

### Prediction Horizon

| Parameter | Default | Description |
|---|---|---|
| `N` | `24` | Number of prediction steps |
| `dt` | `0.05` | Timestep [s] |
| `feedback_delay_s` | `0.0` | Latency compensation lookahead [s] |

### Kinematic Limits

| Parameter | Default | Description |
|---|---|---|
| `v_max` | `2.5` | Maximum forward speed [m/s] |
| `v_min` | `-0.1` | Maximum reverse speed [m/s] |
| `omega_max` | `1.5` | Maximum angular rate [rad/s] |
| `a_max` | `1.0` | Maximum linear acceleration [m/s²] |
| `alpha_max` | `5.0` | Maximum angular acceleration [rad/s²] |

### Corridor

| Parameter | Default | Description |
|---|---|---|
| `d_hard` | `0.05` | Corridor half-width [m] |
| `w_slack` | `100.0` | Quadratic slack penalty — increase to enforce tighter corridor |
| `funnel_decay_tau` | `5.0` | Time constant for funnel narrowing after recovery [s] |

### Tracking Costs

| Parameter | Default | Description |
|---|---|---|
| `Q_xy` | `20.0` | Lateral + longitudinal position tracking weight |
| `Q_theta` | `20.0` | Heading tracking weight |
| `Q_xy_terminal` | `50.0` | Terminal position weight |
| `Q_theta_terminal` | `30.0` | Terminal heading weight |
| `R_v` | `0.0` | Absolute velocity cost (usually left at zero) |
| `R_omega` | `0.0` | Absolute angular rate cost |
| `R_rate_v` | `3.0` | Velocity rate-of-change penalty (smooths acceleration) |
| `R_rate_omega` | `3.0` | Angular rate-of-change penalty (smooths steering) |

### Low-Speed / Heading Correction

| Parameter | Default | Description |
|---|---|---|
| `stanley_k` | `2.5` | Stanley gain (cross-track correction at low speed) |
| `stanley_v_min` | `0.15` | Speed below which Stanley correction is fully active [m/s] |
| `stanley_decay` | `0.15` | Blend decay rate between Stanley and MPC heading |

### Goal and Fallback

| Parameter | Default | Description |
|---|---|---|
| `goal_threshold` | `0.03` | Distance to goal endpoint to declare arrival [m] |
| `goal_cte_scale` | `2.0` | CTE scale factor used in near-goal detection |
| `fallback_decay` | `0.8` | Velocity multiplier applied each step when OSQP fails |
| `blend_alpha` | `1.0` | Reference blend weight on path change (`1.0` = use new reference only) |

</details>

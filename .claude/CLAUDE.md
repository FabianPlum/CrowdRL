# CrowdRL — Development Guide

## What this is

MARL-learned pedestrian navigation policies trained in procedural 2D environments, validated against IAS-7 (Forschungszentrum Jülich) controlled experiment data. Replaces/augments hand-crafted locomotion models in JuPedSim.

## Workflow rules

- When fixing bugs or optimizing, make ONE change at a time and verify it works before proceeding. Never push intermediate/broken state to the repo. If a fix surfaces new issues, stop and confirm direction with me before continuing.
- When I describe a change using domain terms (reward vs physics force, penalty type names, research doc vs implementation plan), ask for clarification if ambiguous rather than assuming. Rewards/losses affect the learning signal; physics forces change the simulation directly.
- This project uses Python 3.12, PyTorch (NOT JAX), and targets CUDA GPU training. When debugging CUDA/torch issues, try the simplest fix first and avoid chaining speculative approaches (e.g., LD_LIBRARY_PATH, conftest hacks). If a first attempt fails, pause and explain options before trying the next.

## Pre-commit / pre-push checklist

Always run the full test suite (`uv run pytest`) AND check linting (`uv run ruff check .`) before committing or pushing. Do not push without confirming all tests pass and lint is clean.

## Package architecture

```
crowdrl-core          ← shared foundation, NO RL/JuPedSim deps
crowdrl-env           ← depends on core + Gymnasium
crowdrl-torch         ← depends on env + PyTorch; GPU-batched reimplementation
                        of the env step. Training-only, NOT in the deployment
                        path, held in step with core by parity tests
crowdrl-train         ← depends on env + PyTorch
crowdrl-jupedsim      ← depends on core + ONNX Runtime (JuPedSim supplied
                        out-of-band, see below)
```

Five packages, not four. Build order: core → env → torch/train → jupedsim. The only artefact crossing from training to deployment is an `.onnx` policy file.

Note the real entry point is root `train_mappo.py` (~1,900 LOC), not `crowdrl-train/train.py` — the latter is the library loop.

## crowdrl-core

Pure geometry/perception/action library. Submodules:

- **geometry**: Shapely polygon handling, constrained Delaunay triangulation, navmesh construction, wall-segment extraction
- **navmesh**: A* on triangle adjacency graph + funnel algorithm (Simple Stupid Funnel) for true shortest-path computation through portal edges. Provides next-waypoint direction (2D) and path-deviation scalar (1D)
- **sensing**: Raycast engine (N rays, configurable FOV, head-anchored) + K-nearest-neighbour social query
- **observation**: Assembles full obs vector from WorldState. Single function, identical in training and deployment
- **action**: Maps 4D policy output → desired velocity + torso angle + head angle. Enforces ±90° head constraint
- **collision**: Elliptical agent collision detection + contact forces (used by training env; raycasts also need this)

### WorldState — the critical interface

Flat dataclass consumed by observation builder and sensing. Contains:
- Agent positions, velocities, torso orientations, head orientations, body dimensions (shoulder width, chest depth), goal positions (all as numpy arrays)
- Walkable polygon (Shapely Polygon with holes) + precomputed wall segments
- Precomputed navmesh (triangle adjacency + centroid graph + portal edges for funnel algorithm)

Both crowdrl-env and crowdrl-jupedsim populate WorldState. The obs builder never knows which produced it. **If WorldState population is correct, observations are numerically identical between training and deployment.** This is the transfer guarantee — test it.

## crowdrl-env

Gymnasium environment. Key components:

### Procedural geometry generator (tiers 0–5)

- **Tier 0**: Open fields (convex polygons, no obstacles)
- **Tier 1**: Corridors + bottlenecks (width 0.8–5.0m, aperture 0.6–2.0m)
- **Tier 2**: Branching corridors, T-junctions, L-bends, crossroads
- **Tier 3**: Rooms with furniture/obstacles, randomised exits
- **Tier 4**: Full building floors (rooms + corridors + stairwell zones)
- **Tier 5**: Multi-floor evacuation (Tier 4 floors connected via portal zones)

Higher tiers compose from lower-tier primitives. All output Shapely Polygons. Start with Tiers 0–2.

### Geometry format

All geometries are Shapely Polygons with holes, matching JuPedSim convention. Walkable area = polygon exterior; obstacles = polygon holes. Import IAS-7 test geometries through the same interface.

### Solvability verification

A* on navmesh verifies all (spawn, goal) pairs. Three modes:
- **Prune**: remove unsolvable agents, keep geometry
- **Regenerate**: discard geometry if >30% agents unsolvable
- **Strict**: all agents must be solvable (validation runs)

### Reward (3-tier)

- **Tier 1** — Sparse: goal bonus (+10), collision penalty (−1/step), timeout (−5)
- **Tier 2** — Smoothness: jerk penalty, angular acceleration penalty, preferred-speed deviation
- **Tier 3** — Distributional style matching from PeTrack trajectory data (velocity autocorrelation, neighbour-distance distributions)

## Observation space (80D base → 129D fully instrumented; 89D in production)

| Component | Dims | Details |
|-----------|------|---------|
| Ego state | 8 | goal dir (2), velocity (2), speed (1), preferred speed (1, raw m/s), torso angle (1), head angle rel. torso (1) |
| Social | K×7 = 56 | K=8 nearest: rel pos (2), rel vel (2), body orient (1), body dims (2) |
| Raycasts | N = 16 | Head-anchored, 200° FOV, normalised distances. Optional 2-channel (distance + hit-type) → 32D |
| Navmesh (optional) | 3 | Next-waypoint direction (2) + path deviation (1) |
| Temporal memory (optional) | 6 | Own-trajectory history: displacement from spawn, cumulative path length, path efficiency, elapsed fraction, windowed displacement + goal progress |
| Neighbour velocity history (optional) | K×2 = 16 | Per tracked neighbour, velocity change over the last W_n steps (acceleration proxy) |
| Neighbour trajectory features (optional) | K×3 = 24 | Per tracked neighbour, its own path efficiency + windowed displacement + goal progress |

All in egocentric frame. `ObsConfig.obs_dim` is the authority — do not compute it by hand.

- Base (ego + social + 1-channel rays): **80D**; with 2-channel rays 96D.
- Fully instrumented: **129D** (1-channel) / 145D (2-channel).
- **The shipped policy is 89D**: ego 8 + social 56 + rays 16 + navmesh 3 + temporal 6, with `use_goal_direction=False` (navigates by the routed waypoint alone) and `use_jupedsim_style_routing=True` (router-style waypoint at JuPedSim's 0.2 m portal inset, `path_deviation` pinned to 0.0). The guiding principle: **do not train on a signal deployment cannot supply.**

## Action space (4D continuous)

1. Desired speed (scalar; maps [-1, +1] to [-max_backward_speed, +max_forward_speed], negative = backing up)
2. Desired heading change (scalar)
3. Desired torso orientation change (scalar)
4. Desired head orientation change relative to torso (scalar, clamped ±90°)

Head and torso are independently actuated. Raycasts follow head. Torso change alters collision ellipse orientation.

## Training

- **Algorithm**: MAPPO (PPO with parameter sharing across agents)
- **Agent count**: 20–100 per episode, randomised
- **Curriculum**: start Tier 0–1 low density, ramp up tier and agent count
- **Export**: PyTorch → ONNX

## JuPedSim integration (crowdrl-jupedsim)

`LearnedPolicyModel` implements JuPedSim's operational model interface. Per timestep:

1. Read JuPedSim agent states → populate WorldState
2. Call core observation builder (same code as training)
3. Batch ONNX inference → 4D actions
4. Call core action interpreter → desired velocities
5. Return to JuPedSim simulation loop

**Orientation**: JuPedSim 2.0's custom-model layer lets the adapter own an arbitrary immutable per-agent state, so torso angle, head angle, preferred speed, body dimensions and the memory buffers are simply fields on the frozen `CrowdRLAgentState`. A neighbour's state is readable during the callback, so social sensing gets them too — the deployment observation is reconstructed faithfully, not approximated. (Earlier drafts weighed a private side-channel dict vs. a C++ PR extending JuPedSim's agent struct; both are retired.)

**JuPedSim is NOT installed as a dependency.** `crowdrl-jupedsim` declares none: the custom-model layer exists only on upstream `main` (2.0, unreleased), and declaring `jupedsim>=1.0` made `uv sync` install a 1.x wheel that silently shadowed the local source build. Supply 2.0 out-of-band via a `.pth` in the venv's site-packages pointing at the build's `lib/` and `python_modules/jupedsim`. Every JuPedSim-dependent test uses `pytest.importorskip`, so the suite stays green without a build.

## Current state

- **All five packages are active.** crowdrl-core, crowdrl-env, crowdrl-torch (the GPU training path that `train_mappo.py` actually drives), crowdrl-train, and crowdrl-jupedsim.
- **Delivered**: the JuPedSim deployment path — `LearnedPolicyModel`, the self-describing ONNX artefact (`example_model/policy_r0125.onnx`, schema v2: obs/action config + trained dynamics + provenance embedded), e2e scenarios, and `LockstepPolicyModel` as a byte-exact validation instrument. Also the wall-reward reshape (`feat/wall-reward-shaping`): graded + closing-speed-weighted wall proximity, wall-normal impact, and `wall_collision_penalty_cap`, all behind default-off flags in both reward engines. Suite: 774 passing without a local JuPedSim 2.0 build (5 JuPedSim-dependent modules `importorskip` away).
- **Open**: Tier 3 (distributional/style) reward, Tier 4–5 geometry, the IAS-7 geometry importer, the cross-model benchmark runner (LearnedPolicyModel vs CollisionFreeSpeedModel/SocialForceModel), and the behavioural weak spot — high-density scenarios where the policy trades goal completion for collision avoidance. The wall-shaping flags are **implemented but measured as a net regression so far**: against a post-fix control over 18 matched checkpoints the `wallshape` arm raises wall contact 38% and costs 0.062 high-density goal, winning only on the dense-corridor scenario it was designed around. The `wall_collision_penalty_cap` arm that might fix this was stopped at r0350 and has never been evaluated. See `plan/handover_2026-08-07.md`.
- **Reference doc**: `plan/CrowdRL_Project_Plan_v10.md` (full design rationale, milestones, risks, and the dated implementation progress log). v9 and earlier are superseded. Superseded plan docs, handovers and branch summaries live in `plan/archive/`.

## Development tooling

- **Package manager**: [uv](https://docs.astral.sh/uv/). Always use `uv run` to execute tools (pytest, ruff, pre-commit, etc.) and `uv sync` / `uv add` to manage dependencies. Never use bare `pip` or `pip install`.
- **Workspace**: uv workspace defined in root `pyproject.toml`. Install everything with `uv sync --all-packages --extra dev`.
- **Dev setup**: `make dev` installs all packages + dev deps + pre-commit hooks in one command.
- **Linting / formatting**: ruff (config in root `pyproject.toml`). Pre-commit hooks run ruff check + ruff format on every commit.
- **Pre-commit**: Installed automatically by `make dev`. Manual: `uv run pre-commit install`. Config in `.pre-commit-config.yaml`.
- **Testing**: `make test` or `uv run pytest`. Configured in root `pyproject.toml`.
- **CI**: GitHub Actions (`.github/workflows/ci.yml`) -- ruff lint + pytest on Python 3.12.

## Known issues

### Notebook editing

The NotebookEdit tool is unreliable for cell positioning. Prefer editing notebook JSON directly via Read/Edit/Bash when precise cell placement matters.

### Mojibake in notebooks / markdown

Unicode characters (em dash `—`, en dash `–`, degree `°`, arrows `→`) frequently get double-encoded in `.ipynb` files: UTF-8 bytes are misread as CP1252, then re-encoded as UTF-8, producing garbled sequences like `â€"`, `Ã‚Â°`, `â†'`. **When writing to notebooks, always use plain ASCII alternatives** (`--`, `-`, `deg`, `->`) instead of Unicode dashes, arrows, or special characters. If you spot mojibake in an existing file, fix it.

## Key design principles

1. **One observation builder, used everywhere.** Never duplicate obs construction logic.
2. **WorldState is the contract.** All perception code consumes WorldState only.
3. **Geometry is always Shapely Polygons.** Procedural and imported geometries share one interface.
4. **Test in isolation.** Core is testable with hand-built WorldState before any RL code exists.
5. **Ablation-friendly.** Observation components (raycasts, social, navmesh signals), action dimensions (2D/3D/4D), reward tiers, and FOV settings are all toggleable via config.
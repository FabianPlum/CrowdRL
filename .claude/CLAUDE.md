# CrowdRL — Development Guide

## What this is

MARL-learned pedestrian navigation policies trained in procedural 2D environments, validated against IAS-7 (Forschungszentrum Jülich) controlled experiment data. Replaces/augments hand-crafted locomotion models in JuPedSim.

## Package architecture

```
crowdrl-core          ← shared foundation, NO RL/JuPedSim deps
crowdrl-env           ← depends on core + Gymnasium
crowdrl-train         ← depends on env + PyTorch
crowdrl-jupedsim      ← depends on core + JuPedSim + ONNX Runtime
```

Build order: core → env → train → jupedsim. The only artefact crossing from train to jupedsim is an `.onnx` policy file.

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

## Observation space (~80–95D per agent)

| Component | Dims | Details |
|-----------|------|---------|
| Ego state | 7 | goal dir (2), velocity (2), heading (1), torso angle (1), head angle rel. torso (1) |
| Social | K×7 = 56 | K=8 nearest: rel pos (2), rel vel (2), body orient (1), body dims (2) |
| Raycasts | N = 16 | Head-anchored, 200° FOV, normalised distances. Optional 2-channel (distance + hit-type) → 32D |
| Navmesh (optional) | 3 | Next-waypoint direction (2) + path deviation (1) |

All in egocentric frame. Total: 79D (1-channel rays) to 98D (2-channel + navmesh).

## Action space (4D continuous)

1. Desired speed (scalar)
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

**Orientation gap**: JuPedSim agents have no torso/head angles. Strategy A (initial): adapter tracks orientation state privately. Strategy B (future): extend JuPedSim agent model via PR.

## Current state

- **Active**: Building crowdrl-core (WorldState, geometry, navmesh) and crowdrl-env (procedural generator Tiers 0–2)
- **Not started**: crowdrl-train, crowdrl-jupedsim, Tier 3 reward
- **Reference doc**: CrowdRL_Project_Plan_v6.md (full design rationale, milestones, risks)

## Development tooling

- **Package manager**: [uv](https://docs.astral.sh/uv/). Always use `uv run` to execute tools (pytest, ruff, pre-commit, etc.) and `uv sync` / `uv add` to manage dependencies. Never use bare `pip` or `pip install`.
- **Workspace**: uv workspace defined in root `pyproject.toml`. Install everything with `uv sync --all-packages --extra dev`.
- **Dev setup**: `make dev` installs all packages + dev deps + pre-commit hooks in one command.
- **Linting / formatting**: ruff (config in root `pyproject.toml`). Pre-commit hooks run ruff check + ruff format on every commit.
- **Pre-commit**: Installed automatically by `make dev`. Manual: `uv run pre-commit install`. Config in `.pre-commit-config.yaml`.
- **Testing**: `make test` or `uv run pytest`. Configured in root `pyproject.toml`.
- **CI**: GitHub Actions (`.github/workflows/ci.yml`) -- ruff lint + pytest on Python 3.12/3.13.

## Known issues

### Mojibake in notebooks / markdown

Unicode characters (em dash `—`, en dash `–`, degree `°`, arrows `→`) frequently get double-encoded in `.ipynb` files: UTF-8 bytes are misread as CP1252, then re-encoded as UTF-8, producing garbled sequences like `â€"`, `Ã‚Â°`, `â†'`. **When writing to notebooks, always use plain ASCII alternatives** (`--`, `-`, `deg`, `->`) instead of Unicode dashes, arrows, or special characters. If you spot mojibake in an existing file, fix it.

## Key design principles

1. **One observation builder, used everywhere.** Never duplicate obs construction logic.
2. **WorldState is the contract.** All perception code consumes WorldState only.
3. **Geometry is always Shapely Polygons.** Procedural and imported geometries share one interface.
4. **Test in isolation.** Core is testable with hand-built WorldState before any RL code exists.
5. **Ablation-friendly.** Observation components (raycasts, social, navmesh signals), action dimensions (2D/3D/4D), reward tiers, and FOV settings are all toggleable via config.

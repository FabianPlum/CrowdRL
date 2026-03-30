# GPU-Native Navmesh Waypoint Signals

**Status: Implemented** (2026-03-30)

## Problem

Agents in Tier 1-2 geometries (corridors, T-junctions, bottlenecks) only see a
straight-line goal direction that points through walls. They need shortest-path
guidance (navmesh waypoints) in their observation vector, but the current navmesh
code (A* + funnel) runs on CPU and cannot be called per-step without destroying
GPU throughput (>100k steps/s target).

## Key Insight

The navmesh is **static per episode** and waypoints are **sparse** (typically
1-8 turning points). The navigation task doesn't change every step -- only the
agent's position relative to the pre-computed path changes. This means:

1. Pre-compute the full funnel waypoint sequence per agent **once at episode
   reset** (CPU, amortised over thousands of steps).
2. Store the waypoint sequences as a padded GPU tensor.
3. Each step, run a **pure-tensor lookup** to find each agent's next waypoints
   and produce the 3D observation signal. No CPU round-trip.

## Design

### Phase 1: Pre-compute waypoints at reset (CPU)

At episode generation time (already CPU-bound -- geometry, spawning, solvability):

```
For each solvable agent i:
    path = shortest_path(navmesh, position_i, goal_i, agent_radius_i)
    waypoints_i = path[1:]  # drop start position, keep goal as final waypoint
```

Pad all agents' waypoint lists to a fixed `MAX_WAYPOINTS` (default 16) and
store as:

```
waypoints:             (n_agents, MAX_WAYPOINTS, 2)  float32 -- world-frame XY
n_waypoints:           (n_agents,)                   int32   -- actual count per agent
waypoint_path_lengths: (n_agents, MAX_WAYPOINTS)     float32 -- cumulative remaining
                                                               distance from each wp to goal
```

These arrays are padded to `MAX_AGENTS` and transferred to GPU alongside the
other reset data (positions, goals, wall segments, etc.). This adds
`MAX_AGENTS * MAX_WAYPOINTS * 2 * 4 bytes` = `64 * 16 * 2 * 4` = **8 KB** per
env -- negligible.

### Phase 2: Per-step GPU waypoint lookup

Each step, for each active agent, we produce a 3D signal:
`[waypoint_dir_x, waypoint_dir_y, path_deviation]` in ego frame.

The lookup is:

1. **Gather the two next waypoints.** A per-agent `waypoint_cursor` (int tensor)
   tracks progress through the waypoint sequence. `cursor_a = cursor` is the
   current waypoint; `cursor_b = min(cursor + 1, n_waypoints - 1)` is the next.
   Both are gathered via `torch.gather` on the `(E, N, MAX_WP, 2)` tensor.

2. **Blend the two waypoints.** Given `w_a` (current) and `w_b` (next):

   ```
   d_a = ||p - w_a||
   d_b = ||p - w_b||

   # Closer waypoint has LESS influence -- it's about to be crossed.
   # Farther waypoint has MORE influence -- it's the upcoming turn.
   weight_a = d_a / (d_a + d_b + eps)   # small when close to w_a
   weight_b = d_b / (d_a + d_b + eps)   # large when far from w_b
   blended_target = weight_a * w_a + weight_b * w_b
   ```

   When agent is right at `w_a` (d_a ~ 0), `weight_a ~ 0` and the signal
   points purely toward `w_b`. As the agent moves toward `w_b`, the blend
   shifts smoothly.

   **Edge case -- only one waypoint (the goal) remains:**
   When `cursor_a == cursor_b` (last waypoint), both point to the same
   location. The blended target is the goal itself with effective weight 1.0.

3. **Compute direction and deviation.**

   ```
   direction = normalize(blended_target - p)   # unit vector, world frame

   # Rotate to ego frame using pre-computed cos/sin of -torso_orientation
   dir_ego_x = cos_h * direction_x - sin_h * direction_y
   dir_ego_y = sin_h * direction_x + cos_h * direction_y

   # Path deviation: (remaining_path / euclidean_to_goal) - 1, clamped >= 0
   remaining_path = d_a + waypoint_path_lengths[cursor_a]
   euclidean = ||p - goal||
   path_deviation = (remaining_path / euclidean) - 1.0
   ```

4. **Advance the cursor** (in `step.py`, before observation building).

   ```
   if ||p - w_cursor|| < crossing_threshold:  # default 0.5m
       cursor += 1
   cursor = clamp(cursor, max=n_waypoints - 1)
   ```

   The cursor is monotonic -- once a waypoint is passed, it's never
   reconsidered.

### All operations are pure tensor ops

- Waypoint gathering: `torch.gather` on `(E, N, MAX_WP, 2)` using cursor indices
- Distance computation: standard vector norms
- Blending: element-wise weighted sum
- Ego-frame rotation: 2D rotation using torso orientation cos/sin (reuses
  cos_h/sin_h already computed for ego state)
- Cursor advance: `torch.where(d < threshold, cursor + 1, cursor)` with
  `clamp(max=n_waypoints - 1)`
- Path deviation: cumulative segment lengths indexed by cursor (O(1) lookup)

**Zero CPU involvement per step. Fully `torch.compile`-compatible.**

## Implementation

### `observation.py` (crowdrl-core) -- ObsConfig

- Added `navmesh_max_waypoints: int = 16` to control the padding size

### `types.py` -- EnvConfig + TorchWorldState

**EnvConfig:**
- Added `use_navmesh: bool = False`
- Added `max_waypoints: int = 16`
- Added `waypoint_crossing_threshold: float = 0.5`
- Mapped `use_navmesh` from `CrowdEnvConfig.obs.use_navmesh` in `from_crowd_env_config()`

**TorchWorldState:**
- Added `waypoints: Tensor  # (E, N, MAX_WP, 2)` -- pre-computed world-frame waypoints
- Added `n_waypoints: Tensor  # (E, N) int32` -- actual waypoint count per agent
- Added `waypoint_cursor: Tensor  # (E, N) int32` -- current progress index
- Added `waypoint_path_lengths: Tensor  # (E, N, MAX_WP)` -- cumulative remaining
  path length from each waypoint to the goal

### `episode_factory.py` -- Pre-compute waypoints

- After solvability check, compute `shortest_path()` per agent
- Extract waypoints (drop start position, keep intermediate + goal)
- Compute cumulative path lengths backwards from goal
- Return `"waypoints"`, `"n_waypoints"`, `"waypoint_path_lengths"` in the dict

### `geometry_repr.py` -- Pad waypoint arrays

- Extended `prepare_reset_data()` with optional `waypoints`, `n_waypoints`,
  `waypoint_path_lengths`, `max_waypoints` parameters
- Zero-pads to `(MAX_AGENTS, MAX_WP, ...)` shapes

### `batched_env.py` -- Thread waypoints through reset pipeline

- `_generate_reset_data()`: passes waypoint arrays from episode factory
- `_data_to_tensors()`: converts waypoint numpy arrays to GPU tensors
- `_stack_reset_data()`: includes waypoint tensors in TorchWorldState
- `_apply_completed_resets()`: assigns waypoint tensors + resets cursor on env reset
- `reset_all()`: passes waypoint state to `build_observations()`

### `observation.py` (crowdrl-torch) -- GPU navmesh signals

- Added `compute_navmesh_signals()` function (pure tensor ops):
  - Gathers current + next waypoints via cursor indices
  - Distance-weighted blending (closer = less influence)
  - Direction normalisation + ego-frame rotation
  - Path deviation from pre-computed cumulative lengths
  - Zeros out agents with no waypoints
- Modified `build_observations()` to accept optional waypoint tensors and
  conditionally concatenate the 3D navmesh signal

### `step.py` -- Advance waypoint cursor

- Step 10 (after wall enforcement, before state construction): computes distance
  to current waypoint, advances cursor via `torch.where` when within threshold
- Updated state construction to carry `waypoints`, `n_waypoints`,
  `waypoint_cursor`, `waypoint_path_lengths`
- Passes waypoint state to `build_observations()`

### Deviation from plan: cursor advancement logic

The plan proposed two criteria for passing a waypoint ("closer to next than
current" OR "within threshold"). The implementation uses only the threshold
criterion (`d < 0.5m`) which is simpler and sufficient -- the monotonic cursor
already prevents re-visiting passed waypoints.

## Computational Cost

**Per episode reset (CPU, amortised):**
- ~50 A* + funnel calls (one per agent), on navmeshes with ~50-100 triangles
- Already within the geometry generation thread pool time budget
- Adds ~1ms per episode reset -- invisible vs geometry generation (~5-10ms)

**Per step (GPU):**
- Two `torch.gather` on `(E, N, MAX_WP, 2)` (current + next waypoint)
- Two vector norms `(E, N)` for distances
- Weighted sum + normalisation: element-wise ops on `(E, N, 2)`
- One 2D rotation: 4 multiplies + 2 adds on `(E, N)`
- One conditional cursor advance: `torch.where` on `(E, N)`
- **Total: ~10 element-wise tensor ops on (E, N) tensors -- comparable cost to
  a single layer of the contact force computation. Well within torch.compile
  fusion capability.**

**Memory:**
- Waypoints: `E * MAX_AGENTS * MAX_WP * 2 * 4` = `64 * 64 * 16 * 2 * 4` = 512 KB
- Path lengths: `E * MAX_AGENTS * MAX_WP * 4` = 256 KB
- Cursor: `E * MAX_AGENTS * 4` = 16 KB
- **Total: ~800 KB -- negligible vs existing state tensors**

## Validation

1. **Equivalence test**: For a set of known geometries (Tier 1-2), verify that
   the GPU waypoint signals match the CPU `next_waypoint_direction()` output
   at the start of an episode (before any cursor advancement).

2. **Cursor advancement test**: Step agents along a known corridor path, verify
   the cursor advances at the correct positions and the blended direction
   transitions smoothly.

3. **Throughput test**: Measure steps/sec with and without navmesh signals.
   Target: <5% throughput reduction (the lookup is ~10 cheap tensor ops vs
   ~200 ops in the full step function).

4. **Training integration**: Run notebook 06 with `use_navmesh=True`, verify
   obs_dim=82 flows through correctly and training converges on Tier 1-2
   geometries with improved goal rates vs the no-navmesh baseline.

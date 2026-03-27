# Speedup & Parallelisation Plan

**Baseline**: ~1,769 agent-steps/sec with 32 env workers, high CPU / low GPU utilisation.

**Root cause**: The environment `step()` dominates wall-clock time. Neural network
forward/backward passes are tiny (~100 agents x 79D through 2x256 MLP). The GPU sits
idle while CPU workers grind through pure-Python scalar loops for raycasting, collision
detection, KNN sensing, and wall enforcement.

---

## Changes (in implementation order)

### 1. Vectorize `interpret_actions_batch` (action.py)

**Current**: Python list comprehension calling scalar `interpret_action()` per agent.
Each call does trig, clipping, angle normalisation one agent at a time.

**Change**: Rewrite as fully vectorized numpy operating on `(N, action_dim)` arrays.
Return a `BatchActionResult` dataclass with `(N, 2)` velocities and `(N,)` orientation
arrays instead of a list of `ActionResult` objects.

**Expected speedup**: 5-10x on action interpretation. Eliminates N Python function calls
and enables the caller (`crowd_env.step`) to do vectorized velocity assignment.

### 2. Vectorize velocity/force loops in `crowd_env.py` `step()`

**Current**: Lines 203-212 loop over agents for velocity damping; lines 230-234 loop
for contact force application. Both are trivially vectorizable.

**Change**: Replace with masked numpy operations:
```python
mask = self._active_mask
self._world.velocities[mask] = (
    cfg.velocity_damping * batch_result.desired_velocities[mask]
    + (1 - cfg.velocity_damping) * self._world.velocities[mask]
)
```
Same for contact forces and orientation updates.

**Expected speedup**: 2-5x on physics integration.

### 3. Deduplicate `detect_collisions` call

**Current**: `detect_collisions()` called in `compute_contact_forces()` (line 156) and
again in `step()` (line 222). Identical O(N^2) computation done twice per step.

**Change**: `compute_contact_forces` takes an optional pre-computed collisions list.
`step()` calls `detect_collisions()` once, passes result to both force computation and
reward collision mask.

**Expected speedup**: 2x on collision computation.

### 4. Vectorize collision detection (collision.py)

**Current**: Nested Python loops over all agent pairs, calling `ellipse_overlap()` per
pair with scalar trig.

**Change**: Fully vectorized `detect_collisions_vectorized()`:
- Compute all pairwise position differences as `(N, N, 2)` array
- Batch rotation matrices from torso orientations `(N, 2, 2)`
- Vectorized algebraic distance computation for both directions
- Extract colliding pairs with `np.where(overlap > 0)`

**Expected speedup**: 3-5x on collision detection.

### 5. Vectorize `compute_contact_forces` (collision.py)

**Current**: Python loop over collision pairs for spring-damper forces. Python loop
over agents x wall segments for exponential wall repulsion.

**Change**:
- Agent-agent forces: vectorized using collision pair indices from step 4.
- Wall forces: batch all agents against all segments using vectorized
  `_point_to_segment_nearest_batch()`. Compute distances and normals as `(N, W, 2)`
  arrays, apply exponential repulsion with numpy broadcasting.

**Expected speedup**: 2-3x on force computation.

### 6. Replace Shapely in `enforce_wall_boundaries` with numpy geometry

**Current**: Per-agent Shapely `Point()`, `boundary.distance()`, `polygon.contains()`,
`nearest_points()`. Creates Python objects every step, calls into Shapely C++ per agent.

**Change**: Vectorized numpy implementation:
- Compute all-agents-to-all-segments distances in one batch operation (reuse the batch
  point-to-segment function from step 5).
- Minimum distance per agent = boundary distance.
- Inside/outside test: use winding number algorithm vectorized over agents.
- Only process agents that fail the `inside and dist >= radius` check.

**Expected speedup**: 2-3x on wall enforcement.

### 7. Vectorize `knn_social` (sensing.py)

**Current**: Per-agent Python loop computing distances, appending to list, sorting.
O(N^2 log N) total.

**Change**: `knn_social_batch()` operating on all agents at once:
- Pairwise distance matrix: `np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)`
- Top-K via `np.argpartition` (O(N) vs O(N log N) sort)
- Batch feature extraction: relative positions, velocities, orientations computed with
  vectorized indexing and rotation matrices.

**Expected speedup**: 3-10x on social sensing.

### 8. Vectorize `cast_rays` (sensing.py)

**Current**: Triple nested loop: for each agent, for each ray, for each wall segment +
each other agent. Pure Python with scalar numpy calls. This is the single biggest
bottleneck.

**Change**: `cast_rays_batch()` operating on all agents simultaneously:

**Ray-segment intersections** (all agents x all rays x all segments):
- Pre-compute ray directions for all agents: `(N, R, 2)` from head angles
- Batch `_ray_segment_intersection`: vectorized cross-products and parameter solving
  on `(N, R, W)` shaped arrays
- Find per-ray minimum hit distance with `np.min` over segment axis

**Ray-ellipse intersections** (all agents x all rays x all other agents):
- Spatial pre-filter: agents outside `max_range` radius don't need testing
- Batch coordinate transform into each target ellipse's local frame
- Vectorized quadratic solver on `(N, R, M)` arrays (M = nearby agents)
- Combine with wall hits to get final per-ray readings

**Expected speedup**: 5-20x on raycasting (the dominant cost).

### 9. Vectorize `build_observations_batch` (observation.py)

**Current**: Python loop calling `build_observation()` per agent, which internally
calls `knn_social()` and `cast_rays()` per agent.

**Change**: Top-level `build_observations_batch()` calls the new batch functions:
1. Batch ego state computation (vectorized trig + rotation)
2. `knn_social_batch()` from step 7
3. `cast_rays_batch()` from step 8
4. Concatenate all components

The per-agent `build_observation()` remains for deployment (crowdrl-jupedsim) but
training uses the batch path.

**Expected speedup**: Combines gains from steps 7-8; eliminates N Python function calls.

### 10. Training loop config improvements

**Changes**:
- Increase default `n_steps_per_collect` from 2048 to 4096 for better GPU batch sizes.
- Make env reset in `RolloutCollector` non-blocking: send reset command immediately,
  skip that env for the current step, pick up the result next iteration. Prevents one
  env's slow reset (geometry generation + solvability) from blocking all others.

**Expected speedup**: 10-20% throughput gain from reduced idle time and better GPU
utilisation.

---

## Files modified

| File | Changes |
|------|---------|
| `crowdrl-core/src/crowdrl_core/action.py` | Add `interpret_actions_batch_vectorized`, return `BatchActionResult` |
| `crowdrl-core/src/crowdrl_core/collision.py` | Vectorize `ellipse_overlap`, `detect_collisions`, `compute_contact_forces`, `enforce_wall_boundaries` |
| `crowdrl-core/src/crowdrl_core/sensing.py` | Add `cast_rays_batch`, `knn_social_batch` |
| `crowdrl-core/src/crowdrl_core/observation.py` | Rewrite `build_observations_batch` to use batch sensing |
| `crowdrl-env/src/crowdrl_env/crowd_env.py` | Use vectorized action/physics/collision, pass collisions once |
| `crowdrl-train/src/crowdrl_train/config.py` | Increase `n_steps_per_collect` default |
| `crowdrl-train/src/crowdrl_train/rollout_collector.py` | Async env reset |

## Correctness guarantees

- All existing per-agent functions (`build_observation`, `cast_rays`, `knn_social`,
  `interpret_action`, `ellipse_overlap`) are preserved unchanged for deployment use and
  as reference implementations.
- New batch functions are tested against the scalar versions for numerical equivalence.
- Existing test suite must pass unchanged.

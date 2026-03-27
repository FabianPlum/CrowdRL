# GPU-Accelerated Environment Migration Plan

> **Status update (2026-03-27):** The original plan targeted JAX. After evaluation,
> we pivoted to a **PyTorch-native vectorized environment** instead. See
> [Decision: JAX → PyTorch](#decision-jax--pytorch) for the rationale. The JAX
> implementation (`packages/crowdrl-jax/`) will be removed once the PyTorch
> replacement is validated.

---

## Decision: JAX → PyTorch

### Why JAX was considered

The training loop is CPU-bound: 100 subprocess workers saturate 32 CPU cores
while the RTX 4090 sits idle. JAX's `jax.vmap` + `jax.jit` promised to move the
env hot loop (raycasting, collision, KNN) onto the GPU in one compiled kernel.

### Why JAX is the wrong tool here

1. **Two GPU frameworks, worst of both worlds.** Training stays in PyTorch
   (ONNX export path to JuPedSim is non-negotiable). Running JAX + PyTorch on
   the same GPU means two memory allocators, two CUDA contexts, and
   `XLA_PYTHON_CLIENT_MEM_FRACTION` hacks to split VRAM. DLPack zero-copy
   helps, but synchronisation between JAX's async dispatch and PyTorch's eager
   mode adds latency and subtle bugs.

2. **PureJaxRL is unreachable — the main payoff of JAX for RL.** The canonical
   reason to use JAX in RL is end-to-end `jax.jit` (env + policy + optimizer in
   one compiled graph, zero host interaction). We can never get there because:
   - Policy trains in PyTorch and exports to ONNX for JuPedSim deployment.
   - Geometry generation uses Shapely (CPU, Python-only).
   - Every episode reset crosses the CPU/GPU boundary.
   JAX-as-vectorized-env-only pays JAX's costs (functional paradigm, padding,
   debugging difficulty) without its main benefit.

3. **Complete code duplication.** Every `crowdrl-core` module was re-implemented
   in JAX (`action.py`, `collision.py`, `sensing.py`, `observation.py`,
   `reward.py`, `walls.py` — 15 files). Future changes to observation space,
   action dims, or reward terms must be made in two places and verified for
   numerical equivalence. This directly violates design principle #1: "One
   observation builder, used everywhere."

4. **Debugging cost.** JAX's JIT-compiled, functional model means no
   step-through debugging inside `env_step`, `jax.debug.print` breaks
   compilation, shape errors surface as cryptic XLA failures, and NaN
   propagation in JIT is hard to trace. For a research project where reward,
   observations, and physics are still evolving, this is a real cost.

5. **Dependency weight.** `jax[cuda12]` + `jaxlib` + matching `cudnn`/`cublas`
   versions alongside PyTorch's own CUDA stack creates version coupling
   nightmares and bloats the environment.

### Why PyTorch vectorized env is better

A PyTorch-native vectorized environment provides the **same GPU acceleration**
(raycasting, collision, KNN as GPU tensor ops) while eliminating every JAX-
specific problem:

- **No DLPack transfers** — observations are already `torch.Tensor` on the
  training device.
- **No second GPU memory allocator** — one CUDA context, one memory pool.
- **One dependency stack** — PyTorch is already required for training.
- **`torch.vmap`** (PyTorch 2.0+) provides the same vectorisation primitive.
- **Standard debugging** — breakpoints, print statements, PyTorch profiler all
  work normally.
- **Same porting effort** — the functional rewrite is nearly identical, just
  `jnp` → `torch` with minor API differences.

The original motivation (GPU-accelerate the env hot loop, eliminate subprocess
IPC) is fully addressed. The expected 5–50× throughput improvement still applies.

---

## Motivation

The CrowdRL training loop is **CPU-bound**. With 100 subprocess workers, 32 CPU cores are saturated while the RTX 4090 sits idle. The neural network (54K-param MLP) completes forward/backward passes in microseconds; the environment `step()` takes milliseconds per worker. Moving the environment hot loop to GPU-accelerated tensor operations eliminates:

- **Per-step CPU bottleneck**: raycasting, collision, KNN all run as GPU kernels
- **Subprocess IPC overhead**: no more pipe serialisation between 100 workers
- **CPU→GPU→CPU transfers**: observations and actions stay on-device
- **Process management complexity**: `SubprocVecEnv` replaced by `jax.vmap`

Expected throughput improvement: **5–50×** depending on env count and agent density.

---

## Architecture Overview

```
                    ┌─────────────────────────────────┐
                    │          CPU (per reset)         │
                    │  Shapely geometry generation     │
                    │  Delaunay triangulation          │
                    │  NavMesh construction            │
                    │  Agent spawning + solvability    │
                    │  → wall_segments, positions,     │
                    │    goals, body dims arrays       │
                    └──────────────┬──────────────────┘
                                   │ jnp.array transfer
                    ┌──────────────▼──────────────────┐
                    │         GPU (per step)           │
                    │  ┌─────────────────────────┐    │
                    │  │  jax.vmap(env_step)      │    │
                    │  │  over N_ENVS             │    │
                    │  │                          │    │
                    │  │  1. interpret_actions     │    │
                    │  │  2. collision detection   │    │
                    │  │  3. contact forces        │    │
                    │  │  4. physics integration   │    │
                    │  │  5. wall enforcement      │    │
                    │  │  6. raycasting            │    │
                    │  │  7. KNN social sensing    │    │
                    │  │  8. observation assembly  │    │
                    │  │  9. reward computation    │    │
                    │  └─────────────────────────┘    │
                    │           │                      │
                    │   DLPack zero-copy transfer      │
                    │           │                      │
                    │  ┌────────▼────────────────┐    │
                    │  │  PyTorch Actor-Critic    │    │
                    │  │  forward pass (existing) │    │
                    │  └────────┬────────────────┘    │
                    │           │                      │
                    │   DLPack zero-copy transfer      │
                    │           ▼                      │
                    │   actions → next env_step        │
                    └─────────────────────────────────┘
```

### Package structure

```
packages/
  crowdrl-core/       # UNCHANGED — NumPy reference implementations
  crowdrl-env/        # UNCHANGED — CPU Gymnasium env for debugging/viz
  crowdrl-jax/        # NEW
    pyproject.toml
    src/crowdrl_jax/
      __init__.py
      types.py          # JaxWorldState pytree, StaticEnvConfig
      geometry_repr.py  # Shapely polygon → padded segment arrays
      action.py         # interpret_actions (JAX)
      collision.py      # ellipse overlap + contact forces (JAX)
      walls.py          # winding-number containment + projection (JAX)
      sensing.py        # cast_rays + knn_social (JAX)
      observation.py    # build_observations (JAX)
      reward.py         # compute_rewards (JAX)
      step.py           # env_step pure function + jax.vmap
      batched_env.py    # BatchedJaxEnv: manages N envs, async CPU resets
      interop.py        # DLPack JAX↔PyTorch transfers
  crowdrl-train/      # MODIFIED — new JaxRolloutCollector
```

Dependencies: `crowdrl-jax` depends on `crowdrl-core` (for geometry generation at reset time) and `jax[cuda12]`.

---

## GPU State Representation

### JaxWorldState

Replaces `WorldState`. All fields are JAX arrays with fixed `MAX_AGENTS` / `MAX_SEGMENTS` padding.

```python
@jax.tree_util.register_pytree_node_class
class JaxWorldState:
    # Agent state — (MAX_AGENTS,) or (MAX_AGENTS, 2), padded
    positions:          jax.Array  # (MAX_AGENTS, 2)
    velocities:         jax.Array  # (MAX_AGENTS, 2)
    torso_orientations: jax.Array  # (MAX_AGENTS,)
    head_orientations:  jax.Array  # (MAX_AGENTS,)
    shoulder_widths:    jax.Array  # (MAX_AGENTS,)
    chest_depths:       jax.Array  # (MAX_AGENTS,)
    goal_positions:     jax.Array  # (MAX_AGENTS, 2)
    preferred_speeds:   jax.Array  # (MAX_AGENTS,)
    active_mask:        jax.Array  # (MAX_AGENTS,) bool — False for padding AND terminated agents

    # Geometry — static within an episode
    wall_segments:      jax.Array  # (MAX_SEGMENTS, 2, 2)
    n_segments:         jax.Array  # scalar int32

    # Reward temporal state
    prev_velocities:       jax.Array  # (MAX_AGENTS, 2)
    prev_goal_distances:   jax.Array  # (MAX_AGENTS,)
    prev_accelerations:    jax.Array  # (MAX_AGENTS, 2)
    prev_headings:         jax.Array  # (MAX_AGENTS,)
    prev_heading_changes:  jax.Array  # (MAX_AGENTS,)

    # Bookkeeping
    n_agents:               jax.Array  # scalar int32 (real count ≤ MAX_AGENTS)
    step_count:             jax.Array  # scalar int32
    cumulative_terminated:  jax.Array  # (MAX_AGENTS,) bool
```

### Padding strategy

`MAX_AGENTS = 64` covers curriculum range (5–50 agents). `MAX_SEGMENTS = 128` covers Tier 0–2 geometries. Padding slots have `active_mask = False` and are zeroed — they participate in tensor ops but contribute nothing to physics, rewards, or observations via masking.

With `MAX_AGENTS = 64`: pairwise collision tensor is `(64, 64, 2)` = 32 KB. Raycasting tensor is `(64, 16, 128)` for wall intersections = 512 KB. Both trivial for GPU memory.

### Geometry representation for GPU

Raycasting already operates on `wall_segments: (S, 2, 2)` — no Shapely involved during stepping. The only Shapely dependency in the step loop is `enforce_wall_boundaries` which calls `polygon.contains()`. This is replaced by a **winding-number point-in-polygon test** against the same segment array:

```python
def point_in_polygon(points, segments, n_segments):
    """Ray-crossing test: for each point, count crossings with all segments.
    points: (N, 2), segments: (MAX_SEG, 2, 2), n_segments: int
    Returns: (N,) bool — True if inside polygon.
    """
    # Vectorised over points and segments simultaneously
    # Standard even-odd rule, O(N × S), fully parallelisable
```

---

## Component Porting Details

### 1. Action interpretation

**Source**: `crowdrl_core.action.interpret_actions_batch`
**Complexity**: Low — pure vectorised math (`clip`, `cos`, `sin`, column stacking)
**JAX translation**: Replace `np` → `jnp`. Return updated state arrays instead of `BatchActionResult` dataclass.

### 2. Collision detection

**Source**: `crowdrl_core.collision.detect_collisions`
**Current approach**: Pairwise ellipse overlap with `(M, M)` distance matrix, upper-triangle extraction.
**JAX approach**: Compute full `(MAX_AGENTS, MAX_AGENTS)` pairwise overlap matrix. Mask padded/inactive agents. No need for sparse pair extraction — the dense matrix is only `64 × 64 = 4096` entries.

### 3. Contact forces

**Source**: `crowdrl_core.collision.compute_contact_forces`
**Current issue**: Uses `np.add.at` for scatter-accumulation of forces per agent.
**JAX approach**: Compute `(N, N, 2)` pairwise force tensor directly. Sum along axis 1 to get per-agent net force. With `N = 64`, the `(64, 64, 2)` tensor is 32 KB — negligible. Avoids scatter entirely.

```python
# Pairwise displacement: (N, N, 2)
delta = positions[:, None, :] - positions[None, :, :]
# Pairwise forces: (N, N, 2) — includes overlap check, spring-damper
pairwise_forces = compute_pairwise_forces(delta, velocities, body_dims, active_mask)
# Per-agent net force: (N, 2)
net_forces = jnp.sum(pairwise_forces, axis=1)
```

### 4. Wall enforcement

**Source**: `crowdrl_core.collision.enforce_wall_boundaries`
**Current**: Shapely `polygon.contains()` + `nearest_points()` fallback
**JAX approach**:
1. **Containment test**: Ray-crossing (even-odd) against segment array — `(N_agents, N_segments)` broadcast
2. **Projection**: Vectorised nearest-point-on-segment (already exists as `_points_to_segments_nearest_batch`) — direct JAX translation
3. **Wall repulsion forces**: Point-to-segment distance with exponential decay — already vectorised NumPy

### 5. Raycasting (biggest single speedup)

**Source**: `crowdrl_core.sensing.cast_rays_batch`
**Current**: Vectorised NumPy with shapes `(M, R, W)` for wall intersections, `(M, R, T)` for agent intersections.
**JAX translation**: Near-direct. Key shapes:
- Wall rays: `(MAX_AGENTS, N_RAYS, MAX_SEGMENTS)` = `(64, 16, 128)` = 128K intersections
- Agent rays: `(MAX_AGENTS, N_RAYS, MAX_AGENTS)` = `(64, 16, 64)` = 65K intersections
- All trivially parallelisable on GPU

The ray-segment intersection math is identical:
```python
# For each (agent, ray, segment): parametric intersection test
# t_ray = cross(seg_dir, agent_to_seg_start) / cross(ray_dir, seg_dir)
# t_seg = cross(ray_dir, agent_to_seg_start) / cross(ray_dir, seg_dir)
# valid = (0 ≤ t_ray ≤ max_range) & (0 ≤ t_seg ≤ 1)
# distance = min valid t_ray per (agent, ray)
```

### 6. KNN social sensing

**Source**: `crowdrl_core.sensing.knn_social_batch`
**Current**: Pairwise distance matrix `(M, N)`, argsort for K nearest, fancy indexing to gather.
**JAX approach**: `(MAX_AGENTS, MAX_AGENTS)` distance matrix, `jax.lax.top_k` (negated for bottom-K), `jnp.take` for gathering. Mask padded agents with `jnp.inf` distance.

### 7. Observation assembly

**Source**: `crowdrl_core.observation.build_observations_batch`
**JAX approach**: Concatenate ego state (7D) + KNN social (K×7 = 56D) + raycasts (16D) into `(MAX_AGENTS, obs_dim)`. All inputs already computed as JAX arrays.

### 8. Reward computation

**Source**: `crowdrl_env.reward.compute_rewards`
**Complexity**: Low — pure array math with masks. Direct `np` → `jnp` translation. Temporal state (`prev_velocities`, etc.) is part of `JaxWorldState` instead of mutable `RewardState`.

---

## The Core Step Function

```python
@functools.partial(jax.jit, static_argnums=(2,))
def env_step(
    state: JaxWorldState,
    actions: jax.Array,        # (MAX_AGENTS, 4)
    config: StaticEnvConfig,   # static — shapes, dt, reward weights, etc.
) -> tuple[JaxWorldState, jax.Array, jax.Array, jax.Array, dict]:
    """Pure functional environment step. No side effects.

    Returns: (new_state, observations, rewards, dones, info)
    """
    # 1. Interpret actions → desired velocities, orientation changes
    # 2. Velocity blending (exponential smoothing toward desired)
    # 3. Collision detection (pairwise ellipse overlap)
    # 4. Contact forces (pairwise tensor, sum)
    # 5. Wall repulsion forces
    # 6. Physics integration (Euler step with forces)
    # 7. Wall boundary enforcement (winding number + projection)
    # 8. Reward computation (goal check, collisions, progress, smoothness)
    # 9. Update active_mask (terminated agents)
    # 10. Build observations (ego + KNN + raycasts)
    # All operations masked by active_mask + n_agents


# Batch across environments
batched_step = jax.vmap(env_step, in_axes=(0, 0, None))
# batched_step(all_states, all_actions, config) → processes N_ENVS in one kernel
```

---

## Training Integration (Phase 1: JAX Env + PyTorch Training)

### DLPack zero-copy transfer

Both JAX and PyTorch arrays live on the same GPU. DLPack provides zero-copy views:

```python
# JAX → PyTorch (observations)
obs_jax = ...  # (N_ENVS, MAX_AGENTS, obs_dim) on GPU
obs_torch = torch.from_dlpack(jax.dlpack.to_dlpack(obs_jax))

# PyTorch → JAX (actions)
actions_torch = ...  # from actor-critic forward pass
actions_jax = jnp.from_dlpack(torch.to_dlpack(actions_torch))
```

No data copy, no CPU roundtrip. The existing PyTorch MAPPO code (`networks.py`, `mappo.py`, `buffer.py`) is **unchanged**.

### BatchedJaxEnv

Replaces `SubprocVecEnv` entirely:

```python
class BatchedJaxEnv:
    """Manages N_ENVS environments on GPU with async CPU resets."""

    def __init__(self, n_envs, max_agents, max_segments, env_config):
        self.n_envs = n_envs
        self.max_agents = max_agents
        self.max_segments = max_segments
        self.env_config = env_config

        # Thread pool for CPU-side geometry generation
        self.reset_pool = ThreadPoolExecutor(max_workers=8)

        # Pre-generate initial episodes on CPU, transfer to GPU
        self.states = self._initial_reset_all()

    def step(self, actions: jax.Array) -> tuple[...]:
        """Step all envs in one vmapped GPU kernel."""
        self.states, obs, rewards, dones, infos = batched_step(
            self.states, actions, self.config
        )

        # Check which envs have all agents done
        episode_over = self._check_episode_over(self.states)

        # Async CPU reset for finished envs
        for env_idx in jnp.where(episode_over)[0]:
            self._initiate_async_reset(int(env_idx))

        # Collect completed resets (non-blocking)
        self._apply_completed_resets()

        return obs, rewards, dones, infos

    def _generate_episode_cpu(self, seed):
        """CPU-side: Shapely geometry + spawning → arrays."""
        # Uses existing crowdrl-env geometry/spawn code
        # Returns: wall_segments, positions, goals, body_dims, ...
        ...

    def _apply_reset(self, env_idx, reset_data):
        """Write CPU-generated arrays into GPU state for one env."""
        # state = state.at[env_idx].set(...)
        ...
```

### JaxRolloutCollector

Replaces `RolloutCollector`. The collect loop becomes:

```python
def collect(self, n_agent_steps):
    steps = 0
    while steps < n_agent_steps:
        # 1. Reshape obs: (N_ENVS, MAX_AGENTS, obs_dim) → (N_ENVS * MAX_AGENTS, obs_dim)
        obs_flat = obs.reshape(-1, obs_dim)

        # 2. DLPack → PyTorch, forward pass
        obs_torch = torch.from_dlpack(jax.dlpack.to_dlpack(obs_flat))
        with torch.no_grad():
            actions_torch, log_probs, values = actor_critic(obs_torch)

        # 3. DLPack → JAX, reshape back
        actions_jax = jnp.from_dlpack(torch.to_dlpack(actions_torch))
        actions_jax = actions_jax.reshape(n_envs, max_agents, action_dim)

        # 4. Batched GPU step — ALL envs in one kernel
        obs, rewards, dones, infos = batched_env.step(actions_jax)

        # 5. Store transitions, count steps
        steps += int(active_agents.sum())
```

**No subprocess pipes. No CPU env stepping. No IPC serialisation.** The entire collect loop stays on GPU except for the occasional reset.

---

## Reset Pipeline

Resets are the CPU/GPU boundary. Design for pipelining:

1. **Episode length**: ~200 steps at 5–50 agents → episodes last ~200 steps
2. **Reset frequency**: With 1000 envs, ~5 resets/step on average
3. **Reset cost**: Geometry generation + spawning takes ~5–20ms on CPU
4. **Pipelining**: 8 CPU threads pre-generate episodes while GPU steps. With 200-step episodes and 8 threads, there is ample time to keep ahead.

For envs awaiting reset, the GPU simply skips them (mask out in `jax.vmap`). When the CPU result arrives, the state slice is updated via `state.at[env_idx].set(...)`.

---

## NavMesh Signals

Disabled for Phase 1 (`use_navmesh = False`, already the default). The 79D observation space works without navmesh (7 ego + 56 social + 16 rays).

Phase 2 option: precompute full waypoint paths at reset time on CPU, store as `(MAX_AGENTS, MAX_WAYPOINTS, 2)` array on GPU. Per step, find nearest waypoint via vectorised distance — no graph search on GPU.

---

## Constants and Tuning

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `MAX_AGENTS` | 64 | Covers curriculum (5–50), allows headroom |
| `MAX_SEGMENTS` | 128 | Covers Tier 0–2 geometries |
| `N_RAYS` | 16 | Matches current `ObsConfig` default |
| `K_NEAREST` | 8 | Matches current `ObsConfig` default |
| `N_ENVS` | 1024+ | GPU can handle thousands; tune for throughput |
| Reset threads | 8 | Keep ahead of episode completion rate |

Memory per env: ~150 KB (all state arrays). 1024 envs ≈ 150 MB — negligible on a 24 GB GPU.

---

## Implementation Phases

### Phase 1: JAX env core + training integration

1. Create `crowdrl-jax` package with `pyproject.toml`, dependencies
2. Define `JaxWorldState` pytree and `StaticEnvConfig`
3. Implement `geometry_repr.py` — Shapely polygon → padded segment/position arrays
4. Port `action.py` — `interpret_actions` in JAX (simplest, validates pattern)
5. Port `collision.py` — pairwise ellipse overlap + `(N,N,2)` contact forces
6. Port `walls.py` — winding-number containment + nearest-point projection
7. Port `sensing.py` — `cast_rays` + `knn_social` in JAX (biggest speedup)
8. Port `observation.py` — assemble obs vector from JAX arrays
9. Port `reward.py` — reward computation in JAX
10. Wire up `step.py` — `env_step` pure function + `jax.vmap`
11. Build `batched_env.py` — `BatchedJaxEnv` with async CPU reset pool
12. Build `interop.py` — DLPack transfer utilities
13. New `JaxRolloutCollector` in `crowdrl-train`
14. Numerical equivalence tests against CPU `CrowdEnv`

### Phase 2: Optimisation

- Profile JAX kernels, tune `MAX_AGENTS` / `MAX_SEGMENTS`
- Experiment with `N_ENVS` = 1024, 2048, 4096
- Add precomputed navmesh waypoint signals
- Benchmark wall-clock speedup vs CPU baseline

### Phase 3 (optional): Pure JAX training

- Port Actor-Critic to Flax/Equinox
- Port PPO update to JAX + optax
- End-to-end `jax.jit` of collect + update (PureJaxRL style)
- Eliminates all DLPack transfers

---

## Testing Strategy

### Numerical equivalence

Run identical episodes (same seed, geometry, actions) through both CPU (`CrowdEnv.step`) and JAX (`env_step`). Verify:

- Observations match to `atol=1e-5`
- Rewards match to `atol=1e-6`
- Collision detection agrees on all pairs
- Raycasts produce same distances
- Wall enforcement keeps agents inside polygon

### Component-level tests

Port each function independently. For each:
1. Generate random inputs
2. Run NumPy reference implementation
3. Run JAX implementation
4. Assert `jnp.allclose`

### Training convergence

Run both CPU and JAX training for same number of agent-steps. Verify similar learning curves (goal rate, reward). Exact match not expected (float32 vs float64, different RNG streams) but convergence characteristics should be comparable.

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| JAX recompilation on shape changes | Fixed padding with `MAX_AGENTS`/`MAX_SEGMENTS` — shapes never change |
| Winding-number accuracy for complex polygons with holes | Test against Shapely `contains()` on all Tier 0–2 geometries |
| DLPack overhead | Zero-copy on same GPU; benchmark shows <0.1ms per transfer |
| CPU reset becoming bottleneck at high env counts | Thread pool + pipelining; monitor with profiling |
| Float32 (JAX default) vs float64 (NumPy default) drift | Acceptable for RL; verify convergence empirically |
| Variable agent counts across envs | Padding + masking — standard approach, used by JaxMARL/Brax |

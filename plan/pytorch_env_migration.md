# PyTorch Vectorized Environment — Migration Status

**Branch**: `feature/jax-env`
**Date**: 2026-03-27

---

## Summary

Replaced the JAX GPU-accelerated environment (`crowdrl-jax`) with a
PyTorch-native implementation (`crowdrl-torch`). The motivation is documented
in [plan/jax_environment_migration.md](jax_environment_migration.md#decision-jax--pytorch).

## What changed

### New: `packages/crowdrl-torch/` (15 source files)

Complete PyTorch port of the JAX vectorized environment. All env stepping
(raycasting, collision detection, KNN social sensing, physics integration,
reward computation) runs as GPU tensor operations.

| Module | Lines | What it does |
|--------|-------|--------------|
| `types.py` | 130 | `TorchWorldState` dataclass, `EnvConfig` NamedTuple, `make_initial_state()` |
| `action.py` | 75 | 4D policy output → desired velocity + orientation changes |
| `walls.py` | 175 | Ray-crossing point-in-polygon, nearest-point-on-segment, wall enforcement |
| `collision.py` | 175 | Dense (E,N,N) pairwise ellipse overlap + spring-damper contact forces |
| `sensing.py` | 300 | Ray-wall (E,N,R,S) + ray-ellipse (E,N,R,N) intersections, K-nearest social |
| `observation.py` | 80 | Ego(7) + social(56) + rays(16) = 79D observation assembly |
| `reward.py` | 60 | Goal bonus, collision penalty, progress shaping |
| `step.py` | 140 | `batched_step()` — composes all modules, `torch.compile`-ready |
| `batched_env.py` | 200 | `BatchedTorchEnv` — N_ENVS management, async CPU resets via ThreadPoolExecutor |
| `geometry_repr.py` | 110 | `prepare_reset_data()` — pads CPU arrays to fixed MAX_AGENTS/MAX_SEGMENTS |
| `episode_factory.py` | 140 | `make_episode_factory()` — wraps crowdrl-env geometry/spawn for CPU reset |
| `torch_collector.py` | 250 | `TorchRolloutCollector` — rollout collection, no DLPack transfers |

### Modified

- **`pyproject.toml` (root)** — Added `packages/crowdrl-torch` to workspace members and sources.
- **`plan/jax_environment_migration.md`** — Added decision record explaining why JAX was replaced.

### Not yet modified

- **`crowdrl-train`** — No changes yet. The existing `SubprocVecEnv` +
  `RolloutCollector` path in `06_full_training.ipynb` continues to work
  unchanged. The new `BatchedTorchEnv` + `TorchRolloutCollector` is an
  alternative backend, not a replacement of the CPU path.

## Design decisions

1. **Manual batching** — All tensors carry `(E, N, ...)` shape (E=envs, N=max_agents).
   No `torch.vmap`. The JAX code was already vectorized over agents; we added the
   env batch dimension explicitly.

2. **`torch.compile` ready** — `batched_step()` is a pure function of tensors.
   Apply `@torch.compile(mode="reduce-overhead")` once validated for CUDA graph
   support. Not enabled yet to keep debugging simple.

3. **Same padding strategy** — MAX_AGENTS=64, MAX_SEGMENTS=128 with `active_mask`.
   Dense pairwise tensors (64×64) are negligible on GPU.

4. **No DLPack** — Observations and actions are `torch.Tensor` end-to-end.
   The JAX collector required JAX→numpy→torch→numpy→JAX per step. The PyTorch
   collector only transfers to CPU for per-env `RolloutBuffer` storage (numpy GAE).

5. **CPU-only code reused verbatim** — `geometry_repr.py` and `episode_factory.py`
   are pure numpy, copied from JAX. They run in the thread pool at reset time.

## Equivalence tests

8 tests verify numerical equivalence against `crowdrl-core` reference
implementations (atol=1e-4, rtol=1e-3 for float32 vs float64):

| Test | Status |
|------|--------|
| Action interpretation | ✅ |
| Collision detection | ✅ |
| Point-in-polygon | ✅ |
| Nearest-point-on-segment | ✅ |
| Raycasting (wall + agent ellipse) | ✅ |
| KNN social sensing | ✅ |
| Full observation builder | ✅ |
| Reward computation | ✅ |

Run: `uv run pytest packages/crowdrl-torch/tests/test_equivalence.py -v`

## What's next

1. **Integration test** — Wire `BatchedTorchEnv` + `TorchRolloutCollector` into
   a training loop (notebook 07 or a new `_train_torch_batched` path in `train.py`).
2. **Benchmark** — Compare wall-clock throughput: CPU `SubprocVecEnv` (100 workers)
   vs `BatchedTorchEnv` (256–1024 envs on GPU).
3. **`torch.compile`** — Enable and verify no recompilations after warmup.
4. **Remove `crowdrl-jax`** — Once the PyTorch path is validated end-to-end.

## Compatibility with existing training

The existing training path (`06_full_training.ipynb`) is **unaffected**:
- It uses `SubprocVecEnv` + `RolloutCollector` from `crowdrl-train`
- No imports from `crowdrl-torch` or `crowdrl-jax`
- The `crowdrl-torch` package is an additional workspace member, not a replacement
- All `crowdrl-env` tests continue to pass (86/86)

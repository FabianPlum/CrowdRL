# Why the GPU environment is PyTorch and not JAX

> **Decision record. Closed 2026-03-27, and it has held.**
>
> The GPU-accelerated environment was originally planned in JAX; `packages/crowdrl-jax/`
> existed and reimplemented 15 `crowdrl-core` modules. After evaluation we pivoted to a
> **PyTorch-native vectorized environment** (`crowdrl-torch`) and deleted the JAX package.
> That environment now delivers >100k agent-steps/sec and is the real training path, so the
> decision is settled -- this file exists to answer "why not JAX?" without re-running the
> evaluation.
>
> *Trimmed 2026-08-03:* this file previously carried ~450 further lines of the abandoned
> JAX porting plan (`JaxWorldState` pytrees, per-module port notes, `BatchedJaxEnv`
> sketches, a three-phase rollout). None of it was ever built and all of it described an
> API we do not use. It is recoverable from git history if the question ever genuinely
> reopens.

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

# DDP Single-Node Multi-GPU Training Plan

**Pattern**: DD-PPO (Decentralized Distributed PPO)
**Scope**: Single node, multiple GPUs. Multi-node is out of scope.
**Reference**: Wijmans et al. (2019) "DD-PPO" (ICLR 2020), arXiv:1911.00357

---

## Architecture

Each GPU rank independently:

1. Runs its own `BatchedTorchEnv` on `cuda:{local_rank}` with `n_envs` environments
2. Collects rollouts via its own `TorchRolloutCollector`
3. Computes PPO loss on its own local rollout data

Gradients are averaged across ranks via explicit all-reduce after `backward()`.
This makes the effective batch = `local_batch * world_size`.

```
  Rank 0 (cuda:0)             Rank 1 (cuda:1)            Rank K (cuda:K)
  +------------------+        +------------------+        +------------------+
  | BatchedTorchEnv  |        | BatchedTorchEnv  |        | BatchedTorchEnv  |
  | (64 envs)        |        | (64 envs)        |  ...   | (64 envs)        |
  +--------+---------+        +--------+---------+        +--------+---------+
           |                           |                           |
  +--------v---------+        +--------v---------+        +--------v---------+
  | TorchRollout     |        | TorchRollout     |        | TorchRollout     |
  | Collector        |        | Collector        |        | Collector        |
  | (1.28M steps)    |        | (1.28M steps)    |        | (1.28M steps)    |
  +--------+---------+        +--------+---------+        +--------+---------+
           |                           |                           |
           |          all_reduce(gradients)                        |
           +-------------------+-------+---------------------------+
                               |
                    +----------v-----------+
                    | MAPPOUpdater         |
                    | (identical weights   |
                    |  on every rank)      |
                    +----------------------+
```

Launch: `torchrun --standalone --nproc_per_node=N script.py`

---

## Design Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Env distribution | Linear scaling (same n_envs per rank) | Straightforward, predictable throughput gain |
| 2 | Gradient sync | Manual all-reduce (not DDP wrapper) | Actor uses `evaluate_actions()`, not `forward()` -- DDP hooks wouldn't fire. Manual all-reduce is explicit and what CleanRL uses |
| 3 | Normalizer sync | All-reduce every rollout | Safe default. **TODO**: reduce frequency once early curriculum phases stabilise |
| 4 | Curriculum sync | Rank 0 aggregates, broadcasts | Negligible memory (~32 KB of episode stat dicts per rollout) |
| 5 | Seeding | `base_seed + rank` for envs, identical model init seed | Diverse rollouts, deterministic start |
| 6 | LR scaling | No scaling | Not standard in RL (CleanRL keeps LR fixed). KL early stopping self-regulates |
| 7 | Backend | NCCL | Standard for single-node GPU |

---

## Batch Semantics

Current notebook setup (06_full_training.ipynb):
- `N_ENVS = 64`, `N_STEPS_PER_COLLECT = 64 * 20000 = 1,280,000 agent-steps`

With DDP, each rank collects 1.28M independently:

| GPUs | Local batch | Effective global batch |
|------|-------------|----------------------|
| 1    | 1.28M       | 1.28M                |
| 2    | 1.28M       | 2.56M                |
| 4    | 1.28M       | 5.12M                |

Each rank runs PPO epochs on its own 1.28M. The gradient all-reduce averages across
ranks, equivalent to training on the global batch split into K equal minibatches.
No LR adjustment needed.

---

## Synchronisation Points

| When | What | Mechanism | Cost |
|------|------|-----------|------|
| Every backward() | Actor + Critic gradients | `allreduce_gradients()` -- flatten, all_reduce(SUM), div(world_size) | ~2 all-reduce calls per minibatch |
| Every rollout | Obs normalizer (mean/var/count) | Parallel Welford merge via all-reduce | 3 all-reduce calls |
| Every rollout | Reward normalizer (return var + running return) | Same Welford merge + avg running return | 4 all-reduce calls |
| Every rollout | Episode stats | `all_gather_object()` to rank 0 | 1 gather call, ~KB payload |
| On phase change | Curriculum state | `broadcast_object_list()` from rank 0 | Rare, ~bytes |

---

## Files Changed

### New: `packages/crowdrl-torch/src/crowdrl_torch/distributed.py`

DDP lifecycle and synchronisation helpers:

- `init_distributed(backend="nccl")` -- init process group, return (rank, world_size, device)
- `cleanup_distributed()` -- destroy process group
- `is_distributed()` / `is_main_rank()` / `get_rank()` / `get_world_size()`
- `allreduce_gradients(model)` -- flatten + all_reduce + unflatten gradient sync
- `sync_reward_normalizer(normalizer, device)` -- all-reduce CPU-based reward normalizer stats
- `gather_episode_stats(local_episodes)` -- all_gather_object to rank 0
- `broadcast_curriculum_state(curriculum_manager, src=0)` -- broadcast phase from rank 0

### Modified: `packages/crowdrl-torch/src/crowdrl_torch/normalizer.py`

- Add `TorchRunningNormalizer.sync_across_ranks()` -- parallel Welford merge via all-reduce

### Modified: `packages/crowdrl-train/src/crowdrl_train/mappo.py`

- Add `distributed: bool` flag to `MAPPOUpdater.__init__()`
- Insert `allreduce_gradients()` calls between `backward()` and `clip_grad_norm_()` for both actor and critic

### Modified: `packages/crowdrl-train/src/crowdrl_train/config.py`

- Add `DDPConfig` frozen dataclass (backend field)

### Updated exports

- `crowdrl_torch/__init__.py` -- export distributed module functions
- `crowdrl_train/__init__.py` -- export DDPConfig

### New: `packages/crowdrl-torch/tests/test_distributed.py`

Unit tests (single-process, no GPU required):
- `allreduce_gradients` is a no-op when dist not initialized
- `sync_across_ranks` is a no-op when dist not initialized
- `gather_episode_stats` returns input when dist not initialized
- `broadcast_curriculum_state` is a no-op when dist not initialized
- `init_distributed` / `cleanup_distributed` lifecycle

---

## Usage Pattern (training script)

```python
from crowdrl_torch.distributed import (
    init_distributed, cleanup_distributed, is_main_rank,
    gather_episode_stats, broadcast_curriculum_state,
    sync_reward_normalizer,
)

# --- Init ---
rank, world_size, device = init_distributed()
seed = base_seed + rank  # diverse env rollouts

# --- Create per-rank components ---
env = BatchedTorchEnv(n_envs=N_ENVS, ..., device=device, seed=seed)
actor_critic = ActorCritic(config.network).to(device)
updater = MAPPOUpdater(actor_critic, config.ppo, device)  # auto-detects distributed

obs_normalizer = TorchRunningNormalizer((obs_dim,), device=device)
collector = TorchRolloutCollector(env, actor_critic, obs_normalizer, ...)

# --- Training loop ---
for rollout in range(N_ROLLOUTS):
    # 1. Collect (independent per rank)
    local_episodes = collector.collect(n_agent_steps)
    batch = collector.compute_gae_and_flatten(gamma, gae_lambda)

    # 2. PPO update (gradients synced inside updater.update())
    metrics = updater.update(batch)

    # 3. Sync normalizers
    obs_normalizer.sync_across_ranks()
    if reward_normalizer:
        sync_reward_normalizer(reward_normalizer, device)

    # 4. Curriculum (rank 0 decides, broadcasts)
    all_episodes = gather_episode_stats(local_episodes)
    if is_main_rank():
        for ep in all_episodes:
            curriculum.report_episode(EpisodeStats(...))
    broadcast_curriculum_state(curriculum)

    # 5. Logging/checkpointing (rank 0 only)
    if is_main_rank():
        logger.log_metrics(...)
        if rollout % checkpoint_interval == 0:
            save_checkpoint(...)

cleanup_distributed()
```

---

## What This Does NOT Change

- No changes to BatchedTorchEnv, step.py, observation.py, sensing.py, collision.py
- No changes to RolloutBuffer, FlatBatch, ActorCritic, Actor, Critic networks
- No changes to the CPU-based SubprocVecEnv training path (crowdrl-train)
- No multi-node support (would need different launcher + network config)
- No LR scaling or batch size adjustments

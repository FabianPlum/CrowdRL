"""Single-node multi-GPU training via PyTorch DDP (DD-PPO pattern).

Each GPU rank independently:

1. Runs its own ``BatchedTorchEnv`` with *n_envs* environments
2. Collects rollouts via ``TorchRolloutCollector``
3. Computes PPO loss on its own local rollout data

Gradients are averaged across ranks via explicit ``all_reduce`` after
``backward()``, making the effective batch = local_batch * world_size.

Launch::

    torchrun --standalone --nproc_per_node=N script.py

References
----------
- Wijmans et al. (2019) "DD-PPO" (ICLR 2020), arXiv:1911.00357
- CleanRL ``ppo_atari_multigpu.py``
- PyTorch DDP docs: https://docs.pytorch.org/docs/stable/notes/ddp.html
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn

from crowdrl_train.normalizer import RewardNormalizer


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def init_distributed(backend: str = "nccl") -> tuple[int, int, torch.device]:
    """Initialise the distributed process group for single-node DDP.

    Reads ``RANK``, ``LOCAL_RANK``, and ``WORLD_SIZE`` environment
    variables set by ``torchrun``.

    Returns
    -------
    rank : int
    world_size : int
    device : torch.device
        ``cuda:{local_rank}`` when CUDA is available, else ``cpu``.
    """
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


def cleanup_distributed() -> None:
    """Destroy the process group (safe to call even if not initialised)."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Rank queries
# ---------------------------------------------------------------------------


def is_distributed() -> bool:
    """Return ``True`` when a distributed process group is active."""
    return dist.is_available() and dist.is_initialized()


def is_main_rank() -> bool:
    """Return ``True`` on rank 0 (or when not running distributed)."""
    return not is_distributed() or dist.get_rank() == 0


def get_rank() -> int:
    """Current rank (0 when not distributed)."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """World size (1 when not distributed)."""
    return dist.get_world_size() if is_distributed() else 1


# ---------------------------------------------------------------------------
# Gradient synchronisation
# ---------------------------------------------------------------------------


def allreduce_gradients(model: nn.Module) -> None:
    """Average gradients across all DDP ranks.

    Flattens every ``.grad`` tensor into a single contiguous buffer so
    that only **one** ``all_reduce`` call is issued (much faster than
    per-parameter all-reduce).

    No-op when distributed is not active.
    """
    if not is_distributed():
        return

    grads: list[Tensor] = []
    params_with_grad: list[nn.Parameter] = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.flatten())
            params_with_grad.append(p)

    if not grads:
        return

    flat = torch.cat(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(dist.get_world_size())

    offset = 0
    for p in params_with_grad:
        numel = p.grad.numel()
        p.grad.copy_(flat[offset : offset + numel].view_as(p.grad))
        offset += numel


# ---------------------------------------------------------------------------
# Normaliser synchronisation
# ---------------------------------------------------------------------------


def sync_reward_normalizer(
    reward_normalizer: RewardNormalizer,
    device: torch.device,
) -> None:
    """Synchronise a CPU-based ``RewardNormalizer`` across ranks.

    Merges the internal ``RunningNormalizer`` (return-variance tracker)
    using the parallel Welford formula and averages the running-return
    estimate.

    Parameters
    ----------
    reward_normalizer : RewardNormalizer
        The normalizer to synchronise (modified in-place).
    device : torch.device
        GPU device for the temporary all-reduce tensors.
    """
    if not is_distributed():
        return

    rn = reward_normalizer._return_var  # RunningNormalizer (CPU, numpy)

    # Convert to GPU tensors
    mean_t = torch.tensor(rn.mean, dtype=torch.float64, device=device)
    var_t = torch.tensor(rn.var, dtype=torch.float64, device=device)
    count_t = torch.tensor([rn.count], dtype=torch.float64, device=device)
    running_ret = torch.tensor(
        [reward_normalizer._running_return], dtype=torch.float64, device=device
    )

    # --- Parallel Welford merge ---
    total_count = count_t.clone()
    dist.all_reduce(total_count, op=dist.ReduceOp.SUM)

    if total_count.item() < 1e-3:
        return

    # Weighted mean: sum(count_i * mean_i) / total_count
    weighted_mean = mean_t * count_t
    dist.all_reduce(weighted_mean, op=dist.ReduceOp.SUM)
    new_mean = weighted_mean / total_count

    # Parallel variance
    delta = mean_t - new_mean
    weighted_var = count_t * (var_t + delta**2)
    dist.all_reduce(weighted_var, op=dist.ReduceOp.SUM)
    new_var = weighted_var / total_count

    # Average running return
    dist.all_reduce(running_ret, op=dist.ReduceOp.SUM)
    running_ret.div_(dist.get_world_size())

    # Write back to CPU numpy arrays
    rn.mean = new_mean.cpu().numpy()
    rn.var = new_var.cpu().numpy()
    rn.count = total_count.item()
    reward_normalizer._running_return = running_ret.item()


# ---------------------------------------------------------------------------
# Episode stats & curriculum synchronisation
# ---------------------------------------------------------------------------


def gather_episode_stats(local_episodes: list[dict]) -> list[dict]:
    """Gather completed episode stats from all ranks to rank 0.

    Returns the combined list on rank 0, empty list on other ranks.
    When not running distributed, returns the input unchanged.
    """
    if not is_distributed():
        return local_episodes

    gathered: list[list[dict] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_episodes)

    if dist.get_rank() == 0:
        combined: list[dict] = []
        for episodes in gathered:
            if episodes is not None:
                combined.extend(episodes)
        return combined
    return []


def broadcast_curriculum_state(curriculum_manager: object, src: int = 0) -> None:
    """Broadcast curriculum phase and counters from *src* rank.

    After rank 0 makes advancement decisions, all ranks must agree on
    the current phase so environments generate consistent geometries.

    Parameters
    ----------
    curriculum_manager
        Any object with ``state_dict()`` / ``load_state_dict()`` methods
        (i.e. ``CurriculumManager``).
    src : int
        Source rank (default 0).
    """
    if not is_distributed():
        return

    state = [curriculum_manager.state_dict()] if dist.get_rank() == src else [None]
    dist.broadcast_object_list(state, src=src)

    if dist.get_rank() != src:
        curriculum_manager.load_state_dict(state[0])


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def distributed_seed(base_seed: int) -> int:
    """Return a per-rank seed: ``base_seed + rank``.

    Guarantees each rank collects rollouts from different environment
    configurations while the model starts from identical weights
    (initialised with the un-shifted *base_seed*).
    """
    return base_seed + get_rank()


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

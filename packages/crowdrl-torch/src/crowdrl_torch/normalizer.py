"""GPU-resident running normalizer for observations.

Keeps mean/var as tensors on device, avoiding CPU roundtrips during
the collect loop. The Welford update runs as tensor ops on the GPU.

Compatible with ``crowdrl_train.normalizer.RunningNormalizer`` for
checkpointing: ``state_dict()`` / ``load_state_dict()`` use numpy
arrays with the same keys.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

if TYPE_CHECKING:
    from crowdrl_train.normalizer import RunningNormalizer


# Diagnostic (CROWDRL_NAN_TRIPWIRE=1): catch the EXACT step the running mean/var
# first go non-finite -- in update() or the DDP merge -- with the batch that did
# it (per-feature max|x|, worst row) and whether it was the local update or the
# cross-rank sync. Default OFF.
_NORM_TRIPWIRE = os.environ.get("CROWDRL_NAN_TRIPWIRE", "") == "1"

# Cap on the running sample count -- prevents float64 overflow of m_a = var*count.
# sync_across_ranks() re-sums the merged count every rollout, so without a cap the
# count grows GEOMETRICALLY (~doubles per rollout, since both ranks carry the full
# merged total) and crosses ~1e307 in a few hundred rollouts -> var*count -> inf
# -> var NaN -> every normalized obs NaN -> run death (the deterministic r355
# collapse). The count only sets the Welford update weight (batch/count), so
# capping it just makes the normalizer a stable large-window estimator -- which
# is also the intended near-frozen behaviour once the obs distribution settles.
_MAX_COUNT = 1e8


def _norm_is_poisoned(norm) -> bool:
    """One GPU sync: are the running stats non-finite OR runaway-huge?

    Flags non-finite mean/var, and also a variance that has exploded past 1e12
    (std > 1e6 -- 1e10x a normal obs feature). Catching the runaway BEFORE it
    saturates to inf/nan surfaces the batch that is actually driving it.
    """
    stacked = torch.stack([norm.mean, norm.var])
    return (not bool(torch.isfinite(stacked).all().item())) or bool((norm.var > 1e12).any().item())


def _dump_norm_poison(stage, norm, batch, batch_mean, batch_var, batch_count, **extra):
    """Report + dump + SystemExit when the running stats first go non-finite."""
    bad_mean = (~torch.isfinite(norm.mean)).nonzero(as_tuple=False).flatten().tolist()
    bad_var = (~torch.isfinite(norm.var)).nonzero(as_tuple=False).flatten().tolist()
    lines = [
        "\n" + "@" * 78,
        f"@ NORMALIZER POISONED @ stage='{stage}'",
        f"@ count={norm.count:.6g}  batch_count={batch_count}  extra={extra}",
        f"@ non-finite MEAN cols ({len(bad_mean)}/{norm.mean.numel()}): {bad_mean[:48]}",
        f"@ non-finite VAR  cols ({len(bad_var)}/{norm.var.numel()}): {bad_var[:48]}",
    ]
    dump = {
        "stage": stage,
        "count": float(norm.count),
        "batch_count": batch_count,
        "bad_mean_feats": bad_mean,
        "bad_var_feats": bad_var,
        "mean": norm.mean.detach().cpu(),
        "var": norm.var.detach().cpu(),
        **extra,
    }
    if batch is not None:
        bmax = batch.abs().max(dim=0).values  # (D,)
        worst_feat = int(bmax.argmax())
        worst_row = int(batch[:, worst_feat].abs().argmax())
        top = sorted(range(batch.shape[1]), key=lambda i: -float(bmax[i]))[:8]
        lines.append(f"@ batch max|x|={float(bmax.max()):.5g} at feature {worst_feat}")
        lines.append(
            f"@ batch top-8 features by max|x|: {[(f, round(float(bmax[f]), 3)) for f in top]}"
        )
        dump["batch_max_per_feat"] = bmax.detach().cpu()
        dump["batch_worst_row"] = batch[worst_row].detach().cpu()
        dump["batch_mean"] = None if batch_mean is None else batch_mean.detach().cpu()
        dump["batch_var"] = None if batch_var is None else batch_var.detach().cpu()
    lines.append("@" * 78)
    print("\n".join(lines), flush=True)
    try:
        torch.save(dump, "/tmp/crowdrl_norm_poison.pt")
        print("@ dumped -> /tmp/crowdrl_norm_poison.pt", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"@ (dump failed: {exc})", flush=True)
    raise SystemExit(f"Normalizer stats went non-finite at stage='{stage}'")


class TorchRunningNormalizer:
    """Welford's online algorithm on GPU tensors.

    Mirrors the API of ``crowdrl_train.normalizer.RunningNormalizer``
    but keeps statistics on-device for zero-copy normalization.

    Parameters
    ----------
    shape : tuple[int, ...]
        Feature shape (e.g. ``(obs_dim,)``).
    device : torch.device
        Device for mean/var tensors.
    clip : float
        Clamp normalised values to ``[-clip, clip]``.
    epsilon : float
        Small constant for numerical stability in division.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        device: torch.device | str = "cpu",
        clip: float = 10.0,
        epsilon: float = 1e-8,
    ):
        self.shape = shape
        self.device = torch.device(device)
        self.clip = clip
        self.epsilon = epsilon

        self.mean = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self.var = torch.ones(shape, dtype=torch.float64, device=self.device)
        self.count: float = 1e-4

    def update(self, batch: Tensor | np.ndarray) -> None:
        """Update running statistics with a batch of samples.

        Parameters
        ----------
        batch : (..., *shape) tensor or numpy array — transferred internally.
        """
        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        batch = batch.to(dtype=torch.float64, device=self.device)
        if batch.ndim == len(self.shape):
            batch = batch.unsqueeze(0)
        batch = batch.reshape(-1, *self.shape)

        # Drop any non-finite samples. A single NaN/Inf observation must never
        # permanently poison the running mean/var -- that would NaN every future
        # normalized observation and silently, unrecoverably kill the run. A
        # transient physics glitch should degrade gracefully, not be fatal.
        reduce_dims = tuple(range(1, batch.ndim))
        finite_rows = torch.isfinite(batch).all(dim=reduce_dims)
        if not bool(finite_rows.all()):
            batch = batch[finite_rows]
        if batch.shape[0] == 0:
            return

        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, correction=0)
        batch_count = batch.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

        if _NORM_TRIPWIRE and _norm_is_poisoned(self):
            _dump_norm_poison("update", self, batch, batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: Tensor, batch_var: Tensor, batch_count: int
    ) -> None:
        """Parallel Welford update from pre-computed moments."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = min(total_count, _MAX_COUNT)

    def normalize(self, x: Tensor | np.ndarray) -> Tensor | np.ndarray:
        """Normalize input. Accepts and returns tensors or numpy arrays.

        Parameters
        ----------
        x : (..., *shape) tensor or numpy array.

        Returns
        -------
        Normalised output, same type as input.
        If tensor: float32 on self.device.
        If numpy: float64 array (for compatibility with CPU training code).
        """
        if isinstance(x, np.ndarray):
            # CPU path — compatible with crowdrl_train functions that pass numpy
            mean_np = self.mean.cpu().numpy()
            std_np = np.sqrt(self.var.cpu().numpy() + self.epsilon)
            return np.clip((x - mean_np) / std_np, -self.clip, self.clip)

        x = x.to(device=self.device, dtype=torch.float32)
        mean_f32 = self.mean.float()
        std_f32 = (self.var + self.epsilon).float().sqrt()
        return torch.clamp((x - mean_f32) / std_f32, -self.clip, self.clip)

    def state_dict(self) -> dict:
        """Serialise for checkpointing (numpy arrays, same keys as CPU version)."""
        return {
            "mean": self.mean.cpu().numpy().copy(),
            "var": self.var.cpu().numpy().copy(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from checkpoint."""
        self.mean = torch.tensor(state["mean"], dtype=torch.float64, device=self.device)
        self.var = torch.tensor(state["var"], dtype=torch.float64, device=self.device)
        # Cap a possibly-overflowed count from a checkpoint trained before the cap
        # existed (the runaway-count bug inflated it past 1e300), so the first
        # update's m_a = var*count cannot overflow.
        self.count = min(float(state["count"]), _MAX_COUNT)

    def sync_across_ranks(self) -> None:
        """Merge running statistics across DDP ranks via parallel Welford.

        After calling, each rank holds the combined statistics as if all
        data from every rank had been processed by a single normalizer.

        No-op when ``torch.distributed`` is not initialised.

        Note: called every rollout by default. For more complex curricula
        where early-phase behaviour has stabilised, the caller may reduce
        sync frequency to every K rollouts.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return

        _pre_finite = (not _norm_is_poisoned(self)) if _NORM_TRIPWIRE else True

        local_count = torch.tensor([self.count], dtype=torch.float64, device=self.device)

        total_count = local_count.clone()
        dist.all_reduce(total_count, op=dist.ReduceOp.SUM)

        if total_count.item() < 1e-3:
            return

        # Weighted mean: sum(count_i * mean_i) / total_count
        weighted_mean = self.mean * local_count
        dist.all_reduce(weighted_mean, op=dist.ReduceOp.SUM)
        new_mean = weighted_mean / total_count

        # Parallel variance: sum(count_i * (var_i + (mean_i - new_mean)^2)) / total_count
        delta = self.mean - new_mean
        weighted_var = local_count * (self.var + delta**2)
        dist.all_reduce(weighted_var, op=dist.ReduceOp.SUM)
        new_var = weighted_var / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = min(total_count.item(), _MAX_COUNT)

        if _NORM_TRIPWIRE and _norm_is_poisoned(self):
            _dump_norm_poison(
                "sync",
                self,
                None,
                None,
                None,
                0,
                total_count=float(total_count.item()),
                local_stats_were_finite=_pre_finite,
            )

    @staticmethod
    def from_cpu_normalizer(
        cpu_norm: RunningNormalizer, device: torch.device | str = "cpu"
    ) -> "TorchRunningNormalizer":
        """Create from an existing CPU RunningNormalizer."""
        tn = TorchRunningNormalizer(
            shape=cpu_norm.mean.shape,
            device=device,
            clip=cpu_norm.clip,
            epsilon=cpu_norm.epsilon,
        )
        tn.load_state_dict(cpu_norm.state_dict())
        return tn

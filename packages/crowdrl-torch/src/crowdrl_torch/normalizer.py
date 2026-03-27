"""GPU-resident running normalizer for observations.

Keeps mean/var as tensors on device, avoiding CPU roundtrips during
the collect loop. The Welford update runs as tensor ops on the GPU.

Compatible with ``crowdrl_train.normalizer.RunningNormalizer`` for
checkpointing: ``state_dict()`` / ``load_state_dict()`` use numpy
arrays with the same keys.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


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

        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, correction=0)
        batch_count = batch.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

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
        self.count = total_count

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
        self.count = state["count"]

    @staticmethod
    def from_cpu_normalizer(
        cpu_norm: "RunningNormalizer", device: torch.device | str = "cpu"
    ) -> "TorchRunningNormalizer":
        """Create from an existing CPU RunningNormalizer."""
        tn = TorchRunningNormalizer(
            shape=cpu_norm.mean.shape, device=device,
            clip=cpu_norm.clip, epsilon=cpu_norm.epsilon,
        )
        tn.load_state_dict(cpu_norm.state_dict())
        return tn

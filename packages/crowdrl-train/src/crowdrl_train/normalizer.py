"""Running normalization for observations, rewards, and value targets.

Observation normalization: Andrychowicz et al. (2021): "Always use observation
normalization." Clip to [-10, 10] (Huang et al. 2022, detail #29).

Value normalization: Yu et al. (2022): "often helps and never hurts MAPPO."
ValueNorm normalises value targets to zero mean, unit variance.

Reward normalization: Yu et al. (2022), Huang et al. (2022, detail #30):
divide by running std of returns, NO mean subtraction (would change optimal policy).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Cap the running sample count. The DDP reward-normalizer sync
# (``crowdrl_torch.distributed.sync_reward_normalizer``) re-sums the *already
# merged* count every rollout, so without a cap the count grows geometrically
# (~doubles per rollout) and overflows float64 within a few hundred rollouts --
# then ``m_a = var * count`` becomes ``inf``, ``var`` becomes ``NaN``, and every
# normalized reward is ``NaN`` (the deterministic r360 collapse). This is the twin
# of the observation-normalizer overflow capped in ``crowdrl_torch.normalizer``
# (``_MAX_COUNT`` there). Capping bounds ``m_a`` while leaving a stable
# large-window estimator -- the intended near-frozen running-statistics behaviour.
_MAX_COUNT = 1e8


class RunningNormalizer:
    """Welford's online algorithm for running mean and variance.

    Used for observation normalization (shared across all agents with
    parameter sharing) and reward/return normalization.
    """

    def __init__(self, shape: tuple[int, ...], clip: float = 10.0, epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count: float = 1e-4  # small initial count for numerical stability
        self.clip = clip
        self.epsilon = epsilon

    def update(self, batch: NDArray) -> None:
        """Update running statistics with a batch of samples.

        Parameters
        ----------
        batch : (..., *shape) array — one or more samples.
        """
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == len(self.mean.shape):
            batch = batch[np.newaxis]
        # Reshape to (N, *shape) for batch processing
        batch = batch.reshape(-1, *self.mean.shape)

        # Drop non-finite samples (mirror TorchRunningNormalizer): a single NaN/Inf
        # sample must never permanently poison the running mean/var, which would
        # NaN every future normalized value and unrecoverably kill the run.
        finite_rows = np.isfinite(batch).all(axis=tuple(range(1, batch.ndim)))
        if not finite_rows.all():
            batch = batch[finite_rows]
        if batch.shape[0] == 0:
            return

        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: NDArray, batch_var: NDArray, batch_count: int
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

    def normalize(self, x: NDArray) -> NDArray:
        """Normalize input using current statistics.

        Returns clipped (x - mean) / sqrt(var + eps).
        """
        return np.clip(
            (x - self.mean) / np.sqrt(self.var + self.epsilon),
            -self.clip,
            self.clip,
        )

    def state_dict(self) -> dict:
        """Serialise for checkpointing."""
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from checkpoint."""
        self.mean = state["mean"].copy()
        self.var = state["var"].copy()
        # Cap a possibly-overflowed count from a checkpoint trained before the cap
        # existed (the runaway-count bug inflated it past 1e300), so the first
        # update's m_a = var * count cannot overflow.
        self.count = min(float(state["count"]), _MAX_COUNT)


class RewardNormalizer:
    """Normalises rewards by dividing by running std of returns.

    Huang et al. (2022, detail #30): track running variance of discounted
    return sums. Divide rewards by sqrt(var). No mean subtraction —
    subtracting the mean changes the optimal policy.
    """

    def __init__(self, gamma: float = 0.99, clip: float = 10.0, epsilon: float = 1e-8):
        self.gamma = gamma
        self.clip = clip
        self.epsilon = epsilon
        self._return_var = RunningNormalizer(shape=(1,), clip=clip, epsilon=epsilon)
        self._running_return: float = 0.0

    def normalize(self, rewards: NDArray, dones: NDArray) -> NDArray:
        """Normalize a batch of per-agent rewards.

        Parameters
        ----------
        rewards : (n_agents,) rewards for one timestep
        dones : (n_agents,) bool — True if agent done this step

        Returns
        -------
        (n_agents,) normalised rewards
        """
        # Update running return estimate for variance tracking
        # Use mean reward/done across agents as the scalar signal
        mean_reward = rewards.mean()
        mean_done = dones.astype(np.float64).mean()
        self._running_return = self._running_return * self.gamma * (1.0 - mean_done) + mean_reward
        self._return_var.update(np.array([self._running_return]))

        # Divide by std, no mean subtraction
        std = np.sqrt(self._return_var.var[0] + self.epsilon)
        return np.clip(rewards / std, -self.clip, self.clip)

    def state_dict(self) -> dict:
        return {
            "return_var": self._return_var.state_dict(),
            "running_return": self._running_return,
        }

    def load_state_dict(self, state: dict) -> None:
        self._return_var.load_state_dict(state["return_var"])
        self._running_return = state["running_return"]

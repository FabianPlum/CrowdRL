"""Rollout buffer with per-agent GAE for variable agent counts.

Design rationale:
- CrowdEnv returns batched arrays where n_agents varies between episodes.
- With MAPPO parameter sharing, every agent-step is an independent training
  sample (Yu et al. 2022). We flatten all active agent-steps into one batch.
- GAE must handle mid-episode agent termination: when an agent reaches its
  goal (done=True), its advantage chain resets (no bootstrapping).
- Inactive agents (active_mask=False) are excluded from the flat batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass
class FlatBatch:
    """Flattened batch of active agent-steps, ready for PPO update.

    All tensors have shape (total_active_steps, ...) on the target device.
    """

    obs: torch.Tensor
    """(N, obs_dim) observations."""

    actions_raw: torch.Tensor
    """(N, action_dim) unclipped actions (for log-prob re-evaluation)."""

    log_probs: torch.Tensor
    """(N,) log-probabilities under the collection policy."""

    advantages: torch.Tensor
    """(N,) GAE advantages."""

    returns: torch.Tensor
    """(N,) discounted returns (advantages + values)."""

    values: torch.Tensor
    """(N,) value estimates from collection."""

    @property
    def batch_size(self) -> int:
        return self.obs.shape[0]

    def minibatch_indices(
        self, n_minibatches: int, generator: torch.Generator
    ) -> list[torch.Tensor]:
        """Return shuffled index splits for minibatch PPO updates.

        With n_minibatches=1 (MAPPO default), returns a single index tensor
        covering the full batch.
        """
        perm = torch.randperm(self.batch_size, generator=generator)
        if n_minibatches <= 1:
            return [perm]
        chunk_size = self.batch_size // n_minibatches
        indices = []
        for i in range(n_minibatches):
            start = i * chunk_size
            end = start + chunk_size if i < n_minibatches - 1 else self.batch_size
            indices.append(perm[start:end])
        return indices


class RolloutBuffer:
    """Collects rollout data from CrowdEnv and computes GAE.

    Stores per-timestep arrays with the actual agent count (no padding).
    Handles variable n_agents across episodes and mid-episode termination.

    Usage:
        buffer = RolloutBuffer(obs_dim, action_dim, device)
        # Collect one or more episodes:
        for each timestep:
            buffer.add(obs, actions_raw, log_probs, rewards, values, dones, active_mask)
        buffer.compute_gae(last_values=0, last_dones=True, gamma, gae_lambda)
        batch = buffer.flatten()
        # PPO update with batch
        buffer.clear()
    """

    def __init__(self, obs_dim: int, action_dim: int, device: torch.device):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        # Per-timestep storage (lists of arrays, each with shape (n_agents_t, ...))
        self._obs: list[NDArray] = []
        self._actions_raw: list[NDArray] = []
        self._log_probs: list[NDArray] = []
        self._rewards: list[NDArray] = []
        self._values: list[NDArray] = []
        self._dones: list[NDArray] = []
        self._active_masks: list[NDArray] = []

        # Episode boundary tracking
        self._episode_starts: list[int] = [0]
        self._n_agents_per_episode: list[int] = []

        # Computed after GAE
        self._advantages: list[NDArray] = []
        self._returns: list[NDArray] = []

    @property
    def total_steps(self) -> int:
        """Total timesteps stored (not agent-steps)."""
        return len(self._obs)

    @property
    def total_active_agent_steps(self) -> int:
        """Total active agent-steps across all stored timesteps."""
        return sum(int(m.sum()) for m in self._active_masks)

    def add(
        self,
        obs: NDArray,
        actions_raw: NDArray,
        log_probs: NDArray,
        rewards: NDArray,
        values: NDArray,
        dones: NDArray,
        active_mask: NDArray,
    ) -> None:
        """Store one timestep of data for all agents.

        Parameters
        ----------
        obs : (n_agents, obs_dim) — observations (already normalised if applicable)
        actions_raw : (n_agents, action_dim) — unclipped action samples
        log_probs : (n_agents,) — log-probabilities of raw actions
        rewards : (n_agents,) — per-agent rewards
        values : (n_agents,) — per-agent value estimates
        dones : (n_agents,) bool — terminated | truncated
        active_mask : (n_agents,) bool — True if agent is active this step
        """
        self._obs.append(obs.copy())
        self._actions_raw.append(actions_raw.copy())
        self._log_probs.append(log_probs.copy())
        self._rewards.append(rewards.copy())
        self._values.append(values.copy())
        self._dones.append(dones.astype(np.bool_).copy())
        self._active_masks.append(active_mask.astype(np.bool_).copy())

    def mark_episode_end(self) -> None:
        """Mark the current timestep as the end of an episode.

        Must be called after the last add() of each episode (when
        info['episode_over'] is True).
        """
        self._episode_starts.append(len(self._obs))
        if len(self._obs) > 0:
            self._n_agents_per_episode.append(self._obs[-1].shape[0])

    def compute_gae(
        self,
        last_values: NDArray | list[NDArray],
        last_dones: NDArray | list[NDArray],
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE advantages and returns for all stored episodes.

        Processes each episode independently, handling per-agent termination:
        when done[agent]=True, that agent's advantage chain resets.
        Inactive agents get zero advantage and zero return.

        Parameters
        ----------
        last_values : bootstrap values for the final timestep of each episode.
            - Single array (n_agents,): applied to the last episode only;
              completed episodes get zero bootstrap (backward compatible).
            - List of arrays: one per episode in buffer order. Completed
              episodes should pass zeros; incomplete episodes pass V(s_last).
        last_dones : done flags matching ``last_values`` structure.
        gamma : discount factor
        gae_lambda : GAE lambda
        """
        n_steps = len(self._obs)
        if n_steps == 0:
            self._advantages = []
            self._returns = []
            return

        self._advantages = [np.zeros_like(v) for v in self._values]
        self._returns = [np.zeros_like(v) for v in self._values]

        # Build episode ranges
        if self._episode_starts[-1] < n_steps:
            episode_ranges = list(
                zip(
                    self._episode_starts,
                    self._episode_starts[1:] + [n_steps],
                )
            )
        else:
            episode_ranges = list(
                zip(
                    self._episode_starts[:-1],
                    self._episode_starts[1:],
                )
            )

        # Normalise bootstrap args to per-episode lists
        per_episode = isinstance(last_values, list)

        for ep_idx, (ep_start, ep_end) in enumerate(episode_ranges):
            if ep_start >= ep_end:
                continue
            n_agents = self._obs[ep_start].shape[0]

            if per_episode:
                # Per-episode bootstrap provided by caller
                bootstrap_values = last_values[ep_idx][:n_agents].astype(np.float64)
            else:
                # Legacy: single array applies to the last episode only
                if ep_end == n_steps:
                    bootstrap_values = (
                        last_values[:n_agents]
                        if len(last_values) >= n_agents
                        else np.zeros(n_agents, dtype=np.float64)
                    )
                else:
                    bootstrap_values = np.zeros(n_agents, dtype=np.float64)

            # Reverse sweep for GAE
            gae = np.zeros(n_agents, dtype=np.float64)
            next_values = bootstrap_values

            for t in reversed(range(ep_start, ep_end)):
                active = self._active_masks[t]
                rewards = self._rewards[t]
                values = self._values[t]
                dones = self._dones[t]

                # TD error: delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
                not_done = (~dones).astype(np.float64)
                delta = rewards + gamma * next_values * not_done - values

                # GAE: A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
                gae = delta + gamma * gae_lambda * not_done * gae

                # Zero out inactive agents
                gae *= active.astype(np.float64)

                self._advantages[t] = gae.copy()
                self._returns[t] = gae + values

                next_values = values

    def flatten(self) -> FlatBatch:
        """Flatten all active agent-steps into a single batch.

        Only includes timesteps where active_mask is True.
        Converts to torch tensors on self.device.
        """
        if self._advantages is None:
            raise RuntimeError("Call compute_gae() before flatten()")

        # Collect active entries
        all_obs = []
        all_actions = []
        all_log_probs = []
        all_advantages = []
        all_returns = []
        all_values = []

        for t in range(len(self._obs)):
            mask = self._active_masks[t]
            if not mask.any():
                continue
            all_obs.append(self._obs[t][mask])
            all_actions.append(self._actions_raw[t][mask])
            all_log_probs.append(self._log_probs[t][mask])
            all_advantages.append(self._advantages[t][mask])
            all_returns.append(self._returns[t][mask])
            all_values.append(self._values[t][mask])

        if not all_obs:
            # Edge case: no active steps at all
            return FlatBatch(
                obs=torch.zeros((0, self.obs_dim), device=self.device),
                actions_raw=torch.zeros((0, self.action_dim), device=self.device),
                log_probs=torch.zeros(0, device=self.device),
                advantages=torch.zeros(0, device=self.device),
                returns=torch.zeros(0, device=self.device),
                values=torch.zeros(0, device=self.device),
            )

        return FlatBatch(
            obs=torch.as_tensor(np.concatenate(all_obs), dtype=torch.float32, device=self.device),
            actions_raw=torch.as_tensor(
                np.concatenate(all_actions), dtype=torch.float32, device=self.device
            ),
            log_probs=torch.as_tensor(
                np.concatenate(all_log_probs), dtype=torch.float32, device=self.device
            ),
            advantages=torch.as_tensor(
                np.concatenate(all_advantages), dtype=torch.float32, device=self.device
            ),
            returns=torch.as_tensor(
                np.concatenate(all_returns), dtype=torch.float32, device=self.device
            ),
            values=torch.as_tensor(
                np.concatenate(all_values), dtype=torch.float32, device=self.device
            ),
        )

    def clear(self) -> None:
        """Reset buffer for next rollout."""
        self._obs.clear()
        self._actions_raw.clear()
        self._log_probs.clear()
        self._rewards.clear()
        self._values.clear()
        self._dones.clear()
        self._active_masks.clear()
        self._episode_starts = [0]
        self._n_agents_per_episode.clear()
        self._advantages.clear()
        self._returns.clear()

"""Vectorized rollout collector for PyTorch batched environments.

The collect loop stays on GPU with minimal CPU interaction:

1. Observations normalised on-device (``TorchRunningNormalizer``)
2. Policy forward pass on-device — all E*N agents in one batch
3. Batched env step on-device
4. Per-step data stored as padded (E, N, ...) arrays — **no per-env loop**
5. Per-env GAE computed once at the end of collection

The only per-env Python loop is at GAE time (once per ``collect()`` call),
not in the per-step hot path.
"""

from __future__ import annotations

import numpy as np
import torch

from crowdrl_train.buffer import FlatBatch
from crowdrl_train.networks import ActorCritic
from crowdrl_train.normalizer import RewardNormalizer

from crowdrl_torch.batched_env import BatchedTorchEnv
from crowdrl_torch.normalizer import TorchRunningNormalizer


class TorchRolloutCollector:
    """Vectorized rollout collector — no per-env loop in the step hot path.

    Parameters
    ----------
    batched_env : BatchedTorchEnv
    actor_critic : PyTorch policy network (on device)
    obs_normalizer : GPU-resident observation normalizer (or None)
    reward_normalizer : optional reward normalizer (CPU, per-env)
    torch_device : torch.device for inference
    obs_dim, action_dim : dimensions
    """

    def __init__(
        self,
        batched_env: BatchedTorchEnv,
        actor_critic: ActorCritic,
        obs_normalizer: TorchRunningNormalizer | None,
        reward_normalizer: RewardNormalizer | None,
        torch_device: torch.device,
        obs_dim: int,
        action_dim: int,
    ):
        self.env = batched_env
        self.actor_critic = actor_critic
        self.obs_normalizer = obs_normalizer
        self.reward_normalizer = reward_normalizer
        self.torch_device = torch_device
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self._n_envs = batched_env.n_envs
        self._config = batched_env.config
        self._max_agents = self._config.max_agents

        # --- Persistent episode state ---
        # These span multiple ``collect()`` calls so episodes that don't finish
        # within a single rollout are not discarded. The initial reset happens
        # lazily on the first ``collect()`` call.
        self._obs_t: torch.Tensor | None = None
        self._ep_rewards: np.ndarray | None = None
        self._ep_terminated: np.ndarray | None = None
        self._ep_lengths: np.ndarray | None = None
        self._n_agents_t: np.ndarray | None = None

    def collect(self, n_agent_steps: int) -> list[dict]:
        """Collect at least ``n_agent_steps`` across all envs.

        Returns list of completed episode stat dicts.

        Episodes that do not finish within a single ``collect()`` call are
        carried over to the next call via the persistent state attributes,
        so no agent-steps are wasted and recorded episode statistics reflect
        real episodes rather than a biased short-episode tail.
        """
        E = self._n_envs
        N = self._max_agents
        D = self.obs_dim
        A = self.action_dim
        dev = self.torch_device

        # --- Lazy one-time initialisation: reset envs on the first call ---
        if self._obs_t is None:
            states, obs_t = self.env.reset_all()
            self.env.states = states
            self._obs_t = obs_t
            self._ep_rewards = np.zeros((E, N), dtype=np.float64)
            self._ep_terminated = np.zeros((E, N), dtype=np.bool_)
            self._ep_lengths = np.zeros(E, dtype=np.int32)
            self._n_agents_t = self.env.states.n_agents.cpu().numpy().astype(np.int32)

        obs_t = self._obs_t
        ep_rewards = self._ep_rewards
        ep_terminated = self._ep_terminated
        ep_lengths = self._ep_lengths
        n_agents_t = self._n_agents_t
        assert ep_rewards is not None and ep_terminated is not None
        assert ep_lengths is not None and n_agents_t is not None

        # --- Per-step padded storage (fresh each collect) ---
        step_obs: list[np.ndarray] = []
        step_actions_raw: list[np.ndarray] = []
        step_log_probs: list[np.ndarray] = []
        step_rewards: list[np.ndarray] = []
        step_values: list[np.ndarray] = []
        step_dones: list[np.ndarray] = []
        step_active: list[np.ndarray] = []

        # --- Episode boundary tracking (per-env, relative to current collect) ---
        # episode_starts[i] = list of timestep indices (within this collect)
        # where env i started a new episode segment. The first entry is always
        # 0: whatever segment is currently running (possibly from a previous
        # collect) counts as "segment 0" from the buffer's perspective.
        episode_starts: list[list[int]] = [[0] for _ in range(E)]

        agent_idx = np.arange(N)[None, :]  # (1, N)
        completed_episodes: list[dict] = []
        steps_collected = 0
        t = 0  # global timestep counter (within this collect)

        while steps_collected < n_agent_steps:
            # --- Agent masks (numpy, computed once) ---
            agent_mask = agent_idx < n_agents_t[:, None]  # (E, N) bool
            active_np = self.env.states.active_mask.cpu().numpy()  # (E, N)
            real_active = agent_mask & active_np

            # --- Normalise observations on GPU (no CPU roundtrip) ---
            if self.obs_normalizer is not None:
                # Update running stats with active observations
                active_obs = obs_t[torch.as_tensor(real_active, device=dev)]
                if active_obs.shape[0] > 0:
                    self.obs_normalizer.update(active_obs)
                obs_norm_t = self.obs_normalizer.normalize(obs_t)  # (E, N, D) on device
            else:
                obs_norm_t = obs_t

            # --- Forward pass on device (all E*N slots) ---
            with torch.no_grad():
                flat_obs = obs_norm_t.reshape(-1, D)
                actions_t, actions_raw_t, log_probs_t, _ent, values_t = (
                    self.actor_critic.get_action_and_value(flat_obs)
                )

            # --- Step all envs on device ---
            actions_gpu = actions_t.reshape(E, N, A)
            self.env.states, obs_t, rewards_t, terminated_t, truncated_t = self.env.step(
                actions_gpu
            )

            # --- Single bulk transfer to CPU ---
            obs_norm_np = obs_norm_t.cpu().numpy()  # (E, N, D)
            actions_raw_np = actions_raw_t.cpu().numpy().reshape(E, N, A)
            log_probs_np = log_probs_t.cpu().numpy().reshape(E, N)
            values_np = values_t.cpu().numpy().reshape(E, N)
            rewards_np = rewards_t.cpu().numpy()  # (E, N)
            terminated_np = terminated_t.cpu().numpy()
            truncated_np = truncated_t.cpu().numpy()
            dones_np = terminated_np | truncated_np

            # --- Reward normalisation (shared, uses batch statistics) ---
            if self.reward_normalizer is not None:
                # Use mean across all active agents as the scalar signal
                active_rewards = rewards_np[real_active]
                if active_rewards.size > 0:
                    rewards_np = self._normalize_rewards_batched(rewards_np, dones_np, real_active)

            # --- Append padded arrays (one append, no per-env loop) ---
            step_obs.append(obs_norm_np)
            step_actions_raw.append(actions_raw_np)
            step_log_probs.append(log_probs_np)
            step_rewards.append(rewards_np)
            step_values.append(values_np)
            step_dones.append(dones_np)
            step_active.append(active_np)

            # --- Vectorized episode tracking ---
            ep_rewards += rewards_np * active_np
            ep_terminated |= terminated_np
            ep_lengths += np.any(real_active, axis=1).astype(np.int32)
            steps_collected += int(real_active.sum())

            # --- Detect completed episodes (vectorized) ---
            # Use the pre-reset episode_over mask from the env, NOT active_mask.
            # active_mask may already reflect a new episode if an async reset
            # completed within step(), which would cause us to miss the boundary
            # and contaminate ep_terminated with data from multiple episodes.
            episode_over = self.env.episode_over.cpu().numpy()  # (E,) bool

            t += 1

            # Resync n_agents_t from env state every step: this handles
            # async resets that complete between step() calls. For envs that
            # just hit episode_over this step, state.n_agents[i] is still the
            # old value (their reset is pending), so recording below is
            # correct. For envs whose reset just applied, we pick up the new
            # spawn count here for the following iteration.
            new_n_agents = self.env.states.n_agents.cpu().numpy().astype(np.int32)

            if episode_over.any():
                env_tiers = getattr(self.env, "env_tiers", None)
                for i in np.where(episode_over)[0]:
                    n_ag = int(n_agents_t[i])
                    if n_ag == 0:
                        continue

                    # Record episode stats
                    n_reached = int(ep_terminated[i, :n_ag].sum())
                    total_rew = float(ep_rewards[i, :n_ag].sum())
                    ep_dict = {
                        "n_agents": n_ag,
                        "episode_length": int(ep_lengths[i]),
                        "goal_rate": n_reached / n_ag,
                        "n_reached_goal": n_reached,
                        "mean_reward": total_rew / n_ag,
                        "total_reward": total_rew,
                    }
                    if env_tiers is not None:
                        ep_dict["geometry_tier"] = env_tiers[i]
                    completed_episodes.append(ep_dict)

                    # Mark episode boundary for this env (new segment starts
                    # at the next buffer index).
                    episode_starts[i].append(t)

                    # Reset per-env episode tracking
                    ep_rewards[i] = 0.0
                    ep_terminated[i] = False
                    ep_lengths[i] = 0

            # Wholesale resync after processing episode_over so envs whose
            # async reset just applied (with no episode_over this step) also
            # get their n_agents_t updated to the new episode's spawn count.
            n_agents_t[:] = new_n_agents

        # Store the post-step observation for bootstrap (correct s_T,
        # not the s_{T-1} stored in step_obs).  Normalize once here;
        # compute_gae_and_flatten must NOT re-normalize.
        if self.obs_normalizer is not None:
            self._final_obs_norm = self.obs_normalizer.normalize(obs_t).cpu().numpy()
        else:
            self._final_obs_norm = obs_t.cpu().numpy()

        # Persist obs for the next collect so in-progress episodes survive.
        self._obs_t = obs_t

        # --- Store collected data for GAE ---
        self._step_obs = step_obs
        self._step_actions_raw = step_actions_raw
        self._step_log_probs = step_log_probs
        self._step_rewards = step_rewards
        self._step_values = step_values
        self._step_dones = step_dones
        self._step_active = step_active
        self._episode_starts = episode_starts
        self._n_agents_per_env = n_agents_t.copy()
        self._total_steps = t
        self._total_active = steps_collected

        return completed_episodes

    def compute_gae_and_flatten(
        self,
        gamma: float,
        gae_lambda: float,
    ) -> FlatBatch:
        """Compute GAE per env and merge into one FlatBatch.

        This is where per-env splitting happens — once per collect,
        not once per step.
        """
        T = self._total_steps
        E = self._n_envs
        N = self._max_agents
        if T == 0:
            return self._empty_batch()

        all_obs = []
        all_actions = []
        all_log_probs = []
        all_advantages = []
        all_returns = []
        all_values = []

        # We slice each segment over the full max_agents axis and rely on the
        # per-step active_mask to select real agents. Inferring n_ag from the
        # first step's active count breaks when a segment starts mid-episode
        # (carry-over from a previous collect) because terminated agents can
        # be scattered across the row rather than packed at the tail.
        for i in range(E):
            ep_bounds = self._episode_starts[i]

            for ep_idx in range(len(ep_bounds) - 1):
                ep_start = ep_bounds[ep_idx]
                ep_end = ep_bounds[ep_idx + 1]
                if ep_start >= ep_end:
                    continue

                # Extract per-env per-episode data (full N columns)
                ep_obs = [self._step_obs[t][i] for t in range(ep_start, ep_end)]
                ep_act = [self._step_actions_raw[t][i] for t in range(ep_start, ep_end)]
                ep_lp = [self._step_log_probs[t][i] for t in range(ep_start, ep_end)]
                ep_rew = [self._step_rewards[t][i] for t in range(ep_start, ep_end)]
                ep_val = [self._step_values[t][i] for t in range(ep_start, ep_end)]
                ep_done = [self._step_dones[t][i] for t in range(ep_start, ep_end)]
                ep_active = [self._step_active[t][i] for t in range(ep_start, ep_end)]

                # If nothing is active anywhere in the segment, skip entirely
                # (idle env waiting for an async reset to apply).
                if not any(a.any() for a in ep_active):
                    continue

                # Completed segment: bootstrap with zeros.
                bootstrap_values = np.zeros(N, dtype=np.float64)

                gae = np.zeros(N, dtype=np.float64)
                next_values = bootstrap_values
                advantages = [None] * len(ep_obs)
                returns = [None] * len(ep_obs)

                for t_idx in reversed(range(len(ep_obs))):
                    active = ep_active[t_idx].astype(np.float64)
                    rewards = ep_rew[t_idx].astype(np.float64)
                    values = ep_val[t_idx].astype(np.float64)
                    dones = ep_done[t_idx]
                    not_done = (~dones).astype(np.float64)

                    delta = rewards + gamma * next_values * not_done - values
                    gae = delta + gamma * gae_lambda * not_done * gae
                    gae *= active

                    advantages[t_idx] = gae.copy()
                    returns[t_idx] = gae + values
                    next_values = values

                # Flatten active steps for this segment
                for t_idx in range(len(ep_obs)):
                    mask = ep_active[t_idx].astype(np.bool_)
                    if not mask.any():
                        continue
                    all_obs.append(ep_obs[t_idx][mask])
                    all_actions.append(ep_act[t_idx][mask])
                    all_log_probs.append(ep_lp[t_idx][mask])
                    all_advantages.append(advantages[t_idx][mask])
                    all_returns.append(returns[t_idx][mask])
                    all_values.append(ep_val[t_idx][mask])

            # Handle the trailing incomplete segment (no episode_end marker).
            last_start = ep_bounds[-1]
            if last_start < T:
                ep_obs = [self._step_obs[t][i] for t in range(last_start, T)]
                ep_act = [self._step_actions_raw[t][i] for t in range(last_start, T)]
                ep_lp = [self._step_log_probs[t][i] for t in range(last_start, T)]
                ep_rew = [self._step_rewards[t][i] for t in range(last_start, T)]
                ep_val = [self._step_values[t][i] for t in range(last_start, T)]
                ep_done = [self._step_dones[t][i] for t in range(last_start, T)]
                ep_active = [self._step_active[t][i] for t in range(last_start, T)]

                if not any(a.any() for a in ep_active):
                    continue

                # Bootstrap from the post-step observation at collection end.
                # _final_obs_norm[i] is already normalised once.
                last_obs_np = self._final_obs_norm[i]

                with torch.no_grad():
                    obs_gpu = torch.as_tensor(
                        last_obs_np, dtype=torch.float32, device=self.torch_device
                    )
                    bootstrap_values = (
                        self.actor_critic.get_value(obs_gpu).cpu().numpy().astype(np.float64)
                    )

                gae = np.zeros(N, dtype=np.float64)
                next_values = bootstrap_values
                advantages = [None] * len(ep_obs)
                returns_arr = [None] * len(ep_obs)

                for t_idx in reversed(range(len(ep_obs))):
                    active = ep_active[t_idx].astype(np.float64)
                    rewards = ep_rew[t_idx].astype(np.float64)
                    values = ep_val[t_idx].astype(np.float64)
                    dones = ep_done[t_idx]
                    not_done = (~dones).astype(np.float64)

                    delta = rewards + gamma * next_values * not_done - values
                    gae = delta + gamma * gae_lambda * not_done * gae
                    gae *= active

                    advantages[t_idx] = gae.copy()
                    returns_arr[t_idx] = gae + values
                    next_values = values

                for t_idx in range(len(ep_obs)):
                    mask = ep_active[t_idx].astype(np.bool_)
                    if not mask.any():
                        continue
                    all_obs.append(ep_obs[t_idx][mask])
                    all_actions.append(ep_act[t_idx][mask])
                    all_log_probs.append(ep_lp[t_idx][mask])
                    all_advantages.append(advantages[t_idx][mask])
                    all_returns.append(returns_arr[t_idx][mask])
                    all_values.append(ep_val[t_idx][mask])

        if not all_obs:
            return self._empty_batch()

        return FlatBatch(
            obs=torch.as_tensor(
                np.concatenate(all_obs), dtype=torch.float32, device=self.torch_device
            ),
            actions_raw=torch.as_tensor(
                np.concatenate(all_actions), dtype=torch.float32, device=self.torch_device
            ),
            log_probs=torch.as_tensor(
                np.concatenate(all_log_probs), dtype=torch.float32, device=self.torch_device
            ),
            advantages=torch.as_tensor(
                np.concatenate(all_advantages), dtype=torch.float32, device=self.torch_device
            ),
            returns=torch.as_tensor(
                np.concatenate(all_returns), dtype=torch.float32, device=self.torch_device
            ),
            values=torch.as_tensor(
                np.concatenate(all_values), dtype=torch.float32, device=self.torch_device
            ),
        )

    @property
    def total_active_agent_steps(self) -> int:
        return getattr(self, "_total_active", 0)

    def _normalize_rewards_batched(
        self, rewards: np.ndarray, dones: np.ndarray, active: np.ndarray
    ) -> np.ndarray:
        """Normalise rewards using shared RewardNormalizer with batch stats."""
        active_r = rewards[active]
        active_d = dones[active]
        if active_r.size == 0:
            return rewards
        mean_reward = active_r.mean()
        mean_done = active_d.astype(np.float64).mean()
        self.reward_normalizer._running_return = (
            self.reward_normalizer._running_return
            * self.reward_normalizer.gamma
            * (1.0 - mean_done)
            + mean_reward
        )
        self.reward_normalizer._return_var.update(
            np.array([self.reward_normalizer._running_return])
        )
        std = np.sqrt(self.reward_normalizer._return_var.var[0] + self.reward_normalizer.epsilon)
        normalized = rewards / std
        return np.clip(normalized, -self.reward_normalizer.clip, self.reward_normalizer.clip)

    def _empty_batch(self) -> FlatBatch:
        return FlatBatch(
            obs=torch.zeros((0, self.obs_dim), device=self.torch_device),
            actions_raw=torch.zeros((0, self.action_dim), device=self.torch_device),
            log_probs=torch.zeros(0, device=self.torch_device),
            advantages=torch.zeros(0, device=self.torch_device),
            returns=torch.zeros(0, device=self.torch_device),
            values=torch.zeros(0, device=self.torch_device),
        )

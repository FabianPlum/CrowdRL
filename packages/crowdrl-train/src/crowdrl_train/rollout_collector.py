"""Multi-environment rollout collection with central GPU inference.

Collects transitions from a :class:`SubprocVecEnv` into **per-env buffers**
(each env has its own :class:`RolloutBuffer`). The main process batches
observations from all workers into a single GPU forward pass, then
distributes actions back.

Per-env buffers are necessary because different envs have different
``n_agents``, and GAE requires consistent agent counts within an episode.
After collection, GAE is computed per buffer and the results are merged
into a single :class:`FlatBatch` for the PPO update.

Env resets are non-blocking: when an episode finishes, the reset command is
sent immediately but the collector does not wait for the result. The env is
skipped for the current step and its new state is picked up on the next
iteration. This prevents one slow reset from blocking all other envs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray

from crowdrl_train.buffer import FlatBatch, RolloutBuffer
from crowdrl_train.networks import ActorCritic
from crowdrl_train.normalizer import RewardNormalizer, RunningNormalizer
from crowdrl_train.vec_env import SubprocVecEnv


@dataclass
class _EnvState:
    """Per-environment bookkeeping during collection."""

    obs: NDArray
    """Current observation, (n_agents, obs_dim)."""

    n_agents: int
    active_mask: NDArray
    """(n_agents,) bool — agents still active in current episode."""

    cumulative_terminated: NDArray
    """(n_agents,) bool — agents that reached goal so far."""

    episode_length: int = 0
    episode_rewards: NDArray = field(default_factory=lambda: np.array([]))
    info: dict = field(default_factory=dict)
    resetting: bool = False
    """True if this env has a pending reset (non-blocking)."""


class RolloutCollector:
    """Collects rollouts from multiple parallel envs with central inference.

    Each env gets its own :class:`RolloutBuffer` so that timesteps within
    an episode always have consistent ``n_agents``. After collection,
    call :meth:`compute_gae_and_flatten` to get a single merged batch.

    Parameters
    ----------
    vec_env : SubprocVecEnv with N workers
    actor_critic : policy network (on device)
    obs_normalizer : optional observation normalizer (updated in-place)
    reward_normalizer : optional reward normalizer (updated in-place)
    device : torch device for inference
    obs_dim, action_dim : for constructing per-env buffers
    """

    def __init__(
        self,
        vec_env: SubprocVecEnv,
        actor_critic: ActorCritic,
        obs_normalizer: RunningNormalizer | None,
        reward_normalizer: RewardNormalizer | None,
        device: torch.device,
        obs_dim: int,
        action_dim: int,
    ):
        self.vec_env = vec_env
        self.actor_critic = actor_critic
        self.obs_normalizer = obs_normalizer
        self.reward_normalizer = reward_normalizer
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self._n_envs = vec_env.n_envs
        # Per-env buffers — each env's data is self-contained
        self._buffers: list[RolloutBuffer] = [
            RolloutBuffer(obs_dim, action_dim, device) for _ in range(self._n_envs)
        ]
        self._env_states: list[_EnvState] = []

    @property
    def buffers(self) -> list[RolloutBuffer]:
        return self._buffers

    def _finish_pending_resets(self) -> None:
        """Collect results from any envs that were sent a reset command."""
        for i, es in enumerate(self._env_states):
            if not es.resetting:
                continue
            new_obs, new_info = self.vec_env._recv(i)
            n_agents = new_info["n_agents"]
            self._env_states[i] = _EnvState(
                obs=new_obs,
                n_agents=n_agents,
                active_mask=np.ones(n_agents, dtype=np.bool_),
                cumulative_terminated=np.zeros(n_agents, dtype=np.bool_),
                episode_rewards=np.zeros(n_agents, dtype=np.float64),
                info=new_info,
            )

    def collect(self, n_agent_steps: int) -> list[dict]:
        """Collect at least *n_agent_steps* across all envs.

        Transitions are stored in per-env buffers (not interleaved).
        Returns a list of completed-episode stat dicts.
        """
        completed_episodes: list[dict] = []
        steps_collected = 0

        # Clear all buffers
        for buf in self._buffers:
            buf.clear()

        # Reset all envs and init per-env state
        reset_results = self.vec_env.reset_all()
        self._env_states = []
        for obs, info in reset_results:
            n_agents = info["n_agents"]
            self._env_states.append(
                _EnvState(
                    obs=obs,
                    n_agents=n_agents,
                    active_mask=np.ones(n_agents, dtype=np.bool_),
                    cumulative_terminated=np.zeros(n_agents, dtype=np.bool_),
                    episode_rewards=np.zeros(n_agents, dtype=np.float64),
                    info=info,
                )
            )

        # Collection loop
        while steps_collected < n_agent_steps:
            # Pick up any pending async resets from the previous iteration
            self._finish_pending_resets()

            # Determine which envs are ready (not resetting)
            ready_envs = [i for i in range(self._n_envs) if not self._env_states[i].resetting]

            if not ready_envs:
                # All envs are resetting — must wait for at least one
                self._finish_pending_resets()
                ready_envs = list(range(self._n_envs))

            # --- Normalise observations for ready envs ---
            all_obs_norm = []
            n_agents_per_env = []
            for i in ready_envs:
                es = self._env_states[i]
                n_agents_per_env.append(es.n_agents)
                if self.obs_normalizer is not None:
                    active_obs = es.obs[es.active_mask]
                    if active_obs.shape[0] > 0:
                        self.obs_normalizer.update(active_obs)
                    all_obs_norm.append(self.obs_normalizer.normalize(es.obs))
                else:
                    all_obs_norm.append(es.obs)

            # --- Batched GPU forward pass ---
            batch_obs = np.concatenate(all_obs_norm, axis=0)
            with torch.no_grad():
                obs_t = torch.as_tensor(batch_obs, dtype=torch.float32, device=self.device)
                actions, actions_raw, log_probs, _entropy, values = (
                    self.actor_critic.get_action_and_value(obs_t)
                )

            # Split back per env
            actions_np = actions.cpu().numpy()
            actions_raw_np = actions_raw.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()
            values_np = values.cpu().numpy()

            splits = np.cumsum(n_agents_per_env[:-1])
            actions_split = np.split(actions_np, splits)
            actions_raw_split = np.split(actions_raw_np, splits)
            log_probs_split = np.split(log_probs_np, splits)
            values_split = np.split(values_np, splits)

            # --- Step ready envs ---
            # Send step commands to ready envs
            for idx, env_i in enumerate(ready_envs):
                self.vec_env._main_pipes[env_i].send(("step", actions_split[idx]))

            # Receive results from ready envs
            step_results = {}
            for env_i in ready_envs:
                obs, rewards, terminated, truncated, info = self.vec_env._recv(env_i)
                from crowdrl_train.vec_env import StepResult

                step_results[env_i] = StepResult(obs, rewards, terminated, truncated, info)

            # --- Process results per env ---
            for idx, env_i in enumerate(ready_envs):
                es = self._env_states[env_i]
                sr = step_results[env_i]
                buf = self._buffers[env_i]

                rewards = sr.rewards.copy()
                dones = sr.terminated | sr.truncated

                # Normalise rewards
                if self.reward_normalizer is not None:
                    rewards = self.reward_normalizer.normalize(rewards, dones)

                # Store transition in this env's buffer
                buf.add(
                    obs=all_obs_norm[idx],
                    actions_raw=actions_raw_split[idx],
                    log_probs=log_probs_split[idx],
                    rewards=rewards,
                    values=values_split[idx],
                    dones=dones,
                    active_mask=es.active_mask.copy(),
                )

                # Track episode stats
                es.episode_rewards += rewards * es.active_mask
                es.cumulative_terminated |= sr.terminated
                es.active_mask = ~(es.cumulative_terminated | sr.truncated)
                es.episode_length += 1

                # Count active agents this step
                steps_collected += int(es.active_mask.sum())

                episode_over = sr.info.get("episode_over", False)

                if episode_over:
                    # Record completed episode
                    buf.mark_episode_end()
                    n_reached = int(es.cumulative_terminated.sum())
                    completed_episodes.append(
                        {
                            "n_agents": es.n_agents,
                            "episode_length": es.episode_length,
                            "goal_rate": (n_reached / es.n_agents if es.n_agents > 0 else 0.0),
                            "n_reached_goal": n_reached,
                            "mean_reward": (
                                float(es.episode_rewards.sum() / es.n_agents)
                                if es.n_agents > 0
                                else 0.0
                            ),
                            "total_reward": float(es.episode_rewards.sum()),
                            "n_collisions": sr.info.get("n_collisions", 0),
                            "geometry_tier": es.info.get("geometry_tier", "unknown"),
                        }
                    )

                    # Non-blocking reset: send command and mark as resetting
                    self.vec_env._main_pipes[env_i].send(("reset", None))
                    es.resetting = True
                else:
                    es.obs = sr.obs

        # Collect any remaining pending resets so state is clean
        self._finish_pending_resets()

        return completed_episodes

    def compute_gae_and_flatten(
        self,
        gamma: float,
        gae_lambda: float,
    ) -> FlatBatch:
        """Compute GAE per env buffer and merge into one FlatBatch.

        For completed episodes, bootstrap values are zero.
        For in-progress episodes (env still mid-episode at collection end),
        bootstrap values come from the critic.
        """
        all_obs = []
        all_actions = []
        all_log_probs = []
        all_advantages = []
        all_returns = []
        all_values = []

        for i, buf in enumerate(self._buffers):
            if buf.total_steps == 0:
                continue

            # Determine bootstrap for the last episode in this buffer
            n_steps = len(buf._obs)
            last_ep_complete = buf._episode_starts[-1] >= n_steps

            if last_ep_complete:
                # All episodes in this buffer are complete
                n_agents = buf._obs[-1].shape[0]
                last_values = np.zeros(n_agents, dtype=np.float64)
                last_dones = np.ones(n_agents, dtype=np.bool_)
            else:
                # Last episode is incomplete — bootstrap from critic
                last_obs = buf._obs[-1]
                if self.obs_normalizer is not None:
                    last_obs = self.obs_normalizer.normalize(last_obs)
                with torch.no_grad():
                    obs_t = torch.as_tensor(last_obs, dtype=torch.float32, device=self.device)
                    last_values = self.actor_critic.get_value(obs_t).cpu().numpy()
                n_agents = buf._obs[-1].shape[0]
                last_dones = np.zeros(n_agents, dtype=np.bool_)

            buf.compute_gae(last_values, last_dones, gamma, gae_lambda)
            flat = buf.flatten()

            if flat.batch_size > 0:
                all_obs.append(flat.obs)
                all_actions.append(flat.actions_raw)
                all_log_probs.append(flat.log_probs)
                all_advantages.append(flat.advantages)
                all_returns.append(flat.returns)
                all_values.append(flat.values)

        if not all_obs:
            return FlatBatch(
                obs=torch.zeros((0, self.obs_dim), device=self.device),
                actions_raw=torch.zeros((0, self.action_dim), device=self.device),
                log_probs=torch.zeros(0, device=self.device),
                advantages=torch.zeros(0, device=self.device),
                returns=torch.zeros(0, device=self.device),
                values=torch.zeros(0, device=self.device),
            )

        return FlatBatch(
            obs=torch.cat(all_obs),
            actions_raw=torch.cat(all_actions),
            log_probs=torch.cat(all_log_probs),
            advantages=torch.cat(all_advantages),
            returns=torch.cat(all_returns),
            values=torch.cat(all_values),
        )

    @property
    def total_active_agent_steps(self) -> int:
        """Total active agent-steps across all per-env buffers."""
        return sum(buf.total_active_agent_steps for buf in self._buffers)

    def clear(self) -> None:
        """Clear all per-env buffers."""
        for buf in self._buffers:
            buf.clear()

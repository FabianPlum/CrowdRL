"""Curriculum manager for progressive difficulty scheduling.

Implements success-rate-driven phase advancement following:
- Narvekar et al. (2020, JMLR): Zone of Proximal Development — moderate difficulty.
- OpenAI (2019, ADR): advance when success rate exceeds threshold.
- Zhao et al. (2022): agent count as a curriculum variable.

Curriculum phases control:
1. Which geometry tiers are sampled per episode
2. Agent count range per episode
3. When to advance to the next difficulty level
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from crowdrl_env.crowd_env import CrowdEnvConfig
from crowdrl_env.spawner import SpawnConfig

from crowdrl_train.config import CurriculumConfig, CurriculumPhase


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""

    goal_rate: float
    n_agents: int
    episode_length: int
    mean_reward: float


class CurriculumManager:
    """Manages progressive difficulty scheduling during training.

    Tracks a rolling window of episode statistics and advances to harder
    phases when the goal rate threshold is exceeded consistently.
    """

    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_phase_idx: int = 0
        self._episode_history: deque[EpisodeStats] = deque(maxlen=config.metric_window)
        self._episodes_in_phase: int = 0
        self._total_episodes: int = 0

    @property
    def current_phase(self) -> CurriculumPhase:
        return self.config.phases[self.current_phase_idx]

    @property
    def is_terminal_phase(self) -> bool:
        return self.current_phase_idx >= len(self.config.phases) - 1

    @property
    def rolling_goal_rate(self) -> float:
        """Mean goal rate over the metric window."""
        if not self._episode_history:
            return 0.0
        return sum(e.goal_rate for e in self._episode_history) / len(self._episode_history)

    def report_episode(self, stats: EpisodeStats) -> bool:
        """Report episode results. Returns True if phase was advanced.

        Parameters
        ----------
        stats : EpisodeStats — statistics for the completed episode

        Returns
        -------
        True if the curriculum advanced to a new phase
        """
        self._episode_history.append(stats)
        self._episodes_in_phase += 1
        self._total_episodes += 1

        if self.is_terminal_phase:
            return False

        if self._should_advance():
            self._advance()
            return True
        return False

    def _should_advance(self) -> bool:
        """Check if promotion criteria are met."""
        if self._episodes_in_phase < self.config.min_episodes_per_phase:
            return False
        if len(self._episode_history) < self.config.metric_window:
            return False
        threshold = self.current_phase.goal_rate_threshold
        if threshold <= 0.0:
            return False  # Terminal phase
        return self.rolling_goal_rate >= threshold

    def _advance(self) -> None:
        """Move to the next curriculum phase."""
        self.current_phase_idx = min(
            self.current_phase_idx + 1,
            len(self.config.phases) - 1,
        )
        self._episodes_in_phase = 0
        self._episode_history.clear()

    def make_env_config(self, base_config: CrowdEnvConfig) -> CrowdEnvConfig:
        """Create a CrowdEnvConfig with curriculum overrides applied.

        Inherits physics/obs/action/reward settings from base_config,
        overrides geometry tiers and agent count for the current phase.
        """
        phase = self.current_phase
        return CrowdEnvConfig(
            geometry=base_config.geometry,
            geometry_tiers=list(phase.geometry_tiers),
            tier_weights=list(phase.tier_weights) if phase.tier_weights is not None else None,
            spawn=SpawnConfig(
                n_agents_range=phase.n_agents_range,
                shoulder_width_mean=base_config.spawn.shoulder_width_mean,
                shoulder_width_std=base_config.spawn.shoulder_width_std,
                chest_depth_mean=base_config.spawn.chest_depth_mean,
                chest_depth_std=base_config.spawn.chest_depth_std,
                preferred_speed_mean=base_config.spawn.preferred_speed_mean,
                preferred_speed_std=base_config.spawn.preferred_speed_std,
                min_body_dim=base_config.spawn.min_body_dim,
                min_speed=base_config.spawn.min_speed,
                max_speed=base_config.spawn.max_speed,
                min_spawn_separation=base_config.spawn.min_spawn_separation,
                max_spawn_attempts=base_config.spawn.max_spawn_attempts,
            ),
            solvability_mode=base_config.solvability_mode,
            max_unsolvable_fraction=base_config.max_unsolvable_fraction,
            max_regeneration_attempts=base_config.max_regeneration_attempts,
            obs=base_config.obs,
            action=base_config.action,
            reward=base_config.reward,
            dt=base_config.dt,
            contact_stiffness=base_config.contact_stiffness,
            contact_damping=base_config.contact_damping,
            velocity_damping=base_config.velocity_damping,
            max_steps=base_config.max_steps,
        )

    def state_dict(self) -> dict:
        """Serialise for checkpointing."""
        return {
            "current_phase_idx": self.current_phase_idx,
            "episodes_in_phase": self._episodes_in_phase,
            "total_episodes": self._total_episodes,
            "episode_history": [
                {
                    "goal_rate": s.goal_rate,
                    "n_agents": s.n_agents,
                    "episode_length": s.episode_length,
                    "mean_reward": s.mean_reward,
                }
                for s in self._episode_history
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from checkpoint."""
        self.current_phase_idx = state["current_phase_idx"]
        self._episodes_in_phase = state["episodes_in_phase"]
        self._total_episodes = state["total_episodes"]
        self._episode_history.clear()
        for s in state["episode_history"]:
            self._episode_history.append(EpisodeStats(**s))

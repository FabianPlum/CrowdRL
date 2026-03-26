"""Tests for curriculum manager."""

from __future__ import annotations

import pytest

from crowdrl_env.crowd_env import CrowdEnvConfig
from crowdrl_env.geometry_generator import GeometryTier

from crowdrl_train.config import CurriculumConfig, CurriculumPhase
from crowdrl_train.curriculum import CurriculumManager, EpisodeStats


def _make_stats(goal_rate: float = 0.8, n_agents: int = 10) -> EpisodeStats:
    return EpisodeStats(
        goal_rate=goal_rate,
        n_agents=n_agents,
        episode_length=100,
        mean_reward=5.0,
    )


@pytest.fixture
def simple_curriculum() -> CurriculumConfig:
    return CurriculumConfig(
        phases=(
            CurriculumPhase("easy", (GeometryTier.TIER_0,), (5, 10), 0.7),
            CurriculumPhase("hard", (GeometryTier.TIER_1,), (10, 30), 0.6),
            CurriculumPhase("full", (GeometryTier.TIER_0, GeometryTier.TIER_1), (20, 50), 0.0),
        ),
        metric_window=10,
        min_episodes_per_phase=20,
    )


class TestCurriculumManager:
    def test_starts_at_phase_0(self, simple_curriculum: CurriculumConfig):
        mgr = CurriculumManager(simple_curriculum)
        assert mgr.current_phase_idx == 0
        assert mgr.current_phase.name == "easy"

    def test_does_not_advance_before_min_episodes(self, simple_curriculum: CurriculumConfig):
        """Should not advance before min_episodes_per_phase even if threshold met."""
        mgr = CurriculumManager(simple_curriculum)
        for _ in range(19):
            advanced = mgr.report_episode(_make_stats(goal_rate=0.9))
            assert not advanced
        assert mgr.current_phase_idx == 0

    def test_advances_when_threshold_met(self, simple_curriculum: CurriculumConfig):
        """Should advance after min_episodes + metric_window with sufficient goal rate."""
        mgr = CurriculumManager(simple_curriculum)
        # Fill min episodes
        for _ in range(20):
            mgr.report_episode(_make_stats(goal_rate=0.8))
        assert mgr.current_phase_idx == 1  # Should have advanced

    def test_does_not_advance_below_threshold(self, simple_curriculum: CurriculumConfig):
        """Should NOT advance if goal rate is below threshold."""
        mgr = CurriculumManager(simple_curriculum)
        for _ in range(50):
            mgr.report_episode(_make_stats(goal_rate=0.3))
        assert mgr.current_phase_idx == 0

    def test_terminal_phase_never_advances(self, simple_curriculum: CurriculumConfig):
        """The final phase (threshold=0.0) should never advance."""
        mgr = CurriculumManager(simple_curriculum)
        # Force to terminal phase
        mgr.current_phase_idx = 2
        for _ in range(100):
            advanced = mgr.report_episode(_make_stats(goal_rate=1.0))
            assert not advanced
        assert mgr.current_phase_idx == 2

    def test_make_env_config_applies_overrides(self, simple_curriculum: CurriculumConfig):
        """make_env_config should override tiers and agent count."""
        mgr = CurriculumManager(simple_curriculum)
        base = CrowdEnvConfig()
        env_config = mgr.make_env_config(base)

        assert env_config.geometry_tiers == [GeometryTier.TIER_0]
        assert env_config.spawn.n_agents_range == (5, 10)
        # Physics/obs/reward should be inherited from base
        assert env_config.dt == base.dt
        assert env_config.obs == base.obs

    def test_rolling_goal_rate(self, simple_curriculum: CurriculumConfig):
        mgr = CurriculumManager(simple_curriculum)
        for _ in range(5):
            mgr.report_episode(_make_stats(goal_rate=0.8))
        for _ in range(5):
            mgr.report_episode(_make_stats(goal_rate=0.4))
        # Average of 5×0.8 + 5×0.4 = 0.6
        assert mgr.rolling_goal_rate == pytest.approx(0.6)

    def test_state_dict_roundtrip(self, simple_curriculum: CurriculumConfig):
        """State dict should preserve phase and history."""
        mgr = CurriculumManager(simple_curriculum)
        for _ in range(25):
            mgr.report_episode(_make_stats(goal_rate=0.8))

        state = mgr.state_dict()
        restored = CurriculumManager(simple_curriculum)
        restored.load_state_dict(state)

        assert restored.current_phase_idx == mgr.current_phase_idx
        assert restored._total_episodes == mgr._total_episodes

"""Tests for the torch episode factory's navmesh waypoint handling.

Covers the single-source-of-truth waypoint cap, the "final stored waypoint is
the goal" invariant the goal-direction ablation depends on, and the over-cap
regeneration guard (we regenerate the geometry rather than silently truncate a
path that would exceed the cap and strand the agent short of its goal).
"""

import numpy as np
import pytest

from crowdrl_core.action import ActionConfig
from crowdrl_core.observation import ObsConfig
from crowdrl_env.crowd_env import CrowdEnvConfig
from crowdrl_env.geometry_generator import GeometryConfig, GeometryTier
from crowdrl_env.spawner import SpawnConfig
from crowdrl_torch.episode_factory import make_episode_factory
from crowdrl_torch.types import EnvConfig


def _config(cap: int, tier: GeometryTier) -> CrowdEnvConfig:
    return CrowdEnvConfig(
        geometry=GeometryConfig(tier=tier),
        geometry_tiers=[tier],
        spawn=SpawnConfig(n_agents_range=(12, 18)),
        obs=ObsConfig(use_navmesh=True, navmesh_max_waypoints=cap),
        action=ActionConfig(),
    )


class TestWaypointCapUnification:
    """EnvConfig.max_waypoints is derived from ObsConfig.navmesh_max_waypoints."""

    def test_cap_is_single_sourced(self):
        cfg = CrowdEnvConfig(obs=ObsConfig(use_navmesh=True, navmesh_max_waypoints=777))
        ec = EnvConfig.from_crowd_env_config(cfg, max_agents=16)
        assert ec.max_waypoints == 777

    def test_default_cap_is_1024(self):
        cfg = CrowdEnvConfig(obs=ObsConfig(use_navmesh=True))
        ec = EnvConfig.from_crowd_env_config(cfg, max_agents=16)
        assert ec.max_waypoints == 1024


class TestFinalWaypointIsGoal:
    """Every stored funnel path must terminate at the agent's global goal."""

    def test_last_waypoint_equals_goal(self):
        factory = make_episode_factory(_config(1024, GeometryTier.TIER_2))
        checked = 0
        for seed in range(6):
            ep = factory(seed)
            wp, nwp, goals = ep["waypoints"], ep["n_waypoints"], ep["goal_positions"]
            for i in range(len(goals)):
                if nwp[i] > 0:
                    last = wp[i, nwp[i] - 1]
                    assert np.allclose(last, goals[i], atol=1e-6)
                    checked += 1
        assert checked > 0


class TestOverCapRegeneration:
    """A path exceeding the cap triggers regeneration, never a truncated path."""

    def test_returned_episodes_are_never_truncated(self):
        # A modest cap on a complex tier (Tier 3A reaches ~11 waypoints) means
        # some geometries exceed it. Any episode the factory RETURNS must have
        # regenerated to a within-cap geometry, so its last stored waypoint is
        # still the goal. If truncation were reintroduced, a returned episode
        # containing an over-cap path would have last != goal and fail here.
        factory = make_episode_factory(_config(6, GeometryTier.TIER_3A))
        checked = 0
        for seed in range(8):
            try:
                ep = factory(seed)
            except RuntimeError:
                # All regeneration attempts exhausted -- acceptable; nothing to
                # check for this seed.
                continue
            wp, nwp, goals = ep["waypoints"], ep["n_waypoints"], ep["goal_positions"]
            assert int(nwp.max(initial=0)) <= 6  # never stored more than the cap
            for i in range(len(goals)):
                if nwp[i] > 0:
                    assert np.allclose(wp[i, nwp[i] - 1], goals[i], atol=1e-6)
                    checked += 1
        assert checked > 0

    def test_tiny_cap_exhausts_regeneration_attempts(self):
        # cap=1 on a complex tier: essentially every geometry has a multi-corner
        # path, so all regeneration attempts are spent and the factory raises --
        # proving the over-cap branch routes into the regeneration loop rather
        # than returning a truncated episode.
        factory = make_episode_factory(_config(1, GeometryTier.TIER_3A))
        with pytest.raises(RuntimeError):
            for seed in range(20):
                factory(seed)


def _config_jps(flag: bool) -> CrowdEnvConfig:
    return CrowdEnvConfig(
        geometry=GeometryConfig(tier=GeometryTier.TIER_2),
        geometry_tiers=[GeometryTier.TIER_2],
        spawn=SpawnConfig(n_agents_range=(12, 18)),
        obs=ObsConfig(use_navmesh=True, use_jupedsim_style_routing=flag),
        action=ActionConfig(),
    )


class TestJupedsimStyleRoutingPrecompute:
    """Flag on: stored paths use the router's fixed 0.2 m inset, not body radius."""

    def test_same_seed_stored_paths_differ_only_by_inset(self):
        on = make_episode_factory(_config_jps(True))
        off = make_episode_factory(_config_jps(False))
        found_difference = False
        for seed in range(8):
            ep_on, ep_off = on(seed), off(seed)
            # The flag consumes no RNG, so the episode itself is identical...
            assert np.allclose(ep_on["positions"], ep_off["positions"])
            assert np.allclose(ep_on["goal_positions"], ep_off["goal_positions"])
            # ...but bent paths store different funnel corners (0.2 vs ~0.23).
            nwp_on, nwp_off = ep_on["n_waypoints"], ep_off["n_waypoints"]
            for i in range(len(nwp_on)):
                if nwp_on[i] >= 2 and nwp_off[i] >= 2:
                    if not np.allclose(ep_on["waypoints"][i, 0], ep_off["waypoints"][i, 0]):
                        found_difference = True
            if found_difference:
                break
        assert found_difference, "no bent path differed -- the inset switch had no effect"

    def test_final_waypoint_still_goal_under_flag(self):
        factory = make_episode_factory(_config_jps(True))
        checked = 0
        for seed in range(4):
            ep = factory(seed)
            wp, nwp, goals = ep["waypoints"], ep["n_waypoints"], ep["goal_positions"]
            for i in range(len(goals)):
                if nwp[i] > 0:
                    assert np.allclose(wp[i, nwp[i] - 1], goals[i], atol=1e-6)
                    checked += 1
        assert checked > 0

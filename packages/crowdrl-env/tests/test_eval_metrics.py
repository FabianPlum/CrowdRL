"""Tests for crowdrl_env.eval_metrics.

Hand-built EpisodeFrames with known-answer geometry verify that the metrics
helper is wired correctly to the reused crowdrl-core collision/wall routines.
"""

import numpy as np

from crowdrl_env.eval_metrics import aggregate_metrics, compute_episode_metrics
from crowdrl_env.visualiser import EpisodeFrames


def test_speed_goal_and_path_efficiency():
    # One agent walking +x at a constant 0.1 m / 0.1 s = 1.0 m/s, straight line.
    pos = np.array([[[0.0, 0.0]], [[0.1, 0.0]], [[0.2, 0.0]]])  # (3, 1, 2)
    frames = EpisodeFrames(
        positions=pos,
        torso_orientations=np.zeros((3, 1)),
        head_orientations=np.zeros((3, 1)),
        shoulder_widths=np.array([0.22]),
        chest_depths=np.array([0.12]),
        goal_positions=np.array([[1.0, 0.0]]),
        active_masks=np.ones((3, 1), dtype=bool),
        reached_goal=np.array([True]),
        preferred_speeds=np.array([0.5]),  # agent moves at 2x its preferred speed
        dt=0.1,
    )
    m = compute_episode_metrics(frames)
    assert m["goal_rate"] == 1.0
    assert np.isclose(m["mean_speed"], 1.0)
    assert np.isclose(m["speed_over_preferred"], 2.0)
    assert m["frac_steps_above_preferred"] == 1.0
    assert np.isclose(m["path_efficiency"], 1.0)  # perfectly straight


def test_wall_contact_rate():
    # Stationary agent sitting on a wall segment -> distance 0 < body radius.
    pos = np.array([[[0.0, 0.0]], [[0.0, 0.0]]])  # (2, 1, 2)
    walls = np.array([[[-1.0, 0.0], [1.0, 0.0]]])  # segment through the agent
    frames = EpisodeFrames(
        positions=pos,
        torso_orientations=np.zeros((2, 1)),
        head_orientations=np.zeros((2, 1)),
        shoulder_widths=np.array([0.22]),
        chest_depths=np.array([0.12]),
        goal_positions=np.array([[5.0, 5.0]]),
        active_masks=np.ones((2, 1), dtype=bool),
        reached_goal=np.array([False]),
        walls=walls,
        dt=0.1,
    )
    m = compute_episode_metrics(frames)
    assert m["wall_contact_rate"] == 1.0
    assert m["wall_proximity_rate"] == 1.0


def test_no_wall_metrics_without_geometry():
    # No walls and no polygon -> wall metrics are omitted, not guessed.
    pos = np.array([[[0.0, 0.0]], [[0.1, 0.0]]])
    frames = EpisodeFrames(
        positions=pos,
        torso_orientations=np.zeros((2, 1)),
        head_orientations=np.zeros((2, 1)),
        shoulder_widths=np.array([0.22]),
        chest_depths=np.array([0.12]),
        goal_positions=np.array([[5.0, 0.0]]),
        active_masks=np.ones((2, 1), dtype=bool),
        reached_goal=np.array([False]),
        dt=0.1,
    )
    m = compute_episode_metrics(frames)
    assert "wall_contact_rate" not in m
    assert "speed_over_preferred" not in m  # no preferred_speeds given


def test_agent_collision_rate():
    # Two agents overlapping (5 cm apart, ~0.12-0.22 m semi-axes) every frame.
    pos = np.array([[[0.0, 0.0], [0.05, 0.0]], [[0.0, 0.0], [0.05, 0.0]]])  # (2, 2, 2)
    frames = EpisodeFrames(
        positions=pos,
        torso_orientations=np.zeros((2, 2)),
        head_orientations=np.zeros((2, 2)),
        shoulder_widths=np.array([0.22, 0.22]),
        chest_depths=np.array([0.12, 0.12]),
        goal_positions=np.array([[5.0, 0.0], [5.0, 0.0]]),
        active_masks=np.ones((2, 2), dtype=bool),
        reached_goal=np.array([False, False]),
        dt=0.1,
    )
    m = compute_episode_metrics(frames)
    # Both agents are involved in a collision in every frame -> rate ~1.0.
    assert m["agent_collision_rate"] > 0.5


def test_aggregate_metrics_means_over_present_keys():
    agg = aggregate_metrics([{"goal_rate": 1.0, "mean_speed": 2.0}, {"goal_rate": 0.0}])
    assert agg["goal_rate"] == 0.5
    assert agg["mean_speed"] == 2.0  # present in only one episode
    assert aggregate_metrics([]) == {}


def test_freeze_and_stuck_for_frozen_agent():
    # Agent 0 walks +x at 1 m/s and reaches goal; agent 1 sits still and never
    # arrives -> half the active agent-steps are frozen, and the frozen,
    # never-arriving agent counts as stuck.
    pos = np.array(
        [
            [[0.0, 0.0], [5.0, 0.0]],
            [[0.1, 0.0], [5.0, 0.0]],
            [[0.2, 0.0], [5.0, 0.0]],
            [[0.3, 0.0], [5.0, 0.0]],
        ]
    )  # (4, 2, 2)
    frames = EpisodeFrames(
        positions=pos,
        torso_orientations=np.zeros((4, 2)),
        head_orientations=np.zeros((4, 2)),
        shoulder_widths=np.array([0.22, 0.22]),
        chest_depths=np.array([0.12, 0.12]),
        goal_positions=np.array([[0.3, 0.0], [99.0, 0.0]]),
        active_masks=np.ones((4, 2), dtype=bool),
        reached_goal=np.array([True, False]),
        dt=0.1,
    )
    m = compute_episode_metrics(frames)
    assert np.isclose(m["freeze_rate"], 0.5)
    assert np.isclose(m["stuck_agent_frac"], 0.5)


def test_moving_finished_agent_has_no_freeze_or_stuck():
    pos = np.array([[[0.0, 0.0]], [[0.1, 0.0]], [[0.2, 0.0]]])  # (3, 1, 2)
    frames = EpisodeFrames(
        positions=pos,
        torso_orientations=np.zeros((3, 1)),
        head_orientations=np.zeros((3, 1)),
        shoulder_widths=np.array([0.22]),
        chest_depths=np.array([0.12]),
        goal_positions=np.array([[1.0, 0.0]]),
        active_masks=np.ones((3, 1), dtype=bool),
        reached_goal=np.array([True]),
        dt=0.1,
    )
    m = compute_episode_metrics(frames)
    assert m["freeze_rate"] == 0.0
    # All agents reached goal -> no stuck candidates -> key omitted, not guessed.
    assert "stuck_agent_frac" not in m


def test_stationary_unfinished_agent_is_fully_frozen_and_stuck():
    pos = np.array([[[2.0, 2.0]], [[2.0, 2.0]], [[2.0, 2.0]]])  # (3, 1, 2)
    frames = EpisodeFrames(
        positions=pos,
        torso_orientations=np.zeros((3, 1)),
        head_orientations=np.zeros((3, 1)),
        shoulder_widths=np.array([0.22]),
        chest_depths=np.array([0.12]),
        goal_positions=np.array([[9.0, 9.0]]),
        active_masks=np.ones((3, 1), dtype=bool),
        reached_goal=np.array([False]),
        dt=0.1,
    )
    m = compute_episode_metrics(frames)
    assert m["freeze_rate"] == 1.0
    assert m["stuck_agent_frac"] == 1.0

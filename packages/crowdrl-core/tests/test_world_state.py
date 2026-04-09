"""Tests for WorldState dataclass."""

import numpy as np
import pytest

from crowdrl_core.world_state import WorldState


class TestWorldState:
    def test_n_agents(self):
        ws = WorldState(
            positions=np.zeros((5, 2)),
            velocities=np.zeros((5, 2)),
            torso_orientations=np.zeros(5),
            head_orientations=np.zeros(5),
            shoulder_widths=np.full(5, 0.23),
            chest_depths=np.full(5, 0.15),
            masses=np.full(5, 80.0, dtype=np.float64),
            goal_positions=np.zeros((5, 2)),
        )
        assert ws.n_agents == 5

    def test_validate_consistent(self):
        ws = WorldState(
            positions=np.zeros((3, 2)),
            velocities=np.zeros((3, 2)),
            torso_orientations=np.zeros(3),
            head_orientations=np.zeros(3),
            shoulder_widths=np.full(3, 0.23),
            chest_depths=np.full(3, 0.15),
            masses=np.full(3, 80.0, dtype=np.float64),
            goal_positions=np.zeros((3, 2)),
        )
        ws.validate()  # Should not raise

    def test_validate_mismatched_velocities(self):
        ws = WorldState(
            positions=np.zeros((3, 2)),
            velocities=np.zeros((2, 2)),  # Wrong!
            torso_orientations=np.zeros(3),
            head_orientations=np.zeros(3),
            shoulder_widths=np.full(3, 0.23),
            chest_depths=np.full(3, 0.15),
            masses=np.full(3, 80.0, dtype=np.float64),
            goal_positions=np.zeros((3, 2)),
        )
        with pytest.raises(ValueError, match="velocities"):
            ws.validate()

    def test_validate_mismatched_orientations(self):
        ws = WorldState(
            positions=np.zeros((3, 2)),
            velocities=np.zeros((3, 2)),
            torso_orientations=np.zeros(4),  # Wrong!
            head_orientations=np.zeros(3),
            shoulder_widths=np.full(3, 0.23),
            chest_depths=np.full(3, 0.15),
            masses=np.full(3, 80.0, dtype=np.float64),
            goal_positions=np.zeros((3, 2)),
        )
        with pytest.raises(ValueError, match="torso_orientations"):
            ws.validate()

    def test_validate_active_mask(self):
        ws = WorldState(
            positions=np.zeros((3, 2)),
            velocities=np.zeros((3, 2)),
            torso_orientations=np.zeros(3),
            head_orientations=np.zeros(3),
            shoulder_widths=np.full(3, 0.23),
            chest_depths=np.full(3, 0.15),
            masses=np.full(3, 80.0, dtype=np.float64),
            goal_positions=np.zeros((3, 2)),
            active_mask=np.array([True, False]),  # Wrong size!
        )
        with pytest.raises(ValueError, match="active_mask"):
            ws.validate()

    def test_single_agent(self):
        ws = WorldState(
            positions=np.array([[1.0, 2.0]]),
            velocities=np.array([[0.5, 0.0]]),
            torso_orientations=np.array([0.0]),
            head_orientations=np.array([0.1]),
            shoulder_widths=np.array([0.23]),
            chest_depths=np.array([0.15]),
            masses=np.array([80.0]),
            goal_positions=np.array([[9.0, 2.0]]),
        )
        assert ws.n_agents == 1
        ws.validate()

    def test_large_population(self):
        n = 100
        ws = WorldState(
            positions=np.random.randn(n, 2),
            velocities=np.random.randn(n, 2),
            torso_orientations=np.random.randn(n),
            head_orientations=np.random.randn(n),
            shoulder_widths=np.full(n, 0.23),
            chest_depths=np.full(n, 0.15),
            masses=np.full(n, 80.0, dtype=np.float64),
            goal_positions=np.random.randn(n, 2),
        )
        assert ws.n_agents == 100
        ws.validate()

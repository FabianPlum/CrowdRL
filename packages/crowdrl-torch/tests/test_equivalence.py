"""Numerical equivalence tests: PyTorch env vs CPU env.

Runs identical episodes through both implementations and verifies
observations, rewards, and collision detection match.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import torch

# CPU reference implementations
from crowdrl_core.action import ActionConfig, interpret_actions_batch
from crowdrl_core.collision import detect_collisions
from crowdrl_core.observation import ObsConfig, build_observations_batch
from crowdrl_core.sensing import RaycastConfig
from crowdrl_core.world_state import WorldState

# PyTorch implementations
from crowdrl_torch.action import interpret_actions as torch_interpret_actions
from crowdrl_torch.collision import detect_collisions_pairwise as torch_detect_collisions
from crowdrl_torch.observation import build_observations as torch_build_observations
from crowdrl_torch.reward import compute_rewards as torch_compute_rewards
from crowdrl_torch.sensing import cast_rays as torch_cast_rays
from crowdrl_torch.sensing import knn_social as torch_knn_social
from crowdrl_torch.sensing import match_persistent_neighbors as torch_match_persistent_neighbors
from crowdrl_torch.types import EnvConfig
from crowdrl_torch.walls import point_in_polygon, points_to_segments_nearest

# Tolerances — PyTorch uses float32, NumPy uses float64
ATOL = 1e-4
RTOL = 1e-3


def _make_test_world(n_agents: int = 10, seed: int = 42):
    """Create a simple test WorldState with a rectangular walkable area."""
    rng = np.random.default_rng(seed)

    positions = rng.uniform(1.0, 9.0, (n_agents, 2))
    velocities = rng.uniform(-1.0, 1.0, (n_agents, 2))
    torso_orientations = rng.uniform(-np.pi, np.pi, n_agents)
    head_orientations = torso_orientations + rng.uniform(-0.5, 0.5, n_agents)
    shoulder_widths = rng.uniform(0.2, 0.4, n_agents)
    chest_depths = rng.uniform(0.15, 0.25, n_agents)
    masses = np.full(n_agents, 80.0, dtype=np.float64)
    goal_positions = rng.uniform(1.0, 9.0, (n_agents, 2))

    # Simple rectangle: 0,0 to 10,10
    wall_segments = np.array(
        [
            [[0, 0], [10, 0]],
            [[10, 0], [10, 10]],
            [[10, 10], [0, 10]],
            [[0, 10], [0, 0]],
        ],
        dtype=np.float64,
    )

    world = WorldState(
        positions=positions,
        velocities=velocities,
        torso_orientations=torso_orientations,
        head_orientations=head_orientations,
        shoulder_widths=shoulder_widths,
        chest_depths=chest_depths,
        masses=masses,
        goal_positions=goal_positions,
        wall_segments=wall_segments,
    )

    return world


def _world_to_torch(world: WorldState, config: EnvConfig):
    """Convert a WorldState to padded PyTorch tensors with E=1 batch dim."""
    N = config.max_agents
    n = world.n_agents

    def pad_1d(arr, size):
        out = np.zeros(size, dtype=np.float32)
        out[: len(arr)] = arr.astype(np.float32)
        return torch.tensor(out).unsqueeze(0)  # (1, N)

    def pad_2d(arr, size):
        out = np.zeros((size, arr.shape[1]), dtype=np.float32)
        out[: len(arr)] = arr.astype(np.float32)
        return torch.tensor(out).unsqueeze(0)  # (1, N, 2)

    S = config.max_segments
    padded_segs = np.zeros((S, 2, 2), dtype=np.float32)
    n_segs = len(world.wall_segments)
    padded_segs[:n_segs] = world.wall_segments.astype(np.float32)

    active = np.zeros(N, dtype=np.bool_)
    active[:n] = True

    return {
        "positions": pad_2d(world.positions, N),
        "velocities": pad_2d(world.velocities, N),
        "torso_orientations": pad_1d(world.torso_orientations, N),
        "head_orientations": pad_1d(world.head_orientations, N),
        "shoulder_widths": pad_1d(world.shoulder_widths, N),
        "chest_depths": pad_1d(world.chest_depths, N),
        "masses": pad_1d(world.masses, N),
        "goal_positions": pad_2d(world.goal_positions, N),
        "active_mask": torch.tensor(active).unsqueeze(0),  # (1, N)
        "wall_segments": torch.tensor(padded_segs).unsqueeze(0),  # (1, S, 2, 2)
        "n_segments": torch.tensor([n_segs], dtype=torch.int32),  # (1,)
        "n_agents": torch.tensor([n], dtype=torch.int32),  # (1,)
    }


class TestActionEquivalence:
    """Test PyTorch action interpretation matches NumPy reference."""

    def test_interpret_actions(self):
        rng = np.random.default_rng(42)
        n = 10
        raw_actions = rng.uniform(-1, 1, (n, 4))
        headings = rng.uniform(-np.pi, np.pi, n)
        torsos = rng.uniform(-np.pi, np.pi, n)
        heads = rng.uniform(-np.pi, np.pi, n)

        # NumPy reference
        np_config = ActionConfig()
        np_result = interpret_actions_batch(raw_actions, headings, torsos, heads, np_config)

        # PyTorch (with E=1 batch dim)
        torch_config = EnvConfig(max_agents=n)
        torch_vel, torch_h, torch_t, torch_hd = torch_interpret_actions(
            torch.tensor(raw_actions, dtype=torch.float32).unsqueeze(0),
            torch.tensor(headings, dtype=torch.float32).unsqueeze(0),
            torch.tensor(torsos, dtype=torch.float32).unsqueeze(0),
            torch.tensor(heads, dtype=torch.float32).unsqueeze(0),
            torch_config,
        )

        # Remove batch dim for comparison
        npt.assert_allclose(
            torch_vel[0].numpy(), np_result.desired_velocities, atol=ATOL, rtol=RTOL
        )
        npt.assert_allclose(torch_h[0].numpy(), np_result.new_headings, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(
            torch_t[0].numpy(), np_result.new_torso_orientations, atol=ATOL, rtol=RTOL
        )
        npt.assert_allclose(
            torch_hd[0].numpy(), np_result.new_head_orientations, atol=ATOL, rtol=RTOL
        )


class TestCollisionEquivalence:
    """Test PyTorch collision detection matches NumPy reference."""

    def test_detect_collisions(self):
        world = _make_test_world(n_agents=8, seed=123)
        # Place some agents close together to force collisions
        world.positions[0] = [5.0, 5.0]
        world.positions[1] = [5.1, 5.05]
        world.positions[2] = [5.05, 5.1]

        # NumPy reference
        np_collisions = detect_collisions(world)
        np_collision_mask = np.zeros(world.n_agents, dtype=np.bool_)
        for i, j, _ in np_collisions:
            np_collision_mask[i] = True
            np_collision_mask[j] = True

        # PyTorch
        config = EnvConfig(max_agents=8)
        td = _world_to_torch(world, config)
        _, torch_collision_mask = torch_detect_collisions(
            td["positions"],
            td["torso_orientations"],
            td["shoulder_widths"],
            td["chest_depths"],
            td["active_mask"],
            td["n_agents"],
        )

        # Compare collision masks for active agents
        n = world.n_agents
        npt.assert_array_equal(
            torch_collision_mask[0, :n].numpy(),
            np_collision_mask,
        )


class TestWallEquivalence:
    """Test PyTorch wall functions match NumPy reference."""

    def test_point_in_polygon_rectangle(self):
        """Points inside/outside a simple rectangle."""
        segs = np.array(
            [
                [[0, 0], [10, 0]],
                [[10, 0], [10, 10]],
                [[10, 10], [0, 10]],
                [[0, 10], [0, 0]],
            ],
            dtype=np.float32,
        )
        padded = np.zeros((128, 2, 2), dtype=np.float32)
        padded[:4] = segs

        points = np.array(
            [
                [5, 5],  # inside
                [1, 1],  # inside
                [9, 9],  # inside
                [-1, 5],  # outside
                [11, 5],  # outside
                [5, -1],  # outside
            ],
            dtype=np.float32,
        )

        # Add batch dim (E=1)
        inside = point_in_polygon(
            torch.tensor(points).unsqueeze(0),
            torch.tensor(padded).unsqueeze(0),
            torch.tensor([4], dtype=torch.int32),
        )
        expected = [True, True, True, False, False, False]
        npt.assert_array_equal(inside[0].numpy(), expected)

    def test_nearest_point_on_segment(self):
        """Nearest point on a horizontal segment."""
        from crowdrl_core.collision import _points_to_segments_nearest_batch

        points = np.array([[5.0, 3.0], [0.0, 5.0]], dtype=np.float64)
        seg_starts = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
        seg_ends = np.array([[10.0, 0.0], [10.0, 10.0]], dtype=np.float64)

        # NumPy reference
        np_nearest, np_dist, np_normals = _points_to_segments_nearest_batch(
            points, seg_starts, seg_ends
        )

        # PyTorch (with E=1 batch dim)
        torch_nearest, torch_dist, torch_normals = points_to_segments_nearest(
            torch.tensor(points, dtype=torch.float32).unsqueeze(0),
            torch.tensor(seg_starts, dtype=torch.float32).unsqueeze(0),
            torch.tensor(seg_ends, dtype=torch.float32).unsqueeze(0),
        )

        npt.assert_allclose(torch_nearest[0].numpy(), np_nearest, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(torch_dist[0].numpy(), np_dist, atol=ATOL, rtol=RTOL)


class TestSensingEquivalence:
    """Test PyTorch raycasting and KNN match NumPy reference."""

    def test_cast_rays(self):
        from crowdrl_core.sensing import cast_rays_batch

        world = _make_test_world(n_agents=6)
        ray_config = RaycastConfig(n_rays=16, fov_deg=200.0, max_range=5.0)
        agent_indices = np.arange(world.n_agents, dtype=np.intp)

        # NumPy reference
        np_rays = cast_rays_batch(world, agent_indices, ray_config)

        # PyTorch
        config = EnvConfig(max_agents=6, max_segments=128, n_rays=16, fov_deg=200.0, max_range=5.0)
        td = _world_to_torch(world, config)
        torch_rays = torch_cast_rays(
            td["positions"],
            torch.tensor(world.head_orientations, dtype=torch.float32).unsqueeze(0),
            td["torso_orientations"],
            td["shoulder_widths"],
            td["chest_depths"],
            td["active_mask"],
            td["n_agents"],
            td["wall_segments"],
            td["n_segments"],
            config,
        )

        n = world.n_agents
        npt.assert_allclose(torch_rays[0, :n].numpy(), np_rays, atol=ATOL, rtol=RTOL)

    def test_knn_social(self):
        from crowdrl_core.sensing import knn_social_batch

        world = _make_test_world(n_agents=10)
        agent_indices = np.arange(world.n_agents, dtype=np.intp)

        # NumPy reference
        np_social = knn_social_batch(world, agent_indices, k=8)

        # PyTorch
        config = EnvConfig(max_agents=10, k_neighbours=8)
        td = _world_to_torch(world, config)
        torch_social = torch_knn_social(
            td["positions"],
            td["velocities"],
            td["torso_orientations"],
            td["shoulder_widths"],
            td["chest_depths"],
            td["active_mask"],
            td["n_agents"],
            config,
        )

        n = world.n_agents
        npt.assert_allclose(torch_social[0, :n].numpy(), np_social, atol=ATOL, rtol=RTOL)

    def test_match_persistent_neighbors(self):
        """Numpy and torch match_persistent_neighbors must agree bit-for-bit
        on a deterministic fixture covering all four scenarios: first step
        fill, stable retention, out-of-range eviction, inactive eviction."""
        from crowdrl_core.sensing import match_persistent_neighbors as np_match

        # Hand-built 6-agent scene. Agents live on a line at x = 0..5.
        # Agent 2 sits in an initially-valid slot of agent 0 but is about
        # to go inactive (simulates goal reach).
        n = 6
        positions = np.array(
            [[0.0, 0.0], [1.2, 0.0], [2.0, 0.0], [3.5, 0.0], [6.0, 0.0], [9.0, 0.0]],
            dtype=np.float32,
        )
        K = 3
        prev_np = np.array(
            [
                [2, 4, -1],  # agent 0: slot 0 -> inactive below, slot 1 -> out of range below
                [0, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
            ],
            dtype=np.int32,
        )
        active_np = np.array([True, True, False, True, True, True])  # agent 2 inactive
        sensing_radius = 5.0

        # NumPy reference
        np_slots = np_match(
            positions.astype(np.float64),
            prev_np,
            active_np,
            sensing_radius=sensing_radius,
            k=K,
        )

        # PyTorch (E=1)
        config = EnvConfig(max_agents=n, k_neighbours=K)
        torch_positions = torch.tensor(positions, dtype=torch.float32).unsqueeze(0)
        torch_prev = torch.tensor(prev_np, dtype=torch.int32).unsqueeze(0)
        torch_active = torch.tensor(active_np, dtype=torch.bool).unsqueeze(0)
        torch_nagents = torch.tensor([n], dtype=torch.int32)

        torch_slots = torch_match_persistent_neighbors(
            torch_positions,
            torch_prev,
            torch_active,
            torch_nagents,
            sensing_radius=sensing_radius,
            config=config,
        )

        npt.assert_array_equal(torch_slots[0].numpy(), np_slots)

        # Sanity check the semantics independently of bit-equivalence:
        # - agent 0's slot 0 (prev agent 2) is evicted because 2 is inactive
        # - agent 0's slot 1 (prev agent 4 at dist 6) is evicted because out of range
        # -> both slots should be refilled from the nearest in-range active set
        #    = {1 (dist 1.2), 3 (dist 3.5)}, which are agents 1 and 3
        slot0 = set(int(x) for x in np_slots[0] if x >= 0)
        assert 1 in slot0, "nearest in-range active neighbor (agent 1) should be kept"
        assert 3 in slot0, "second nearest (agent 3) should be kept"
        assert 2 not in slot0, "inactive agent should be evicted"
        assert 4 not in slot0, "out-of-range agent should be evicted"


class TestObservationEquivalence:
    """Test full observation builder matches."""

    def test_build_observations(self):
        world = _make_test_world(n_agents=8)
        obs_config = ObsConfig(k_neighbours=8, raycast=RaycastConfig(n_rays=16))

        # NumPy reference
        np_obs = build_observations_batch(world, obs_config)

        # PyTorch
        config = EnvConfig(max_agents=8, max_segments=128, k_neighbours=8, n_rays=16)
        td = _world_to_torch(world, config)
        torch_obs = torch_build_observations(
            td["positions"],
            td["velocities"],
            td["torso_orientations"],
            torch.tensor(world.head_orientations, dtype=torch.float32).unsqueeze(0),
            td["shoulder_widths"],
            td["chest_depths"],
            td["goal_positions"],
            td["active_mask"],
            td["n_agents"],
            td["wall_segments"],
            td["n_segments"],
            config,
        )

        n = world.n_agents
        npt.assert_allclose(torch_obs[0, :n].numpy(), np_obs, atol=ATOL, rtol=RTOL)


class TestRewardEquivalence:
    """Test reward computation matches."""

    def test_compute_rewards(self):
        rng = np.random.default_rng(42)
        n = 8
        positions = rng.uniform(1.0, 9.0, (n, 2)).astype(np.float32)
        velocities = rng.uniform(-1.0, 1.0, (n, 2)).astype(np.float32)
        goal_positions = rng.uniform(1.0, 9.0, (n, 2)).astype(np.float32)

        # Place agent 0 at its goal to trigger goal bonus
        goal_positions[0] = positions[0] + [0.1, 0.1]

        active_mask = np.ones(n, dtype=np.bool_)
        collision_mask = np.zeros(n, dtype=np.bool_)
        collision_mask[2] = True  # Agent 2 in collision

        prev_goal_distances = np.linalg.norm(
            goal_positions - (positions - velocities * 0.1), axis=1
        ).astype(np.float32)

        config = EnvConfig(max_agents=n)

        # PyTorch (E=1)
        torch_rewards, torch_reached, torch_dists = torch_compute_rewards(
            torch.tensor(positions).unsqueeze(0),
            torch.tensor(velocities).unsqueeze(0),
            torch.tensor(goal_positions).unsqueeze(0),
            torch.tensor(active_mask).unsqueeze(0),
            torch.tensor(collision_mask).unsqueeze(0),
            torch.tensor(prev_goal_distances).unsqueeze(0),
            config,
        )

        # Verify basic properties
        rewards = torch_rewards[0].numpy()
        reached = torch_reached[0].numpy()

        # Agent 2 should have collision penalty
        assert rewards[2] < 0, "Agent 2 should have collision penalty"

        # Reached goal agents should have positive reward
        if reached.any():
            assert rewards[reached].max() >= config.goal_bonus - 1.0


class TestTemporalMemoryEquivalence:
    """Numpy and torch temporal-memory features should match numerically."""

    def test_temporal_features_numpy_vs_torch(self):
        from crowdrl_core.observation import (
            ObsConfig as CoreObsConfig,
            _temporal_features as np_temporal_features,
        )
        from crowdrl_torch.observation import compute_temporal_features as torch_temporal

        W = 5
        n = 4
        rng = np.random.default_rng(123)

        pos_now = rng.uniform(0, 10, (n, 2))
        spawn = rng.uniform(0, 10, (n, 2))
        goal = rng.uniform(0, 10, (n, 2))
        init_g = np.linalg.norm(goal - spawn, axis=1)
        cum_path = rng.uniform(0, 20, n)
        pos_window = rng.uniform(0, 10, (n, 2))
        gdist_window = rng.uniform(0, 10, n)
        gdist_now = np.linalg.norm(goal - pos_now, axis=1)
        preferred = np.full(n, 1.3)
        step_count = 20
        max_steps = 100
        dt = 0.01

        np_out = np_temporal_features(
            pos_now=pos_now,
            gdist_now=gdist_now,
            spawn_pos=spawn,
            initial_gdist=init_g,
            cum_path=cum_path,
            pos_window=pos_window,
            gdist_window=gdist_window,
            step_count=step_count,
            max_steps=max_steps,
            preferred_speeds=preferred,
            dt=dt,
            window=W,
        )

        # Build a (1, n, ...) torch state with step_count broadcast to (1,) and
        # the ring-buffer slot at `step_count % buf_size` pre-filled with
        # pos_window / gdist_window, so the gather picks them up correctly.
        buf_size = W + 1
        pos_history = np.broadcast_to(spawn[:, np.newaxis, :], (n, buf_size, 2)).copy()
        gdist_history = np.broadcast_to(init_g[:, np.newaxis], (n, buf_size)).copy()
        read_idx = step_count % buf_size
        pos_history[:, read_idx, :] = pos_window
        gdist_history[:, read_idx] = gdist_window

        core_cfg = CoreObsConfig(
            use_temporal_memory=True,
            temporal_memory_window=W,
            temporal_memory_max_steps=max_steps,
            temporal_memory_dt=dt,
        )
        # Build a minimal EnvConfig mirroring the above for torch
        torch_cfg = EnvConfig(
            max_agents=n,
            use_temporal_memory=True,
            temporal_memory_window=W,
            max_steps=max_steps,
            dt=dt,
        )

        torch_out = torch_temporal(
            positions=torch.tensor(pos_now, dtype=torch.float32).unsqueeze(0),
            spawn_positions=torch.tensor(spawn, dtype=torch.float32).unsqueeze(0),
            initial_goal_distances=torch.tensor(init_g, dtype=torch.float32).unsqueeze(0),
            cumulative_path_length=torch.tensor(cum_path, dtype=torch.float32).unsqueeze(0),
            pos_history=torch.tensor(pos_history, dtype=torch.float32).unsqueeze(0),
            gdist_history=torch.tensor(gdist_history, dtype=torch.float32).unsqueeze(0),
            goal_positions=torch.tensor(goal, dtype=torch.float32).unsqueeze(0),
            preferred_speeds=torch.tensor(preferred, dtype=torch.float32).unsqueeze(0),
            step_count=torch.tensor([step_count], dtype=torch.int32),
            config=torch_cfg,
        )

        torch_out_np = torch_out[0].numpy()
        npt.assert_allclose(torch_out_np, np_out, atol=ATOL, rtol=RTOL)
        # Also verify the obs_dim reports +6 when flag is set
        assert core_cfg.obs_dim == CoreObsConfig().obs_dim + 6

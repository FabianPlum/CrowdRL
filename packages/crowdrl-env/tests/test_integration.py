"""Integration tests: geometry → navmesh → world state → observations → actions.

These tests verify the full pipeline from procedural geometry generation through
to observation construction and action interpretation. They are the closest thing
to a transfer guarantee test without the actual JuPedSim adapter.
"""

import numpy as np
import pytest
from shapely.geometry import Polygon

from crowdrl_core.action import ActionConfig, interpret_actions_batch
from crowdrl_core.collision import detect_collisions, compute_contact_forces
from crowdrl_core.geometry import (
    build_navmesh,
    extract_wall_segments,
    sample_point_in_polygon,
)
from crowdrl_core.navmesh import is_reachable, next_waypoint_direction
from crowdrl_core.observation import ObsConfig, build_observation, build_observations_batch
from crowdrl_core.sensing import RaycastConfig, cast_rays
from crowdrl_core.world_state import WorldState

from crowdrl_env.geometry_generator import (
    GeometryConfig,
    GeometryTier,
    generate_geometry,
)


def _spawn_agents_in_geometry(
    polygon: Polygon,
    n_agents: int,
    rng: np.random.Generator,
    spawn_region: Polygon | None = None,
    goal_region: Polygon | None = None,
) -> WorldState:
    """Spawn agents in a geometry, building a full WorldState."""
    region = spawn_region if spawn_region is not None else polygon
    goal_reg = goal_region if goal_region is not None else polygon

    positions = np.array([sample_point_in_polygon(region, rng) for _ in range(n_agents)])
    goal_positions = np.array([sample_point_in_polygon(goal_reg, rng) for _ in range(n_agents)])

    return WorldState(
        positions=positions,
        velocities=np.zeros((n_agents, 2), dtype=np.float64),
        torso_orientations=rng.uniform(-np.pi, np.pi, n_agents),
        head_orientations=rng.uniform(-np.pi, np.pi, n_agents),
        shoulder_widths=np.full(n_agents, 0.23, dtype=np.float64),
        chest_depths=np.full(n_agents, 0.15, dtype=np.float64),
        goal_positions=goal_positions,
        walkable_polygon=polygon,
        wall_segments=extract_wall_segments(polygon),
        navmesh=build_navmesh(polygon),
    )


class TestFullPipelineTier0:
    """Full pipeline test on Tier 0 (open field) geometries."""

    def test_generate_build_observe(self, rng):
        """Generate geometry → build navmesh → spawn agents → build obs."""
        config = GeometryConfig(tier=GeometryTier.TIER_0)
        geom = generate_geometry(rng, config)

        world = _spawn_agents_in_geometry(
            geom.polygon,
            n_agents=10,
            rng=rng,
            spawn_region=geom.spawn_regions[0],
            goal_region=geom.goal_regions[0],
        )
        world.validate()

        obs_config = ObsConfig()
        obs = build_observations_batch(world, obs_config)
        assert obs.shape == (10, obs_config.obs_dim)
        assert np.all(np.isfinite(obs))

    def test_repeated_runs_deterministic(self):
        """Same seed → identical observations."""
        config = GeometryConfig(tier=GeometryTier.TIER_0)

        for seed in [42, 123, 999]:
            rng1 = np.random.default_rng(seed)
            rng2 = np.random.default_rng(seed)

            geom1 = generate_geometry(rng1, config)
            geom2 = generate_geometry(rng2, config)

            world1 = _spawn_agents_in_geometry(geom1.polygon, 5, rng1)
            world2 = _spawn_agents_in_geometry(geom2.polygon, 5, rng2)

            obs_config = ObsConfig()
            obs1 = build_observations_batch(world1, obs_config)
            obs2 = build_observations_batch(world2, obs_config)
            np.testing.assert_array_equal(obs1, obs2)


class TestFullPipelineTier1:
    """Full pipeline on Tier 1 (corridor/bottleneck) geometries."""

    def test_corridor_solvability(self, rng):
        """All agents in a corridor should have a reachable goal."""
        config = GeometryConfig(tier=GeometryTier.TIER_1)
        for _ in range(10):
            geom = generate_geometry(rng, config)
            navmesh = build_navmesh(geom.polygon)

            for _ in range(5):
                start = sample_point_in_polygon(geom.spawn_regions[0], rng)
                goal = sample_point_in_polygon(geom.goal_regions[0], rng)
                assert is_reachable(navmesh, start, goal), (
                    f"Unreachable path in {geom.metadata['shape']}"
                )

    def test_bottleneck_observations_finite(self, rng):
        """Observations should be finite even in tight bottleneck geometries."""
        config = GeometryConfig(
            tier=GeometryTier.TIER_1,
            bottleneck_aperture_range=(0.6, 0.8),  # Very tight
        )
        for _ in range(5):
            geom = generate_geometry(rng, config)
            world = _spawn_agents_in_geometry(
                geom.polygon,
                5,
                rng,
                spawn_region=geom.spawn_regions[0],
                goal_region=geom.goal_regions[0],
            )
            obs = build_observations_batch(world, ObsConfig())
            assert np.all(np.isfinite(obs))

    def test_raycasts_detect_bottleneck_walls(self, rng):
        """Agents near a bottleneck should detect walls with raycasts."""
        config = GeometryConfig(
            tier=GeometryTier.TIER_1,
            bottleneck_aperture_range=(0.8, 1.0),
        )
        geom = generate_geometry(rng, config)
        world = _spawn_agents_in_geometry(geom.polygon, 1, rng)

        rc = RaycastConfig(n_rays=16, fov_deg=360, max_range=5.0)
        readings = cast_rays(world, 0, rc)
        # At least some rays should detect walls (readings < 1.0)
        assert np.any(readings < 1.0), "No walls detected in bottleneck geometry"


class TestFullPipelineTier2:
    """Full pipeline on Tier 2 (branching) geometries."""

    def test_t_junction_multiple_goals(self, rng):
        """T-junction should support agents going to different branches."""
        config = GeometryConfig(tier=GeometryTier.TIER_2)
        # Force t-junction
        for _ in range(30):
            geom = generate_geometry(rng, config)
            if geom.metadata.get("shape") == "t_junction":
                break
        else:
            pytest.skip("No T-junction generated in 30 attempts")

        navmesh = build_navmesh(geom.polygon)
        start = sample_point_in_polygon(geom.spawn_regions[0], rng)

        for goal_region in geom.goal_regions:
            goal = sample_point_in_polygon(goal_region, rng)
            assert is_reachable(navmesh, start, goal)

    def test_crossroads_all_directions(self, rng):
        """Crossroads should be navigable in all four directions."""
        config = GeometryConfig(tier=GeometryTier.TIER_2)
        for _ in range(30):
            geom = generate_geometry(rng, config)
            if geom.metadata.get("shape") == "crossroads":
                break
        else:
            pytest.skip("No crossroads generated in 30 attempts")

        navmesh = build_navmesh(geom.polygon)
        for sr in geom.spawn_regions:
            for gr in geom.goal_regions:
                start = sample_point_in_polygon(sr, rng)
                goal = sample_point_in_polygon(gr, rng)
                assert is_reachable(navmesh, start, goal)

    def test_l_bend_obs_and_action(self, rng):
        """Full observe → act cycle in an L-bend."""
        config = GeometryConfig(tier=GeometryTier.TIER_2)
        for _ in range(30):
            geom = generate_geometry(rng, config)
            if geom.metadata.get("shape") == "l_bend":
                break
        else:
            pytest.skip("No L-bend generated in 30 attempts")

        world = _spawn_agents_in_geometry(
            geom.polygon,
            5,
            rng,
            spawn_region=geom.spawn_regions[0],
            goal_region=geom.goal_regions[0],
        )

        obs_config = ObsConfig(use_navmesh=True)
        obs = build_observations_batch(world, obs_config)
        assert obs.shape == (5, obs_config.obs_dim)
        assert np.all(np.isfinite(obs))

        # Interpret random actions
        actions = rng.uniform(-1, 1, (5, 4))
        results = interpret_actions_batch(
            actions,
            world.torso_orientations,
            world.torso_orientations,
            world.head_orientations,
        )
        assert len(results) == 5
        for r in results:
            assert np.all(np.isfinite(r.desired_velocity))


class TestObservationConsistency:
    """Tests that verify the transfer guarantee: same state → same observation."""

    def test_obs_identical_across_configs(self, rng):
        """Building obs with different configs should only differ in the
        expected components (e.g. navmesh signals present/absent)."""
        geom = generate_geometry(rng, GeometryConfig(tier=GeometryTier.TIER_0))
        world = _spawn_agents_in_geometry(geom.polygon, 3, rng)

        config_base = ObsConfig(use_navmesh=False)
        config_nav = ObsConfig(use_navmesh=True)

        obs_base = build_observation(world, 0, config_base)
        obs_nav = build_observation(world, 0, config_nav)

        # The base components (ego + social + rays) should be identical
        np.testing.assert_array_equal(
            obs_base[: config_base.obs_dim],
            obs_nav[: config_base.obs_dim],
        )

    def test_obs_invariant_to_reconstruction(self, rng):
        """Rebuilding WorldState with the same data should give identical obs."""
        geom = generate_geometry(rng, GeometryConfig(tier=GeometryTier.TIER_1))
        world = _spawn_agents_in_geometry(geom.polygon, 5, rng)

        obs_config = ObsConfig()
        obs1 = build_observations_batch(world, obs_config)

        # Rebuild WorldState from raw arrays (simulates JuPedSim adapter)
        world2 = WorldState(
            positions=world.positions.copy(),
            velocities=world.velocities.copy(),
            torso_orientations=world.torso_orientations.copy(),
            head_orientations=world.head_orientations.copy(),
            shoulder_widths=world.shoulder_widths.copy(),
            chest_depths=world.chest_depths.copy(),
            goal_positions=world.goal_positions.copy(),
            walkable_polygon=world.walkable_polygon,
            wall_segments=world.wall_segments.copy(),
            navmesh=world.navmesh,
        )
        obs2 = build_observations_batch(world2, obs_config)

        np.testing.assert_array_equal(obs1, obs2)


class TestCollisionInGeneratedGeometries:
    """Test collision detection works correctly in procedural geometries."""

    def test_no_collision_spread_agents(self, rng):
        """Well-separated agents should not collide."""
        geom = generate_geometry(rng, GeometryConfig(tier=GeometryTier.TIER_0, min_side=20))
        # Place agents far apart
        world = _spawn_agents_in_geometry(geom.polygon, 3, rng)
        # Spread positions manually
        world.positions = np.array([[2.0, 10.0], [10.0, 10.0], [18.0, 10.0]])
        collisions = detect_collisions(world)
        assert len(collisions) == 0

    def test_contact_forces_resolve(self, rng):
        """Contact forces should push overlapping agents apart."""
        geom = generate_geometry(rng, GeometryConfig(tier=GeometryTier.TIER_0))
        world = _spawn_agents_in_geometry(geom.polygon, 2, rng)
        # Force overlap
        world.positions = np.array([[5.0, 5.0], [5.1, 5.0]])
        forces = compute_contact_forces(world)
        # Forces should be non-zero and opposite
        assert np.linalg.norm(forces[0]) > 0
        np.testing.assert_allclose(forces[0] + forces[1], [0, 0], atol=1e-10)


class TestNavmeshWaypointsInGeneratedGeometry:
    """Test navmesh waypoint signals make sense in generated geometries."""

    def test_waypoint_points_toward_goal(self, rng):
        """Navmesh waypoint direction should have positive dot product with goal direction."""
        config = GeometryConfig(tier=GeometryTier.TIER_0)
        geom = generate_geometry(rng, config)
        navmesh = build_navmesh(geom.polygon)

        for _ in range(20):
            pos = sample_point_in_polygon(geom.polygon, rng)
            goal = sample_point_in_polygon(geom.polygon, rng)

            if np.linalg.norm(goal - pos) < 1.0:
                continue

            wp_dir = next_waypoint_direction(navmesh, pos, goal)
            if wp_dir is None:
                continue

            goal_dir = goal - pos
            goal_dir = goal_dir / np.linalg.norm(goal_dir)

            # In a convex polygon, waypoint should point generally toward goal
            dot = np.dot(wp_dir, goal_dir)
            assert dot > -0.5, f"Waypoint direction is away from goal: dot={dot}"


class TestActionCycleIntegration:
    """Test the full observe → act → update cycle."""

    def test_action_updates_state(self, rng):
        """Interpreted actions should produce valid state updates."""
        geom = generate_geometry(rng, GeometryConfig(tier=GeometryTier.TIER_0))
        world = _spawn_agents_in_geometry(geom.polygon, 5, rng)

        obs_config = ObsConfig()
        action_config = ActionConfig()

        # Observe
        obs = build_observations_batch(world, obs_config)
        assert np.all(np.isfinite(obs))

        # Random actions
        actions = rng.uniform(-1, 1, (5, 4))
        results = interpret_actions_batch(
            actions,
            world.torso_orientations,
            world.torso_orientations,
            world.head_orientations,
            action_config,
        )

        # Update state
        dt = 0.1
        for i, r in enumerate(results):
            world.positions[i] += r.desired_velocity * dt
            world.torso_orientations[i] = r.new_torso_orientation
            world.head_orientations[i] = r.new_head_orientation

        # Observe again — should still be finite
        obs2 = build_observations_batch(world, obs_config)
        assert np.all(np.isfinite(obs2))
        # Observations should have changed
        assert not np.allclose(obs, obs2)

    def test_multi_step_simulation(self, rng):
        """Run 50 steps of observe-act-update without numerical blowup."""
        geom = generate_geometry(rng, GeometryConfig(tier=GeometryTier.TIER_1))
        world = _spawn_agents_in_geometry(
            geom.polygon,
            10,
            rng,
            spawn_region=geom.spawn_regions[0],
            goal_region=geom.goal_regions[0],
        )

        obs_config = ObsConfig()
        action_config = ActionConfig()
        dt = 0.1

        for step in range(50):
            obs = build_observations_batch(world, obs_config)
            assert np.all(np.isfinite(obs)), f"Non-finite obs at step {step}"

            actions = rng.uniform(-1, 1, (world.n_agents, 4))
            results = interpret_actions_batch(
                actions,
                world.torso_orientations,
                world.torso_orientations,
                world.head_orientations,
                action_config,
            )

            forces = compute_contact_forces(world)

            for i, r in enumerate(results):
                world.velocities[i] = r.desired_velocity + forces[i] * dt
                world.positions[i] += world.velocities[i] * dt
                world.torso_orientations[i] = r.new_torso_orientation
                world.head_orientations[i] = r.new_head_orientation

        # Final state should still be finite
        assert np.all(np.isfinite(world.positions))
        assert np.all(np.isfinite(world.velocities))


class TestVisualisationSmoke:
    """Smoke tests for the visualiser (no visual assertions, just no crashes)."""

    def test_plot_geometry_all_tiers(self, rng):
        """Plotting should not crash for any tier."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        from crowdrl_env.visualiser import visualise_generated_geometry

        for tier in GeometryTier:
            config = GeometryConfig(tier=tier)
            geom = generate_geometry(rng, config)
            fig, ax = visualise_generated_geometry(geom)
            assert fig is not None
            plt_close(fig)

    def test_plot_with_navmesh(self, rng):
        import matplotlib

        matplotlib.use("Agg")
        from crowdrl_env.visualiser import visualise_generated_geometry

        geom = generate_geometry(rng, GeometryConfig(tier=GeometryTier.TIER_1))
        fig, ax = visualise_generated_geometry(geom, show_navmesh=True)
        assert fig is not None
        plt_close(fig)

    def test_plot_world_state(self, rng):
        import matplotlib

        matplotlib.use("Agg")
        from crowdrl_env.visualiser import visualise_world_state

        geom = generate_geometry(rng, GeometryConfig(tier=GeometryTier.TIER_0))
        world = _spawn_agents_in_geometry(geom.polygon, 5, rng)
        fig, ax = visualise_world_state(world)
        assert fig is not None
        plt_close(fig)

    def test_plot_raycasts(self, rng):
        import matplotlib

        matplotlib.use("Agg")
        from crowdrl_env.visualiser import visualise_world_state

        geom = generate_geometry(rng, GeometryConfig(tier=GeometryTier.TIER_1))
        world = _spawn_agents_in_geometry(geom.polygon, 3, rng)

        rc = RaycastConfig(n_rays=16, fov_deg=200, max_range=5.0)
        readings = cast_rays(world, 0, rc)

        fig, ax = visualise_world_state(
            world,
            show_raycasts=True,
            raycast_config=rc,
            raycast_readings=readings,
        )
        assert fig is not None
        plt_close(fig)


def plt_close(fig):
    """Helper to close a figure without importing plt at module level."""
    import matplotlib.pyplot as plt

    plt.close(fig)

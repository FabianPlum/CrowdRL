"""Emulator fidelity: ``router_next_waypoint`` vs JuPedSim's real RoutingEngine.

``use_jupedsim_style_routing`` trains policies on OUR reproduction of the
router's waypoint signal (funnel over a pure CDT with the fixed 0.2 m portal
inset, element [1] verbatim). This test measures that reproduction against the
real thing: ``jupedsim.RoutingEngine(polygon).compute_waypoints(pos, goal)[1]``
is bit-exact the ``next_target`` the tactical layer serves in-simulation (same
C++ path, same geometry canonicalization).

The assertions are DISTRIBUTIONAL, not exact. Accepted divergence sources,
verified against the jupedsim source (rev 49e3ddebd):

- CDT diagonal ties: jupedsim triangulates with CGAL ``Simple_cartesian<double>``
  (unfiltered predicates), we use the ``triangle`` library -- co-circular vertex
  quadruples can legitimately triangulate differently, changing portals.
- Channel tie-breaking: equal-length channels resolve first-found-wins in
  jupedsim's TA* (strict ``<``); our A* breaks ties by heap order.
- Funnel wedge tests: jupedsim's ``triarea2d`` computes in float32; ours in
  float64 -- corner decisions near collinearity can flip.
- Corner-pop boundaries: element [1] is discontinuous where a funnel corner
  pops into/out of line of sight; near those boundaries the two
  implementations may disagree by the full corner angle.
- Narrow portals (< 0.4 m): jupedsim's unclamped inset crosses candidates,
  ours collapses to the portal midpoint. Both geometries here keep apertures
  >= 1.4 m, outside that regime.
"""

import numpy as np
import pytest

pytest.importorskip("jupedsim", reason="requires a JuPedSim 2.0 source build")

import shapely  # noqa: E402

import jupedsim as jps  # noqa: E402
from crowdrl_core.geometry import build_navmesh  # noqa: E402
from crowdrl_core.navmesh import router_next_waypoint  # noqa: E402

# The jupedsim#1625 corner geometry (mirrors tests/test_e2e_jupedsim_trained_policy.py).
CORNER_AREA = shapely.Polygon([(0, 0), (12, 0), (12, 12), (10, 12), (10, 2), (0, 2)])
CORNER_GOAL = (11.0, 11.5)

# The e2e hourglass bottleneck, 1.4 m aperture.
BOTTLENECK_AREA = shapely.Polygon(
    [
        (0, 0),
        (6.8, 0),
        (6.8, 4.3),
        (7.2, 4.3),
        (7.2, 0),
        (14, 0),
        (14, 10),
        (7.2, 10),
        (7.2, 5.7),
        (6.8, 5.7),
        (6.8, 10),
        (0, 10),
    ]
)
BOTTLENECK_GOAL = (13.5, 5.0)

GRID_SPACING = 0.25
WALL_CLEARANCE = 0.3
POSITION_TOL = 0.05  # m
ANGLE_TOL = 3.0  # deg


def _interior_grid(polygon: shapely.Polygon) -> list[tuple[float, float]]:
    """Deterministic sample grid: interior points >= WALL_CLEARANCE off walls."""
    minx, miny, maxx, maxy = polygon.bounds
    boundary = polygon.boundary
    points = []
    for x in np.arange(minx + GRID_SPACING, maxx, GRID_SPACING):
        for y in np.arange(miny + GRID_SPACING, maxy, GRID_SPACING):
            p = shapely.Point(x, y)
            if polygon.contains(p) and boundary.distance(p) >= WALL_CLEARANCE:
                points.append((float(x), float(y)))
    return points


def _measure(polygon: shapely.Polygon, goal: tuple[float, float]):
    """Compare our emulated waypoint against the real router on the grid.

    A sample "agrees" when the waypoints are within POSITION_TOL of each other
    OR the directions from the sample position agree within ANGLE_TOL --
    position agreement covers standing near the waypoint (angles degenerate),
    angle agreement covers far waypoints (small positional offsets are
    irrelevant to the policy, which only consumes the direction).
    """
    navmesh = build_navmesh(polygon)
    engine = jps.RoutingEngine(polygon)
    goal_arr = np.asarray(goal, dtype=np.float64)

    angular_errors = []
    agreements = 0
    samples = _interior_grid(polygon)
    for pos in samples:
        pos_arr = np.asarray(pos, dtype=np.float64)
        real_path = engine.compute_waypoints(pos, goal)
        assert len(real_path) >= 2, f"router returned a degenerate path at {pos}"
        real = np.asarray(real_path[1], dtype=np.float64)
        ours = router_next_waypoint(navmesh, pos_arr, goal_arr)
        assert ours is not None, f"emulator found no route at {pos}"

        position_ok = float(np.linalg.norm(real - ours)) <= POSITION_TOL
        v_real, v_ours = real - pos_arr, ours - pos_arr
        n_real, n_ours = np.linalg.norm(v_real), np.linalg.norm(v_ours)
        if n_real < 1e-9 or n_ours < 1e-9:
            angle = 0.0 if position_ok else 180.0
        else:
            cos_angle = np.clip(np.dot(v_real, v_ours) / (n_real * n_ours), -1.0, 1.0)
            angle = float(np.degrees(np.arccos(cos_angle)))
        angular_errors.append(angle)
        if position_ok or angle <= ANGLE_TOL:
            agreements += 1

    return agreements / len(samples), float(np.median(angular_errors)), len(samples)


class TestRouterEmulationFidelity:
    def test_corner_geometry(self):
        agreement, median_deg, n = _measure(CORNER_AREA, CORNER_GOAL)
        assert n > 300, f"grid too sparse ({n} samples) to be meaningful"
        assert agreement >= 0.90, f"only {agreement:.1%} of {n} samples agree"
        assert median_deg < 1.0, f"median angular error {median_deg:.2f} deg"

    def test_bottleneck_geometry(self):
        agreement, median_deg, n = _measure(BOTTLENECK_AREA, BOTTLENECK_GOAL)
        assert n > 300, f"grid too sparse ({n} samples) to be meaningful"
        assert agreement >= 0.90, f"only {agreement:.1%} of {n} samples agree"
        assert median_deg < 1.0, f"median angular error {median_deg:.2f} deg"

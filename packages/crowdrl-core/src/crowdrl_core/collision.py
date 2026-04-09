"""Elliptical agent collision detection and contact forces.

Agents are modelled as axis-aligned ellipses rotated by their torso orientation.
- Semi-axis along torso forward = chest_depth
- Semi-axis perpendicular to torso = shoulder_width

Used by:
- The training environment (contact force computation)
- The raycast engine (ray-ellipse intersection for agent boundaries)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Point
from shapely.ops import nearest_points

from crowdrl_core.world_state import WorldState


def _rotation_matrix(angle: float) -> NDArray[np.float64]:
    """2x2 rotation matrix for a given angle (radians, CCW positive)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _ellipse_boundary_point(
    direction: NDArray[np.float64],
    semi_depth: float,
    semi_width: float,
    angle: float,
) -> NDArray[np.float64]:
    """Return the point on an ellipse boundary closest to *direction*.

    The ellipse is centred at the origin with semi-axes ``semi_depth``
    (along the local x-axis) and ``semi_width`` (along the local y-axis),
    rotated by ``angle``.

    *direction* is a 2-D vector in global frame indicating the direction
    from the ellipse centre toward the query point.
    """
    rot = _rotation_matrix(-angle)
    d_local = rot @ direction
    d_len = np.sqrt(d_local[0] ** 2 + d_local[1] ** 2)
    if d_len < 1e-12:
        # Degenerate: return point along local x-axis
        rot_back = _rotation_matrix(angle)
        return rot_back @ np.array([semi_depth, 0.0])
    # Normalised direction in local frame
    nx, ny = d_local[0] / d_len, d_local[1] / d_len
    # Parametric point on ellipse boundary in the given direction:
    # p = (a*cos(t), b*sin(t)) where tan(t) = (a*ny)/(b*nx)
    px = semi_depth * nx
    py = semi_width * ny
    scale = 1.0 / np.sqrt((px / semi_depth) ** 2 + (py / semi_width) ** 2)
    local_pt = np.array([px * scale, py * scale])
    rot_back = _rotation_matrix(angle)
    return rot_back @ local_pt


def ellipse_overlap(
    pos_a: NDArray,
    a_width: float,
    a_depth: float,
    a_angle: float,
    pos_b: NDArray,
    b_width: float,
    b_depth: float,
    b_angle: float,
) -> float:
    """Approximate overlap between two oriented ellipses.

    Computes the distance between the closest boundary points of each
    ellipse along the line connecting their centres.  When the sum of
    boundary radii exceeds the centre-to-centre distance the ellipses
    overlap and the returned value is the penetration depth normalised
    by the sum of boundary radii.

    Returns
    -------
    overlap : float
        > 0 if ellipses overlap (penetration depth / sum of radii), 0.0 otherwise.
    """
    d = pos_b - pos_a
    centre_dist = np.sqrt(d[0] ** 2 + d[1] ** 2)

    if centre_dist < 1e-12:
        # Coincident centres -- maximum overlap
        return 1.0

    # Boundary point of A toward B (global offset from A's centre)
    bp_a = _ellipse_boundary_point(d, a_depth, a_width, a_angle)
    # Boundary point of B toward A
    bp_b = _ellipse_boundary_point(-d, b_depth, b_width, b_angle)

    reach_a = np.sqrt(bp_a[0] ** 2 + bp_a[1] ** 2)
    reach_b = np.sqrt(bp_b[0] ** 2 + bp_b[1] ** 2)
    sum_reach = reach_a + reach_b

    if centre_dist >= sum_reach:
        return 0.0

    penetration = sum_reach - centre_dist
    return float(penetration / sum_reach)


def detect_collisions(world: WorldState) -> list[tuple[int, int, float]]:
    """Detect all pairwise agent-agent collisions (vectorized).

    Returns
    -------
    collisions : list of (i, j, overlap)
        Pairs of colliding agent indices and their approximate overlap.
    """
    n = world.n_agents
    if n < 2:
        return []

    # Determine active agents
    if world.active_mask is not None:
        active = world.active_mask
    else:
        active = np.ones(n, dtype=np.bool_)

    active_idx = np.where(active)[0]
    n_active = len(active_idx)
    if n_active < 2:
        return []

    pos = world.positions[active_idx]  # (M, 2)
    angles = world.torso_orientations[active_idx]  # (M,)
    widths = world.shoulder_widths[active_idx]  # (M,)
    depths = world.chest_depths[active_idx]  # (M,)

    # Pairwise displacement vectors: diff[i,j] = pos[j] - pos[i]
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]  # (M, M, 2)

    # Quick distance pre-filter: max possible contact range
    max_radii = np.maximum(widths, depths)
    pair_max_range = max_radii[:, np.newaxis] + max_radii[np.newaxis, :]  # (M, M)
    pair_dist_sq = np.sum(diff**2, axis=-1)  # (M, M)

    # Upper triangle mask + distance filter
    ii, jj = np.triu_indices(n_active, k=1)
    close_mask = pair_dist_sq[ii, jj] < (pair_max_range[ii, jj] ** 2) * 4.0
    ii = ii[close_mask]
    jj = jj[close_mask]

    if len(ii) == 0:
        return []

    # Vectorized ellipse boundary-distance overlap for candidate pairs
    d = diff[ii, jj]  # (P, 2)
    centre_dist = np.sqrt(np.sum(d**2, axis=-1))  # (P,)

    # Boundary reach of ellipse A toward B (in A's local frame)
    cos_a = np.cos(-angles[ii])
    sin_a = np.sin(-angles[ii])
    dx_a = cos_a * d[:, 0] - sin_a * d[:, 1]
    dy_a = sin_a * d[:, 0] + cos_a * d[:, 1]
    d_len_a = np.sqrt(dx_a**2 + dy_a**2)
    safe_len_a = np.where(d_len_a < 1e-12, 1.0, d_len_a)
    nx_a = dx_a / safe_len_a
    ny_a = dy_a / safe_len_a
    px_a = depths[ii] * nx_a
    py_a = widths[ii] * ny_a
    scale_a = 1.0 / np.sqrt((px_a / depths[ii]) ** 2 + (py_a / widths[ii]) ** 2)
    reach_a = np.sqrt((px_a * scale_a) ** 2 + (py_a * scale_a) ** 2)

    # Boundary reach of ellipse B toward A (in B's local frame)
    neg_d = -d
    cos_b = np.cos(-angles[jj])
    sin_b = np.sin(-angles[jj])
    dx_b = cos_b * neg_d[:, 0] - sin_b * neg_d[:, 1]
    dy_b = sin_b * neg_d[:, 0] + cos_b * neg_d[:, 1]
    d_len_b = np.sqrt(dx_b**2 + dy_b**2)
    safe_len_b = np.where(d_len_b < 1e-12, 1.0, d_len_b)
    nx_b = dx_b / safe_len_b
    ny_b = dy_b / safe_len_b
    px_b = depths[jj] * nx_b
    py_b = widths[jj] * ny_b
    scale_b = 1.0 / np.sqrt((px_b / depths[jj]) ** 2 + (py_b / widths[jj]) ** 2)
    reach_b = np.sqrt((px_b * scale_b) ** 2 + (py_b * scale_b) ** 2)

    sum_reach = reach_a + reach_b
    penetration = sum_reach - centre_dist
    # Handle coincident centres
    coincident = centre_dist < 1e-12
    overlaps = np.where(coincident, 1.0, np.where(penetration > 0, penetration / sum_reach, 0.0))
    overlap_mask = overlaps > 0

    # Collect results
    col_ii = ii[overlap_mask]
    col_jj = jj[overlap_mask]
    col_overlaps = overlaps[overlap_mask]

    # Map back to original indices
    return [
        (int(active_idx[i]), int(active_idx[j]), float(o))
        for i, j, o in zip(col_ii, col_jj, col_overlaps)
    ]


def _point_to_segment_nearest(
    point: NDArray[np.float64],
    seg_start: NDArray[np.float64],
    seg_end: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Return the nearest point on a line segment and the distance to it."""
    edge = seg_end - seg_start
    edge_len_sq = np.dot(edge, edge)
    if edge_len_sq < 1e-12:
        return seg_start.copy(), float(np.linalg.norm(point - seg_start))
    t = np.clip(np.dot(point - seg_start, edge) / edge_len_sq, 0.0, 1.0)
    nearest = seg_start + t * edge
    return nearest, float(np.linalg.norm(point - nearest))


def _points_to_segments_nearest_batch(
    points: NDArray[np.float64],
    seg_starts: NDArray[np.float64],
    seg_ends: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Batch nearest-point-on-segment for all points x all segments.

    Parameters
    ----------
    points : (N, 2)
    seg_starts : (W, 2)
    seg_ends : (W, 2)

    Returns
    -------
    nearest_points : (N, W, 2) — nearest point on each segment for each agent
    distances : (N, W) — distances
    normals : (N, W, 2) — unit vectors from nearest point toward agent (or edge normal)
    """
    # edges: (W, 2)
    edges = seg_ends - seg_starts
    edge_len_sq = np.sum(edges**2, axis=-1)  # (W,)

    # diff from each point to each segment start: (N, W, 2)
    diff = points[:, np.newaxis, :] - seg_starts[np.newaxis, :, :]

    # Project: t = dot(diff, edge) / edge_len_sq, clamped to [0, 1]
    dot_prod = np.sum(diff * edges[np.newaxis, :, :], axis=-1)  # (N, W)
    safe_len_sq = np.where(edge_len_sq < 1e-12, 1.0, edge_len_sq)
    t = np.clip(dot_prod / safe_len_sq, 0.0, 1.0)

    # Handle degenerate segments
    t = np.where(edge_len_sq[np.newaxis, :] < 1e-12, 0.0, t)

    # Nearest points on segments: (N, W, 2)
    nearest = seg_starts[np.newaxis, :, :] + t[:, :, np.newaxis] * edges[np.newaxis, :, :]

    # Vectors from nearest to point: (N, W, 2)
    to_point = points[:, np.newaxis, :] - nearest
    distances = np.sqrt(np.sum(to_point**2, axis=-1))  # (N, W)

    # Normals: from nearest toward point (or edge normal if too close)
    safe_dist = np.where(distances < 1e-10, 1.0, distances)
    normals = to_point / safe_dist[:, :, np.newaxis]

    # For near-zero distances, use edge perpendicular
    edge_normals = np.stack([-edges[:, 1], edges[:, 0]], axis=-1)  # (W, 2)
    edge_norm_len = np.sqrt(np.sum(edge_normals**2, axis=-1, keepdims=True))
    safe_edge_norm = np.where(edge_norm_len < 1e-12, 1.0, edge_norm_len)
    edge_normals = edge_normals / safe_edge_norm

    close_mask = distances < 1e-10  # (N, W)
    normals = np.where(
        close_mask[:, :, np.newaxis],
        edge_normals[np.newaxis, :, :],
        normals,
    )

    return nearest, distances, normals


def compute_min_wall_distances(world: WorldState) -> NDArray[np.float64]:
    """Return the minimum distance from each agent to the nearest wall segment.

    Parameters
    ----------
    world : WorldState

    Returns
    -------
    min_distances : (n_agents,) — distance to nearest wall per agent.
        Returns ``inf`` for agents with no wall segments available.
    """
    n = world.n_agents
    if world.wall_segments is None or len(world.wall_segments) == 0:
        return np.full(n, np.inf, dtype=np.float64)

    seg_starts = np.array([s[0] for s in world.wall_segments])  # (W, 2)
    seg_ends = np.array([s[1] for s in world.wall_segments])  # (W, 2)

    _, distances, _ = _points_to_segments_nearest_batch(
        world.positions, seg_starts, seg_ends
    )  # distances: (N, W)

    return distances.min(axis=1)  # (N,)


def compute_min_agent_distances(world: WorldState) -> NDArray[np.float64]:
    """Return the minimum centre-to-centre distance to the nearest other agent.

    Parameters
    ----------
    world : WorldState

    Returns
    -------
    min_distances : (n_agents,) -- distance to nearest active neighbour.
        Returns ``inf`` for agents with no active neighbours.
    """
    n = world.n_agents
    if n < 2:
        return np.full(n, np.inf, dtype=np.float64)

    if world.active_mask is not None:
        active = world.active_mask
    else:
        active = np.ones(n, dtype=np.bool_)

    # Pairwise distances
    diff = world.positions[np.newaxis, :, :] - world.positions[:, np.newaxis, :]  # (N, N, 2)
    dists = np.sqrt(np.sum(diff**2, axis=-1))  # (N, N)

    # Mask self and inactive agents
    np.fill_diagonal(dists, np.inf)
    inactive_mask = ~active
    dists[:, inactive_mask] = np.inf

    return dists.min(axis=1)  # (N,)


# ---------------------------------------------------------------------------
# Agent-agent + wall forces
# ---------------------------------------------------------------------------


def compute_contact_forces(
    world: WorldState,
    stiffness: float = 30000.0,
    damping: float = 500.0,
    wall_strength: float = 400.0,
    wall_range: float = 0.3,
    collisions: list[tuple[int, int, float]] | None = None,
) -> NDArray[np.float64]:
    """Compute repulsive accelerations for all agents (agent-agent + wall).

    Agent-agent: spring-damper on overlap, divided by agent mass (F=ma).
    Wall: smooth exponential repulsion (following JuPedSim's BoundaryRepulsion),
    divided by agent mass.

    All outputs are accelerations (m/s^2), applied as ``v += accel * dt``.
    Stiffness and wall_strength have units of N (force), and the per-agent
    mass converts them to accelerations.

    Parameters
    ----------
    stiffness : float
        Agent-agent spring force (N per unit overlap).
    damping : float
        Agent-agent velocity-dependent damping (N*s/m).
    wall_strength : float
        Wall repulsion force amplitude (N).
    wall_range : float
        Wall repulsion length scale (m). Force = strength * exp((radius - dist) / range).
    collisions : list of (i, j, overlap), optional
        Pre-computed collision list. If None, detect_collisions() is called.

    Returns
    -------
    accelerations : (n_agents, 2) array
        Net acceleration on each agent (m/s^2).
    """
    n = world.n_agents
    forces = np.zeros((n, 2), dtype=np.float64)

    # --- Agent-agent contact forces ---
    if collisions is None:
        collisions = detect_collisions(world)

    if collisions:
        col_arr = np.asarray(collisions)  # (C, 3)
        ci = col_arr[:, 0].astype(np.intp)
        cj = col_arr[:, 1].astype(np.intp)
        overlaps = col_arr[:, 2]

        diff = world.positions[cj] - world.positions[ci]  # (C, 2)
        dist = np.sqrt(np.sum(diff**2, axis=-1))  # (C,)

        # Normal vectors (fallback to [1, 0] for coincident agents)
        safe_dist = np.where(dist < 1e-10, 1.0, dist)
        normals = diff / safe_dist[:, np.newaxis]
        normals = np.where(
            (dist < 1e-10)[:, np.newaxis],
            np.array([[1.0, 0.0]]),
            normals,
        )

        rel_vel = world.velocities[ci] - world.velocities[cj]  # (C, 2)
        rel_vel_normal = np.sum(rel_vel * normals, axis=-1)  # (C,)

        force_mag = stiffness * overlaps + damping * np.maximum(0.0, rel_vel_normal)
        force_vectors = force_mag[:, np.newaxis] * normals  # (C, 2)

        # Accumulate forces using np.add.at for correct handling of duplicates
        np.add.at(forces, ci, -force_vectors)
        np.add.at(forces, cj, force_vectors)

    # --- Smooth exponential wall repulsion (vectorized) ---
    if world.wall_segments is not None and len(world.wall_segments) > 0:
        if world.active_mask is not None:
            active_idx = np.where(world.active_mask)[0]
        else:
            active_idx = np.arange(n)

        if len(active_idx) > 0:
            seg_starts = np.array([s[0] for s in world.wall_segments])  # (W, 2)
            seg_ends = np.array([s[1] for s in world.wall_segments])  # (W, 2)

            active_pos = world.positions[active_idx]  # (M, 2)
            _, distances, normals = _points_to_segments_nearest_batch(
                active_pos,
                seg_starts,
                seg_ends,
            )

            # Radii per active agent: (M,)
            radii = np.maximum(
                world.shoulder_widths[active_idx],
                world.chest_depths[active_idx],
            )

            # Exponential repulsion: (M, W)
            f_mag = wall_strength * np.exp((radii[:, np.newaxis] - distances) / wall_range)

            # Sum over all wall segments: (M, 2)
            wall_forces = np.sum(f_mag[:, :, np.newaxis] * normals, axis=1)

            forces[active_idx] += wall_forces

    # Convert forces (N) to accelerations (m/s^2) via F = m*a
    masses = world.masses
    forces /= masses[:, np.newaxis]

    return forces


# ---------------------------------------------------------------------------
# Hard wall constraint
# ---------------------------------------------------------------------------


def enforce_wall_boundaries(world: WorldState) -> None:
    """Project agents to stay inside the walkable polygon with body clearance.

    Uses vectorized numpy geometry for the fast path (distance to nearest
    wall segment). Falls back to Shapely for agents that need correction.

    Call **after** physics integration every step.
    Modifies ``world.positions`` and ``world.velocities`` in place.
    """
    polygon = world.walkable_polygon
    if polygon is None:
        return

    n = world.n_agents
    if world.active_mask is not None:
        active_idx = np.where(world.active_mask)[0]
    else:
        active_idx = np.arange(n)

    if len(active_idx) == 0:
        return

    # Quick vectorized pre-filter: agents far from all wall segments in a
    # simple (convex, no-holes) polygon are certainly safe.  When the polygon
    # has holes the distance-to-nearest-segment test is not sufficient because
    # an agent inside a hole can be far from exterior walls yet outside the
    # walkable area.  Detect holes cheaply and skip the pre-filter in that case.
    has_holes = hasattr(polygon, "interiors") and len(list(polygon.interiors)) > 0

    if not has_holes and world.wall_segments is not None and len(world.wall_segments) > 0:
        seg_starts = np.array([s[0] for s in world.wall_segments])
        seg_ends = np.array([s[1] for s in world.wall_segments])

        active_pos = world.positions[active_idx]
        _, distances, _ = _points_to_segments_nearest_batch(
            active_pos,
            seg_starts,
            seg_ends,
        )
        min_wall_dist = distances.min(axis=1)  # (M,)

        radii = np.maximum(
            world.shoulder_widths[active_idx],
            world.chest_depths[active_idx],
        )

        # Generous threshold so agents near corners are not missed.
        # Also always check agents moving fast enough that they could
        # have crossed a boundary in one step (contact forces can
        # produce large velocity spikes).
        speeds = np.linalg.norm(world.velocities[active_idx], axis=1)
        needs_check = (min_wall_dist < radii * 3.0) | (speeds > radii * 10.0)
        check_idx = active_idx[needs_check]
    else:
        check_idx = active_idx

    # Detailed Shapely check only for agents near walls
    boundary = polygon.boundary

    for i in check_idx:
        radius = float(max(world.shoulder_widths[i], world.chest_depths[i]))
        pos = world.positions[i]
        p = Point(pos[0], pos[1])

        boundary_dist = boundary.distance(p)
        inside = polygon.contains(p)

        if inside and boundary_dist >= radius:
            continue

        # Find nearest point on boundary
        nearest_pt = nearest_points(boundary, p)[0]
        nearest_pos = np.array([nearest_pt.x, nearest_pt.y], dtype=np.float64)

        # Compute inward normal
        diff = pos - nearest_pos
        diff_len = np.linalg.norm(diff)

        if diff_len > 1e-10:
            if inside:
                inward = diff / diff_len
            else:
                inward = -diff / diff_len
        else:
            ref = polygon.representative_point()
            inward = np.array([ref.x, ref.y]) - nearest_pos
            inward_len = np.linalg.norm(inward)
            if inward_len < 1e-10:
                continue
            inward /= inward_len

        world.positions[i] = nearest_pos + inward * radius

        vel_into_wall = np.dot(world.velocities[i], -inward)
        if vel_into_wall > 0:
            world.velocities[i] += vel_into_wall * inward


# ---------------------------------------------------------------------------
# Ray-ellipse intersection (used by raycast engine)
# ---------------------------------------------------------------------------


def ray_ellipse_intersection(
    ray_origin: NDArray[np.float64],
    ray_dir: NDArray[np.float64],
    ellipse_centre: NDArray[np.float64],
    semi_width: float,
    semi_depth: float,
    ellipse_angle: float,
) -> float | None:
    """Find the distance along a ray to its intersection with an oriented ellipse boundary.

    The ellipse is centred at `ellipse_centre`, with semi-axes `semi_depth` (along
    forward/torso direction) and `semi_width` (perpendicular), rotated by `ellipse_angle`.

    Parameters
    ----------
    ray_origin : (2,) array
    ray_dir : (2,) unit direction vector
    ellipse_centre : (2,) array
    semi_width : float — perpendicular to torso
    semi_depth : float — along torso forward
    ellipse_angle : float — rotation of the ellipse (radians)

    Returns
    -------
    t : float or None
        Distance along ray to nearest intersection point (t > 0), or None if no hit.
    """
    # Transform ray into ellipse-local coordinates (ellipse becomes unit circle)
    rot = _rotation_matrix(-ellipse_angle)
    origin_local = rot @ (ray_origin - ellipse_centre)
    dir_local = rot @ ray_dir

    # Scale to unit circle
    origin_scaled = np.array([origin_local[0] / semi_depth, origin_local[1] / semi_width])
    dir_scaled = np.array([dir_local[0] / semi_depth, dir_local[1] / semi_width])

    # Solve |origin_scaled + t * dir_scaled|^2 = 1
    a = np.dot(dir_scaled, dir_scaled)
    b = 2.0 * np.dot(origin_scaled, dir_scaled)
    c = np.dot(origin_scaled, origin_scaled) - 1.0

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0:
        return None

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    # Return the nearest positive intersection
    eps = 1e-8
    if t1 > eps:
        return float(t1)
    if t2 > eps:
        return float(t2)
    return None

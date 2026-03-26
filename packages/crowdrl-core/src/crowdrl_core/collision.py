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

from crowdrl_core.world_state import WorldState


def _rotation_matrix(angle: float) -> NDArray[np.float64]:
    """2x2 rotation matrix for a given angle (radians, CCW positive)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def ellipse_overlap(
    pos_a: NDArray, a_width: float, a_depth: float, a_angle: float,
    pos_b: NDArray, b_width: float, b_depth: float, b_angle: float,
) -> float:
    """Approximate overlap between two oriented ellipses.

    Uses the algebraic distance approach: transforms the problem so that
    ellipse A becomes a unit circle, then checks if ellipse B's centre
    is inside ellipse A (and vice versa), returning a proxy overlap measure.

    Returns
    -------
    overlap : float
        > 0 if ellipses overlap (approximate penetration depth), 0.0 otherwise.
    """
    # Vector from A to B
    d = pos_b - pos_a

    # Check if B's centre is inside A's ellipse
    rot_a = _rotation_matrix(-a_angle)
    d_in_a = rot_a @ d
    dist_a = (d_in_a[0] / a_depth) ** 2 + (d_in_a[1] / a_width) ** 2

    # Check if A's centre is inside B's ellipse
    rot_b = _rotation_matrix(-b_angle)
    d_in_b = rot_b @ (-d)
    dist_b = (d_in_b[0] / b_depth) ** 2 + (d_in_b[1] / b_width) ** 2

    # Use the minimum algebraic distance as overlap proxy
    min_dist = min(dist_a, dist_b)
    if min_dist < 1.0:
        return 1.0 - np.sqrt(min_dist)
    return 0.0


def detect_collisions(world: WorldState) -> list[tuple[int, int, float]]:
    """Detect all pairwise agent-agent collisions.

    Returns
    -------
    collisions : list of (i, j, overlap)
        Pairs of colliding agent indices and their approximate overlap.
    """
    n = world.n_agents
    collisions = []

    for i in range(n):
        if world.active_mask is not None and not world.active_mask[i]:
            continue
        for j in range(i + 1, n):
            if world.active_mask is not None and not world.active_mask[j]:
                continue

            overlap = ellipse_overlap(
                world.positions[i],
                world.shoulder_widths[i], world.chest_depths[i],
                world.torso_orientations[i],
                world.positions[j],
                world.shoulder_widths[j], world.chest_depths[j],
                world.torso_orientations[j],
            )
            if overlap > 0:
                collisions.append((i, j, overlap))

    return collisions


def compute_contact_forces(
    world: WorldState,
    stiffness: float = 2000.0,
    damping: float = 50.0,
) -> NDArray[np.float64]:
    """Compute repulsive contact forces for all agents.

    Uses a spring-damper model: F = stiffness * overlap * normal + damping * rel_vel_normal.

    Parameters
    ----------
    stiffness : float
        Spring constant (N/m). Higher = harder collisions.
    damping : float
        Damping coefficient (N·s/m). Prevents oscillation.

    Returns
    -------
    forces : (n_agents, 2) array
        Net contact force on each agent.
    """
    n = world.n_agents
    forces = np.zeros((n, 2), dtype=np.float64)

    collisions = detect_collisions(world)
    for i, j, overlap in collisions:
        # Normal direction: from i to j
        diff = world.positions[j] - world.positions[i]
        dist = np.linalg.norm(diff)
        if dist < 1e-10:
            # Agents exactly on top of each other — use random direction
            normal = np.array([1.0, 0.0])
        else:
            normal = diff / dist

        # Relative velocity along normal (positive = approaching)
        rel_vel = world.velocities[i] - world.velocities[j]
        rel_vel_normal = np.dot(rel_vel, normal)

        # Spring-damper force
        force_magnitude = stiffness * overlap + damping * max(0.0, rel_vel_normal)
        force = force_magnitude * normal

        forces[i] -= force  # Push i away from j
        forces[j] += force  # Push j away from i

    return forces


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
    origin_scaled = np.array(
        [origin_local[0] / semi_depth, origin_local[1] / semi_width]
    )
    dir_scaled = np.array(
        [dir_local[0] / semi_depth, dir_local[1] / semi_width]
    )

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

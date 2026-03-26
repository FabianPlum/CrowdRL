"""Visualiser for CrowdRL geometries, agent states, and observations.

Renders geometries, navmeshes, agent positions/orientations, raycasts,
and spawn/goal regions using matplotlib. Designed for debugging and
generating figures for papers/documentation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as mpatches
from matplotlib.collections import LineCollection
from numpy.typing import NDArray
from shapely.geometry import Polygon

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from crowdrl_core.sensing import RaycastConfig
    from crowdrl_core.world_state import NavMesh, WorldState

    from crowdrl_env.geometry_generator import GeneratedGeometry


def _polygon_to_patch(polygon: Polygon, **kwargs) -> mpatches.Polygon:
    """Convert a Shapely Polygon exterior to a matplotlib patch."""
    coords = np.array(polygon.exterior.coords)
    return mpatches.Polygon(coords, **kwargs)


def plot_geometry(
    polygon: Polygon,
    ax: Axes | None = None,
    show_holes: bool = True,
    walkable_color: str = "#e8e8e8",
    wall_color: str = "#333333",
    obstacle_color: str = "#999999",
    wall_linewidth: float = 2.0,
) -> tuple[Figure, Axes]:
    """Plot a walkable polygon with obstacles.

    Parameters
    ----------
    polygon : Polygon
        Shapely Polygon with holes.
    ax : Axes or None
        Matplotlib axes to plot on. If None, creates a new figure.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig = ax.figure

    # Draw walkable area
    walkable = _polygon_to_patch(
        polygon,
        facecolor=walkable_color,
        edgecolor=wall_color,
        linewidth=wall_linewidth,
        zorder=1,
    )
    ax.add_patch(walkable)

    # Draw obstacles (holes)
    if show_holes:
        for hole in polygon.interiors:
            coords = np.array(hole.coords)
            obs_patch = mpatches.Polygon(
                coords,
                facecolor=obstacle_color,
                edgecolor=wall_color,
                linewidth=wall_linewidth,
                zorder=2,
            )
            ax.add_patch(obs_patch)

    # Set axis limits with padding
    minx, miny, maxx, maxy = polygon.bounds
    pad = max(maxx - minx, maxy - miny) * 0.05
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_navmesh(
    navmesh: NavMesh,
    ax: Axes,
    tri_color: str = "#4a90d9",
    tri_alpha: float = 0.15,
    edge_color: str = "#4a90d9",
    edge_alpha: float = 0.4,
    show_centroids: bool = False,
    show_adjacency: bool = False,
) -> None:
    """Overlay the navigation mesh on an existing plot."""
    for tri in navmesh.triangles:
        triangle = mpatches.Polygon(
            tri,
            closed=True,
            facecolor=tri_color,
            alpha=tri_alpha,
            edgecolor=edge_color,
            linewidth=0.5,
            zorder=3,
        )
        ax.add_patch(triangle)

    if show_centroids:
        ax.scatter(
            navmesh.centroids[:, 0],
            navmesh.centroids[:, 1],
            c=tri_color,
            s=8,
            zorder=5,
            alpha=0.7,
        )

    if show_adjacency:
        lines = []
        for i, neighbours in enumerate(navmesh.adjacency):
            for j in neighbours:
                if j > i:
                    lines.append([navmesh.centroids[i], navmesh.centroids[j]])
        if lines:
            lc = LineCollection(lines, colors=edge_color, alpha=edge_alpha, linewidths=0.8)
            ax.add_collection(lc)


def plot_agents(
    world: WorldState,
    ax: Axes,
    show_orientations: bool = True,
    show_goals: bool = True,
    show_ids: bool = True,
    agent_color: str = "#e74c3c",
    goal_color: str = "#2ecc71",
    inactive_color: str = "#cccccc",
    orientation_length: float = 0.5,
) -> None:
    """Plot agent positions, orientations, and goals."""
    for i in range(world.n_agents):
        is_active = world.active_mask is None or world.active_mask[i]
        color = agent_color if is_active else inactive_color

        pos = world.positions[i]

        # Draw elliptical body
        ellipse = mpatches.Ellipse(
            pos,
            width=2 * world.chest_depths[i],
            height=2 * world.shoulder_widths[i],
            angle=np.degrees(world.torso_orientations[i]),
            facecolor=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.7,
            zorder=10,
        )
        ax.add_patch(ellipse)

        if show_orientations and is_active:
            # Torso forward direction
            torso_dir = np.array(
                [
                    np.cos(world.torso_orientations[i]),
                    np.sin(world.torso_orientations[i]),
                ]
            )
            ax.arrow(
                pos[0],
                pos[1],
                torso_dir[0] * orientation_length,
                torso_dir[1] * orientation_length,
                head_width=0.08,
                head_length=0.05,
                fc=color,
                ec="black",
                linewidth=0.5,
                zorder=11,
            )

            # Head direction (dashed, thinner)
            head_dir = np.array(
                [
                    np.cos(world.head_orientations[i]),
                    np.sin(world.head_orientations[i]),
                ]
            )
            ax.plot(
                [pos[0], pos[0] + head_dir[0] * orientation_length * 0.7],
                [pos[1], pos[1] + head_dir[1] * orientation_length * 0.7],
                color="#3498db",
                linewidth=1.5,
                linestyle="--",
                zorder=11,
            )

        if show_goals and is_active:
            goal = world.goal_positions[i]
            ax.plot(
                goal[0],
                goal[1],
                marker="*",
                markersize=10,
                color=goal_color,
                markeredgecolor="black",
                markeredgewidth=0.5,
                zorder=9,
            )
            ax.plot(
                [pos[0], goal[0]],
                [pos[1], goal[1]],
                color=goal_color,
                linewidth=0.5,
                linestyle=":",
                alpha=0.4,
                zorder=8,
            )

        if show_ids:
            ax.text(
                pos[0],
                pos[1] + world.shoulder_widths[i] + 0.15,
                str(i),
                fontsize=7,
                ha="center",
                va="bottom",
                zorder=12,
            )


def plot_raycasts(
    world: WorldState,
    agent_idx: int,
    config: RaycastConfig,
    readings: NDArray[np.float64],
    ax: Axes,
    ray_color: str = "#f39c12",
    hit_color: str = "#e74c3c",
) -> None:
    """Plot raycast beams for a single agent."""
    origin = world.positions[agent_idx]
    head_angle = world.head_orientations[agent_idx]

    fov_rad = np.radians(config.fov_deg)
    start_angle = head_angle - fov_rad / 2.0

    if config.n_rays == 1:
        ray_angles = np.array([head_angle])
    else:
        ray_angles = np.linspace(start_angle, start_angle + fov_rad, config.n_rays)

    for r, angle in enumerate(ray_angles):
        direction = np.array([np.cos(angle), np.sin(angle)])

        if config.two_channel:
            dist = readings[r, 0] * config.max_range
        else:
            dist = readings[r] * config.max_range

        endpoint = origin + direction * dist
        hit = dist < config.max_range - 0.01

        ax.plot(
            [origin[0], endpoint[0]],
            [origin[1], endpoint[1]],
            color=hit_color if hit else ray_color,
            linewidth=0.8 if hit else 0.4,
            alpha=0.8 if hit else 0.3,
            zorder=7,
        )
        if hit:
            ax.plot(endpoint[0], endpoint[1], "o", color=hit_color, markersize=3, zorder=8)


def plot_spawn_goal_regions(
    geom: GeneratedGeometry,
    ax: Axes,
    spawn_color: str = "#3498db",
    goal_color: str = "#2ecc71",
    alpha: float = 0.2,
) -> None:
    """Overlay spawn and goal regions on the plot."""
    for region in geom.spawn_regions:
        if isinstance(region, Polygon) and not region.is_empty:
            patch = _polygon_to_patch(
                region,
                facecolor=spawn_color,
                edgecolor=spawn_color,
                alpha=alpha,
                linewidth=1.5,
                linestyle="--",
                zorder=4,
            )
            ax.add_patch(patch)

    for region in geom.goal_regions:
        if isinstance(region, Polygon) and not region.is_empty:
            patch = _polygon_to_patch(
                region,
                facecolor=goal_color,
                edgecolor=goal_color,
                alpha=alpha,
                linewidth=1.5,
                linestyle="--",
                zorder=4,
            )
            ax.add_patch(patch)


def visualise_generated_geometry(
    geom: GeneratedGeometry,
    title: str | None = None,
    show_navmesh: bool = False,
    navmesh: NavMesh | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[Figure, Axes]:
    """Complete visualisation of a generated geometry with regions.

    Parameters
    ----------
    geom : GeneratedGeometry
    title : str or None
    show_navmesh : bool
        If True, overlay the navmesh triangulation.
    navmesh : NavMesh or None
        Pre-built navmesh. If None and show_navmesh is True, builds one.
    ax : Axes or None
        If provided, draw into this axes. Otherwise create a new figure.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()
    plot_geometry(geom.polygon, ax=ax)
    plot_spawn_goal_regions(geom, ax=ax)

    if show_navmesh:
        if navmesh is None:
            from crowdrl_core.geometry import build_navmesh

            navmesh = build_navmesh(geom.polygon)
        plot_navmesh(navmesh, ax, show_centroids=True, show_adjacency=True)

    if title is None:
        title = f"Tier {geom.tier.value}: {geom.metadata.get('shape', 'unknown')}"
    ax.set_title(title)

    return fig, ax


def visualise_world_state(
    world: WorldState,
    title: str = "World State",
    show_navmesh: bool = False,
    show_raycasts: bool = False,
    raycast_agent: int | None = None,
    raycast_config: RaycastConfig | None = None,
    raycast_readings: NDArray[np.float64] | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[Figure, Axes]:
    """Complete visualisation of a world state.

    Parameters
    ----------
    world : WorldState
    title : str
    show_navmesh : bool
    show_raycasts : bool
        If True, show raycasts for one agent.
    raycast_agent : int
        Which agent's raycasts to show.
    raycast_config : RaycastConfig
    raycast_readings : NDArray
        Pre-computed readings (avoids recomputing).
    ax : Axes or None
        If provided, draw into this axes. Otherwise create a new figure.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    if world.walkable_polygon is not None:
        plot_geometry(world.walkable_polygon, ax=ax)

    if show_navmesh and world.navmesh is not None:
        plot_navmesh(world.navmesh, ax, show_centroids=True, show_adjacency=True)

    plot_agents(world, ax)

    if show_raycasts and raycast_config is not None and raycast_readings is not None:
        agent_idx = raycast_agent if raycast_agent is not None else 0
        plot_raycasts(world, agent_idx, raycast_config, raycast_readings, ax)

    ax.set_title(title)
    return fig, ax

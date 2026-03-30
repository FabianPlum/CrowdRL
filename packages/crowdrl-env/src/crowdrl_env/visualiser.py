"""Visualiser for CrowdRL geometries, agent states, and observations.

Renders geometries, navmeshes, agent positions/orientations, raycasts,
and spawn/goal regions using matplotlib. Designed for debugging and
generating figures for papers/documentation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
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

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Episode video rendering
# ---------------------------------------------------------------------------


@dataclass
class EpisodeFrames:
    """Per-frame snapshot data collected during an episode for video rendering.

    All arrays have shape ``(n_frames, n_agents, ...)`` unless noted.
    """

    positions: NDArray[np.float64]
    """(n_frames, n_agents, 2)"""

    torso_orientations: NDArray[np.float64]
    """(n_frames, n_agents)"""

    head_orientations: NDArray[np.float64]
    """(n_frames, n_agents)"""

    shoulder_widths: NDArray[np.float64]
    """(n_agents,) — constant across frames."""

    chest_depths: NDArray[np.float64]
    """(n_agents,) — constant across frames."""

    goal_positions: NDArray[np.float64]
    """(n_agents, 2) — constant across frames."""

    polygon: Polygon
    """Walkable polygon for this episode."""

    active_masks: NDArray[np.bool_]
    """(n_frames, n_agents)"""

    reached_goal: NDArray[np.bool_]
    """(n_agents,) — whether each agent reached its goal by episode end."""

    dt: float = 0.01
    """Simulation timestep in seconds (used for timestamp display)."""

    title: str = ""
    """Optional title shown above the plot."""

    @property
    def n_frames(self) -> int:
        return self.positions.shape[0]

    @property
    def n_agents(self) -> int:
        return self.positions.shape[1]


def collect_episode_frames(
    env,
    actor_critic,
    obs_normalizer=None,
    device=None,
    max_steps: int | None = None,
) -> EpisodeFrames:
    """Run one episode and collect per-frame data for video rendering.

    Parameters
    ----------
    env : CrowdEnv
        The environment to run.
    actor_critic
        Policy network (must have ``get_action_and_value``).
    obs_normalizer
        Optional observation normalizer with ``.normalize()`` method.
    device
        Torch device for policy inference.
    max_steps : int or None
        Maximum number of steps. If None, uses the environment's
        ``config.max_steps``.

    Returns
    -------
    EpisodeFrames
        Snapshot data suitable for ``render_episode_video``.
    """
    import torch

    if max_steps is None:
        max_steps = getattr(env.config, "max_steps", 2000)

    obs, info = env.reset()
    n_agents = info["n_agents"]
    world = env._world

    pos_list = [world.positions.copy()]
    torso_list = [world.torso_orientations.copy()]
    head_list = [world.head_orientations.copy()]
    active_list = [
        world.active_mask.copy()
        if world.active_mask is not None
        else np.ones(n_agents, dtype=bool)
    ]
    reached_goal = np.zeros(n_agents, dtype=bool)

    for _step in range(max_steps):
        obs_norm = obs_normalizer.normalize(obs) if obs_normalizer is not None else obs

        with torch.no_grad():
            obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=device)
            actions, _, _, _, _ = actor_critic.get_action_and_value(obs_t)

        obs, _rewards, terminated, _truncated, step_info = env.step(actions.cpu().numpy())

        pos_list.append(world.positions.copy())
        torso_list.append(world.torso_orientations.copy())
        head_list.append(world.head_orientations.copy())
        active_list.append(
            world.active_mask.copy()
            if world.active_mask is not None
            else np.ones(n_agents, dtype=bool)
        )
        reached_goal |= terminated

        if step_info.get("episode_over", False) or reached_goal.all():
            break

    return EpisodeFrames(
        positions=np.stack(pos_list),
        torso_orientations=np.stack(torso_list),
        head_orientations=np.stack(head_list),
        shoulder_widths=world.shoulder_widths.copy(),
        chest_depths=world.chest_depths.copy(),
        goal_positions=world.goal_positions.copy(),
        polygon=world.walkable_polygon,
        active_masks=np.stack(active_list),
        reached_goal=reached_goal,
        dt=env.config.dt if hasattr(env.config, "dt") else 0.01,
    )


def render_episode_video(
    frames: EpisodeFrames,
    output_path: str | Path,
    *,
    fps: int = 20,
    frame_skip: int = 5,
    trail_length: int = 20,
    figsize: tuple[float, float] = (10, 8),
    dpi: int = 100,
    agent_color: str = "#e74c3c",
    goal_color: str = "#2ecc71",
    inactive_color: str = "#cccccc",
    orientation_length: float = 0.4,
    show_trails: bool = True,
) -> Path:
    """Render an episode as an MP4 video.

    Parameters
    ----------
    frames : EpisodeFrames
        Collected frame data from ``collect_episode_frames``.
    output_path : str or Path
        Where to write the MP4 file. Parent directory is created if needed.
    fps : int
        Frames per second in the output video.
    frame_skip : int
        Only render every *frame_skip*-th simulation frame (default 5).
        A value of 1 renders every frame.
    trail_length : int
        Number of past *rendered* positions to show as a fading trail per agent.
    figsize, dpi : tuple, int
        Figure size and resolution.
    show_trails : bool
        Whether to draw trajectory trails behind agents.

    Returns
    -------
    Path
        The output file path.
    """
    from matplotlib.animation import FuncAnimation

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sub-sample frames according to frame_skip
    frame_indices = list(range(0, frames.n_frames, frame_skip))
    n_frames = len(frame_indices)
    n_agents = frames.n_agents
    cmap = plt.get_cmap("tab20", max(n_agents, 1))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_geometry(frames.polygon, ax=ax)

    # Plot goal markers (static)
    for i in range(n_agents):
        color = cmap(i % 20)
        ax.plot(
            frames.goal_positions[i, 0],
            frames.goal_positions[i, 1],
            "*",
            color=color,
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.3,
            zorder=5,
        )

    # Pre-create artist lists for agents
    ellipses = []
    torso_arrows = []
    trail_lines = []

    for i in range(n_agents):
        color = cmap(i % 20)
        e = mpatches.Ellipse(
            (0, 0),
            width=2 * frames.chest_depths[i],
            height=2 * frames.shoulder_widths[i],
            angle=0,
            facecolor=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.8,
            zorder=10,
        )
        ax.add_patch(e)
        ellipses.append(e)

        (arrow,) = ax.plot([], [], color="black", linewidth=1.2, solid_capstyle="round", zorder=11)
        torso_arrows.append(arrow)

        (trail,) = ax.plot([], [], color=color, linewidth=1.0, alpha=0.5, zorder=6)
        trail_lines.append(trail)

    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        zorder=20,
    )

    if frames.title:
        ax.set_title(frames.title, fontsize=12)

    def _update(render_idx: int):
        frame_idx = frame_indices[render_idx]
        pos = frames.positions[frame_idx]
        torso = frames.torso_orientations[frame_idx]
        active = frames.active_masks[frame_idx]

        for i in range(n_agents):
            if not active[i]:
                ellipses[i].set_visible(False)
                torso_arrows[i].set_data([], [])
                continue

            ellipses[i].set_visible(True)
            ellipses[i].set_center(pos[i])
            ellipses[i].set_angle(np.degrees(torso[i]))

            # Torso direction indicator
            dx = np.cos(torso[i]) * orientation_length
            dy = np.sin(torso[i]) * orientation_length
            torso_arrows[i].set_data(
                [pos[i, 0], pos[i, 0] + dx],
                [pos[i, 1], pos[i, 1] + dy],
            )

            # Trail — look back over original frames for smooth trails
            if show_trails:
                start = max(0, frame_idx - trail_length * frame_skip)
                trail = frames.positions[start : frame_idx + 1, i]
                trail_lines[i].set_data(trail[:, 0], trail[:, 1])

        t = frame_idx * frames.dt
        n_reached = frames.reached_goal.sum()
        time_text.set_text(
            f"t = {t:.1f}s  |  frame {frame_idx}/{frames.n_frames - 1}"
            f"  |  {n_reached}/{n_agents} reached goal"
        )

        return ellipses + torso_arrows + trail_lines + [time_text]

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=1000 / fps, blit=True)

    # Prefer imageio-ffmpeg (pip-installable, bundles its own binary)
    # over system ffmpeg, with pillow as last-resort fallback.
    writer = None
    try:
        import imageio_ffmpeg

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        plt.rcParams["animation.ffmpeg_path"] = ffmpeg_path
        writer = "ffmpeg"
    except ImportError:
        from matplotlib.animation import FFMpegWriter

        if FFMpegWriter.isAvailable():
            writer = "ffmpeg"

    if writer is not None:
        anim.save(str(output_path), writer=writer, fps=fps, dpi=dpi)
    else:
        logger.warning("ffmpeg not available, falling back to pillow (.gif)")
        output_path = output_path.with_suffix(".gif")
        anim.save(str(output_path), writer="pillow", fps=fps, dpi=dpi)

    plt.close(fig)
    return output_path

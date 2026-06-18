"""Behavioural eval metrics computed post-hoc from a recorded episode.

Consumes an :class:`~crowdrl_env.visualiser.EpisodeFrames` (the same snapshot
structure used for video rendering) and reuses crowdrl-core's collision and
wall-distance routines, so the numbers match exactly what the environment
computes during a step. No simulation/geometry logic is re-implemented here --
the only original arithmetic is realized speed (position deltas / dt) and path
efficiency (net displacement / distance travelled), neither of which has an
existing routine to import.

Metrics target the behaviour artefacts under investigation:

- ``mean_speed`` / ``speed_over_preferred`` -- are agents moving too fast
  rather than navigating cautiously?
- ``wall_contact_rate`` -- how often do agents touch geometry?
- ``agent_collision_rate`` -- how often do agents overlap each other?
- ``goal_rate`` / ``path_efficiency`` -- are they still solving the task well?
- ``freeze_rate`` / ``stuck_agent_frac`` -- are agents deadlocking (still active
  but near-stationary) rather than navigating? Separates the gridlock failure
  mode from the bulldozing one (which shows up as a high collision rate).

Keys that need data absent from the frames (e.g. ``speed_over_preferred``
without ``preferred_speeds``, or wall metrics without geometry) are omitted
rather than guessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from crowdrl_core.collision import compute_min_wall_distances, detect_collisions
from crowdrl_core.geometry import extract_wall_segments
from crowdrl_core.world_state import WorldState

if TYPE_CHECKING:  # avoid importing the matplotlib-heavy visualiser at runtime
    from crowdrl_env.visualiser import EpisodeFrames


def _frame_world(
    frames: EpisodeFrames, f: int, wall_segments: NDArray[np.float64] | None
) -> WorldState:
    """Rebuild the minimal WorldState that frame ``f`` needs for the reused
    collision / wall-distance functions.

    Only the fields those functions actually read are real. ``velocities`` and
    ``masses`` are required by the WorldState shape contract but unused by
    :func:`detect_collisions` and :func:`compute_min_wall_distances`, so they
    are filled with dummies.
    """
    n = frames.n_agents
    return WorldState(
        positions=frames.positions[f],
        velocities=np.zeros((n, 2), dtype=np.float64),
        torso_orientations=frames.torso_orientations[f],
        head_orientations=frames.head_orientations[f],
        shoulder_widths=frames.shoulder_widths,
        chest_depths=frames.chest_depths,
        masses=np.ones(n, dtype=np.float64),
        goal_positions=frames.goal_positions,
        wall_segments=wall_segments,
        active_mask=frames.active_masks[f],
    )


def compute_episode_metrics(
    frames: EpisodeFrames, *, freeze_speed: float = 0.1
) -> dict[str, float]:
    """Compute behavioural metrics for one recorded episode.

    Parameters
    ----------
    frames : EpisodeFrames
        Per-frame snapshot data (positions, orientations, body dims, active
        masks, geometry). ``walls`` or ``polygon`` enable the wall metrics;
        ``preferred_speeds`` enables the speed-vs-preferred metrics.
    freeze_speed : float
        Realized speed (m/s) below which an active agent counts as frozen for
        ``freeze_rate`` / ``stuck_agent_frac``. Default 0.1 m/s -- effectively
        stationary relative to the ~1.34 m/s preferred speed.

    Returns
    -------
    dict[str, float]
        Flat dict of scalar metrics; see module docstring.
    """
    n_frames = frames.n_frames
    n_agents = frames.n_agents
    dt = frames.dt
    # Effective body radius = larger semi-axis, matching the environment's wall
    # model (collision.py wall repulsion uses max(shoulder_width, chest_depth)).
    radius = np.maximum(frames.shoulder_widths, frames.chest_depths)  # (N,)

    metrics: dict[str, float] = {
        "n_agents": float(n_agents),
        "episode_length": float(n_frames),
        "goal_rate": float(frames.reached_goal.mean()) if n_agents else 0.0,
    }
    if n_frames < 2 or n_agents == 0:
        return metrics

    # --- realized speed from position deltas (active in both endpoints) ---
    step_disp = np.linalg.norm(np.diff(frames.positions, axis=0), axis=-1)  # (F-1, N)
    active_pair = frames.active_masks[:-1] & frames.active_masks[1:]  # (F-1, N)
    step_speed = step_disp / dt
    speeds = step_speed[active_pair]
    if speeds.size:
        metrics["mean_speed"] = float(speeds.mean())
        metrics["p95_speed"] = float(np.percentile(speeds, 95))

        preferred = frames.preferred_speeds
        if preferred is not None:
            pref = np.broadcast_to(np.asarray(preferred, dtype=np.float64), step_speed.shape)
            pref = np.maximum(pref[active_pair], 1e-6)
            metrics["speed_over_preferred"] = float((speeds / pref).mean())
            metrics["frac_steps_above_preferred"] = float((speeds > pref).mean())

    # --- freeze / deadlock: active agents that are near-stationary ---
    # The env deactivates an agent the instant it reaches its goal, so an agent
    # that is still ACTIVE yet barely moving is genuinely stuck, not merely slow.
    # This is the gridlock signature -- distinct from the bulldozing mode, which
    # surfaces as a high collision rate instead.
    if speeds.size:
        metrics["freeze_rate"] = float((speeds < freeze_speed).mean())
        # Terminal deadlock: agents that never reached goal and were
        # near-stationary over the final quarter of the episode.
        not_done = ~frames.reached_goal  # (N,)
        if bool(not_done.any()):
            window = max(1, (n_frames - 1) // 4)
            tail_active = active_pair[-window:]  # (window, N)
            tail_cnt = tail_active.sum(axis=0)  # (N,)
            tail_sum = np.where(tail_active, step_speed[-window:], 0.0).sum(axis=0)
            tail_mean = np.where(tail_cnt > 0, tail_sum / np.maximum(tail_cnt, 1), np.inf)
            stuck = not_done & (tail_cnt > 0) & (tail_mean < freeze_speed)
            metrics["stuck_agent_frac"] = float(stuck.sum()) / n_agents

    # --- wall segments: reuse what frames carry, else extract from polygon ---
    wall_segments: NDArray[np.float64] | None = frames.walls
    if wall_segments is None and frames.polygon is not None:
        wall_segments = extract_wall_segments(frames.polygon)
    if wall_segments is not None:
        wall_segments = np.asarray(wall_segments, dtype=np.float64)
        if wall_segments.size == 0:
            wall_segments = None

    # --- per-frame wall-contact + agent-collision (reuse core routines) ---
    wall_contact_hits = 0
    wall_prox_hits = 0
    coll_agent_hits = 0
    active_steps = 0
    for f in range(n_frames):
        active = frames.active_masks[f]
        n_active = int(active.sum())
        if n_active == 0:
            continue
        active_steps += n_active
        world = _frame_world(frames, f, wall_segments)
        if wall_segments is not None:
            wall_dist = compute_min_wall_distances(world)  # (N,)
            wall_contact_hits += int(((wall_dist < radius) & active).sum())
            wall_prox_hits += int(((wall_dist < 1.5 * radius) & active).sum())
        if n_active >= 2:
            involved: set[int] = set()
            for i, j, _overlap in detect_collisions(world):
                involved.add(i)
                involved.add(j)
            coll_agent_hits += len(involved)

    if active_steps:
        if wall_segments is not None:
            metrics["wall_contact_rate"] = wall_contact_hits / active_steps
            metrics["wall_proximity_rate"] = wall_prox_hits / active_steps
        metrics["agent_collision_rate"] = coll_agent_hits / active_steps

    # --- path efficiency: net displacement / distance travelled, per agent ---
    path_len = (step_disp * active_pair).sum(axis=0)  # (N,)
    net_disp = np.linalg.norm(frames.positions[-1] - frames.positions[0], axis=-1)  # (N,)
    moved = path_len > 1e-6
    if moved.any():
        eff = np.clip(net_disp[moved] / path_len[moved], 0.0, 1.0)
        metrics["path_efficiency"] = float(eff.mean())

    return metrics


def aggregate_metrics(per_episode: list[dict[str, float]]) -> dict[str, float]:
    """Mean each metric across episodes (over the episodes where it is present)."""
    if not per_episode:
        return {}
    keys: set[str] = set()
    for episode in per_episode:
        keys.update(episode)
    out: dict[str, float] = {}
    for key in keys:
        values = [episode[key] for episode in per_episode if key in episode]
        if values:
            out[key] = float(np.mean(values))
    return out

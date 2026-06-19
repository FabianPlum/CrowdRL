"""Export CrowdRL episode trajectories to a Kinora-compatible HDF5 file.

Writes the *pedpy "ped data archive" HDF5* format (DOI 10.34735/ped.2020.3) -- the
format the Kinora Blender visualiser (``/home/fabi/dev/Kinora``) loads via pedpy.
The same files are also readable by ``pedpy.io.load_trajectory_from_ped_data_archive_hdf5``,
so they double as a generic, tool-agnostic trajectory artefact for the broader
CrowdRL system, not just for Kinora.

The hard contract a reader requires (everything else is carried harmlessly):

* root attribute ``wkt_geometry`` -- a Shapely-Polygon WKT *with holes*
  (exterior = walkable area, holes = obstacles). pedpy's ``WalkableArea`` validates
  that every hole is covered by the exterior, so a polygon-less fallback must be a
  hole-free bounding box (see :func:`_resolve_wkt`).
* dataset ``trajectory`` carrying at least the fields ``{id, frame, x, y}`` plus a
  float ``fps`` attribute. ``id`` is 1-indexed, ``frame`` 0-indexed, coords in metres.

On top of that this writer emits, for richer visualisation / downstream analysis:

* extra ``trajectory`` columns (``z``, and optionally ``torso_angle`` / ``head_angle``)
  -- ignored by pedpy/Kinora's required-subset check, available to any consumer.
* ``position_data`` (``id, frame, color``) -- per-agent per-frame scalar in [0, 1]
  driving Kinora's colour ramp (default: speed as a fraction of preferred speed).
* ``personal_details`` -- per-agent static metadata (body dims, goal, reached-goal).
* provenance root attributes + a JSON ``metadata`` blob.

Two entry points: :func:`write_episode_h5` (high level, takes an
:class:`crowdrl_env.visualiser.EpisodeFrames`) and :func:`write_trajectory_h5`
(low level, takes raw arrays -- usable from any trajectory source).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from shapely.geometry import Polygon

    from crowdrl_env.visualiser import EpisodeFrames

logger = logging.getLogger(__name__)

# Provenance defaults written as root attributes (mirrors the ped-data-archive
# example files so the output is self-describing and tool-recognisable).
_ARCHIVE_DOI = "10.34735/ped.2020.3"
_FILE_VERSION = "1.0.0"
_SOURCE = "crowdrl"

# Speed-colour head-room: agents at <= preferred*this map across the full [0, 1]
# ramp, so the common case (walking near preferred speed) uses the colour range
# well rather than saturating.
_SPEED_COLOR_HEADROOM = 1.5

# Human-readable descriptions attached as per-field attrs (self-documenting files,
# matching the archive convention). Only applied to fields actually present.
_FIELD_DESCRIPTIONS: dict[str, str] = {
    "id": "unique identifier for pedestrian (1-indexed)",
    "frame": "frame number (0-indexed)",
    "x": "pedestrian x-coordinate (meter [m])",
    "y": "pedestrian y-coordinate (meter [m])",
    "z": "pedestrian z-coordinate (meter [m])",
    "torso_angle": "torso orientation (radians, 0=+x, CCW positive)",
    "head_angle": "head orientation, absolute (radians, 0=+x, CCW positive)",
    "color": "per-agent scalar in [0, 1] for colour mapping",
    "marker_id": "per-agent marker identifier",
    "shoulder_width": "collision-ellipse half-width perpendicular to torso (meter [m])",
    "chest_depth": "collision-ellipse half-depth along torso-forward axis (meter [m])",
    "preferred_speed": "preferred walking speed (meter/second [m/s])",
    "goal_x": "goal x-coordinate (meter [m])",
    "goal_y": "goal y-coordinate (meter [m])",
    "reached_goal": "1 if the agent reached its goal by episode end, else 0",
}


@dataclass
class OptionalChannels:
    """Optional per-row / per-agent channels carried into the archive.

    Bundles everything beyond the mandatory ``{id, frame, x, y}`` so the writer
    signature stays small and callers can build channels incrementally.
    """

    z: float | NDArray[np.float64] = 0.0
    """Height column. Scalar, ``(n_agents,)`` or ``(n_frames, n_agents)``. CrowdRL
    is 2D so the default 0.0 is honest; Kinora ignores z for placement anyway."""

    extra_fields: dict[str, NDArray] | None = None
    """Name -> ``(n_frames, n_agents)`` array, appended as extra ``trajectory``
    columns. Harmless to pedpy/Kinora (outside the required field subset)."""

    color_scalar: NDArray[np.float64] | None = None
    """``(n_frames, n_agents)`` scalar, clipped to [0, 1] and written as
    ``position_data`` for Kinora's per-agent colour ramp."""

    per_agent_meta: dict[str, NDArray] | None = None
    """Name -> ``(n_agents,)`` array, written as a ``personal_details`` dataset."""

    field_descriptions: dict[str, str] | None = None
    """Optional overrides/additions to :data:`_FIELD_DESCRIPTIONS`."""


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _resolve_wkt(geometry: "Polygon | str | None", walls: NDArray | None = None) -> str:
    """Resolve a walkable-area geometry to a ``wkt_geometry`` string.

    Accepts a Shapely ``Polygon`` (``.wkt`` gives the exact ``POLYGON ((exterior),
    (hole), ...)`` form readers want) or an already-serialised WKT string. When no
    polygon is available but wall segments are, emits a hole-free **bounding-box**
    polygon with a warning -- the only reconstruction guaranteed to pass pedpy's
    ``WalkableArea`` hole-coverage/non-degeneracy validation (segment-soup can yield
    holes the exterior does not cover, making the file un-loadable in Kinora).
    """
    if isinstance(geometry, str):
        return geometry
    if geometry is not None:
        return geometry.wkt
    if walls is not None:
        walls = np.asarray(walls, dtype=np.float64)
        if walls.size:
            from shapely.geometry import box

            flat = walls.reshape(-1, 2)
            mn = flat.min(axis=0)
            mx = flat.max(axis=0)
            logger.warning(
                "kinora_export: no walkable polygon available; emitting a bounding-box "
                "walkable area [%.2f, %.2f]-[%.2f, %.2f] with NO obstacles.",
                mn[0],
                mn[1],
                mx[0],
                mx[1],
            )
            return box(float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])).wkt
    raise ValueError(
        "write_trajectory_h5 needs a Shapely Polygon, a WKT string, or wall segments "
        "to build the required `wkt_geometry`."
    )


# ---------------------------------------------------------------------------
# Structured-array builders
# ---------------------------------------------------------------------------


def _field_np_dtype(arr: NDArray) -> str:
    """Pick a portable structured-field dtype for *arr* (int/bool -> i8, else f8)."""
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
        return "<i8"
    return "<f8"


def _gather_z(z: float | NDArray, fi: NDArray, ai: NDArray) -> NDArray[np.float64]:
    """Broadcast the height channel to one value per emitted (frame, agent) row."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 0:
        return np.full(fi.shape[0], float(z), dtype=np.float64)
    if z.ndim == 1:  # (n_agents,)
        return z[ai]
    return z[fi, ai]  # (n_frames, n_agents)


def _build_trajectory_rows(
    positions: NDArray[np.float64],
    ids: NDArray[np.int64],
    keep: NDArray[np.bool_],
    z: float | NDArray,
    extra_fields: dict[str, NDArray] | None,
    color_scalar: NDArray | None,
) -> tuple[NDArray, NDArray | None]:
    """Emit the ``trajectory`` (and optional ``position_data``) structured arrays.

    One row per kept ``(frame, agent)`` cell, sorted frame-major then by id (the
    archive convention). ``id`` is taken from *ids* (1-indexed by default); ``frame``
    is the 0-based index into *positions* (already contiguous after any subsample).
    """
    extra_fields = extra_fields or {}

    fi, ai = np.nonzero(keep)
    order = np.lexsort((ids[ai], fi))  # primary: frame, secondary: id
    fi, ai = fi[order], ai[order]

    traj_dtype = np.dtype(
        [("id", "<i8"), ("frame", "<i8"), ("x", "<f8"), ("y", "<f8"), ("z", "<f8")]
        + [(name, _field_np_dtype(arr)) for name, arr in extra_fields.items()]
    )
    traj = np.empty(fi.shape[0], dtype=traj_dtype)
    traj["id"] = ids[ai]
    traj["frame"] = fi
    traj["x"] = positions[fi, ai, 0]
    traj["y"] = positions[fi, ai, 1]
    traj["z"] = _gather_z(z, fi, ai)
    for name, arr in extra_fields.items():
        traj[name] = np.asarray(arr)[fi, ai]

    position_data = None
    if color_scalar is not None:
        pd_dtype = np.dtype([("id", "<i8"), ("frame", "<i8"), ("color", "<f8")])
        position_data = np.empty(fi.shape[0], dtype=pd_dtype)
        position_data["id"] = ids[ai]
        position_data["frame"] = fi
        position_data["color"] = np.clip(np.asarray(color_scalar)[fi, ai], 0.0, 1.0)

    return traj, position_data


def _build_personal_details(
    ids: NDArray[np.int64], per_agent_meta: dict[str, NDArray] | None
) -> NDArray | None:
    """Build the per-agent ``personal_details`` structured array, ``id`` first."""
    if not per_agent_meta:
        return None
    n = ids.shape[0]
    fields = [("id", "<i8")] + [
        (name, _field_np_dtype(arr)) for name, arr in per_agent_meta.items()
    ]
    out = np.empty(n, dtype=np.dtype(fields))
    out["id"] = ids
    for name, arr in per_agent_meta.items():
        out[name] = np.asarray(arr)
    return out


# ---------------------------------------------------------------------------
# Low-level writer
# ---------------------------------------------------------------------------


def write_trajectory_h5(
    path: str | Path,
    positions: NDArray[np.float64],
    geometry: "Polygon | str",
    fps: float,
    *,
    ids: NDArray[np.int64] | None = None,
    active_mask: NDArray[np.bool_] | None = None,
    drop_inactive: bool = True,
    channels: OptionalChannels | None = None,
    metadata: dict | None = None,
    compression: str | None = "gzip",
    run_name: str = "crowdrl_episode",
    start_datetime: datetime | None = None,
) -> Path:
    """Write a pedpy/Kinora-compatible "ped data archive" HDF5 file.

    Parameters
    ----------
    path
        Output ``.h5`` path. Parent directories are created.
    positions
        ``(n_frames, n_agents, 2)`` positions in metres, time-major.
    geometry
        Walkable area as a Shapely ``Polygon`` (with holes for obstacles) or a WKT
        string. Becomes the required root ``wkt_geometry`` attribute.
    fps
        Frame rate written as the ``trajectory`` ``fps`` attribute (float).
    ids
        ``(n_agents,)`` agent identifiers. Default ``1..n_agents`` (1-indexed, the
        archive convention).
    active_mask
        ``(n_frames, n_agents)`` bool. When ``drop_inactive`` is True, only rows
        where this is True are emitted (so agents that reach their goal exit the
        trajectory, like real pedestrians leaving the scene). ``None`` => all active.
    drop_inactive
        Whether to honour *active_mask* (see above). When False, every (frame, agent)
        cell is emitted (dense).
    channels
        Optional extra columns / per-agent metadata / colour channel (see
        :class:`OptionalChannels`).
    metadata
        JSON-serialised into the root ``metadata`` attribute.
    compression
        h5py compression for the (large) ``trajectory`` / ``position_data`` datasets.
    run_name, start_datetime
        Provenance: ``run_name`` root attr, and ``start_date`` / ``start_time`` split
        from *start_datetime* (default: now).

    Returns
    -------
    Path
        The written file path.
    """
    import h5py

    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError(f"positions must be (n_frames, n_agents, 2); got {positions.shape}.")
    n_frames, n_agents = positions.shape[:2]

    if ids is None:
        ids = np.arange(1, n_agents + 1, dtype=np.int64)
    else:
        ids = np.asarray(ids, dtype=np.int64)
        if ids.shape != (n_agents,):
            raise ValueError(f"ids must be ({n_agents},); got {ids.shape}.")

    channels = channels or OptionalChannels()

    if active_mask is None or not drop_inactive:
        keep = np.ones((n_frames, n_agents), dtype=bool)
    else:
        keep = np.asarray(active_mask, dtype=bool)
        if keep.shape != (n_frames, n_agents):
            raise ValueError(f"active_mask must be ({n_frames}, {n_agents}); got {keep.shape}.")
    if not keep.any():
        raise ValueError(
            "No active (frame, agent) cells to write -- refusing to write an empty trajectory."
        )

    wkt = _resolve_wkt(geometry)
    traj, position_data = _build_trajectory_rows(
        positions, ids, keep, channels.z, channels.extra_fields, channels.color_scalar
    )
    personal_details = _build_personal_details(ids, channels.per_agent_meta)

    descriptions = dict(_FIELD_DESCRIPTIONS)
    if channels.field_descriptions:
        descriptions.update(channels.field_descriptions)

    def _describe(dataset, names) -> None:
        for name in names:
            if name in descriptions:
                dataset.attrs[name] = descriptions[name]

    dt = start_datetime or datetime.now()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["wkt_geometry"] = wkt
        f.attrs["archive_doi"] = _ARCHIVE_DOI
        f.attrs["file_version"] = _FILE_VERSION
        f.attrs["source"] = _SOURCE
        f.attrs["run_name"] = run_name
        f.attrs["experiment_title"] = run_name
        f.attrs["start_date"] = dt.strftime("%Y-%m-%d")
        f.attrs["start_time"] = dt.strftime("%H:%M:%S")
        if metadata:
            f.attrs["metadata"] = json.dumps(metadata, default=str)

        traj_ds = f.create_dataset("trajectory", data=traj, compression=compression)
        traj_ds.attrs["fps"] = float(fps)
        _describe(traj_ds, traj.dtype.names)

        if position_data is not None:
            pos_ds = f.create_dataset("position_data", data=position_data, compression=compression)
            _describe(pos_ds, position_data.dtype.names)

        if personal_details is not None:
            pd_ds = f.create_dataset("personal_details", data=personal_details)
            _describe(pd_ds, personal_details.dtype.names)

    logger.info(
        "kinora_export: wrote %s (%d rows, %d agents, %d frames, fps=%.3g)",
        path,
        traj.shape[0],
        n_agents,
        n_frames,
        fps,
    )
    return path


# ---------------------------------------------------------------------------
# High-level adapter (EpisodeFrames -> HDF5)
# ---------------------------------------------------------------------------


def _dense_speed(positions: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
    """Per-frame agent speed (m/s) from position deltas, computed on dense arrays.

    Frame 0 is forward-filled from frame 1 (no preceding delta). Computing on the
    full dense array -- before any active-mask drop -- avoids deltas taken across a
    gap of dropped rows.
    """
    n_frames = positions.shape[0]
    speed = np.zeros(positions.shape[:2], dtype=np.float64)
    if n_frames >= 2:
        step = np.linalg.norm(np.diff(positions, axis=0), axis=-1) / dt
        speed[1:] = step
        speed[0] = step[0]
    return speed


def _speed_color(
    speed: NDArray[np.float64], preferred_speeds: NDArray | None
) -> NDArray[np.float64]:
    """Map speed to a [0, 1] colour scalar: fraction of preferred speed if known,
    else normalised by the max observed speed."""
    if preferred_speeds is not None:
        pref = np.asarray(preferred_speeds, dtype=np.float64)
        denom = np.maximum(pref * _SPEED_COLOR_HEADROOM, 1e-6)
        return np.clip(speed / denom[None, :], 0.0, 1.0)
    return np.clip(speed / max(float(speed.max()), 1e-6), 0.0, 1.0)


def write_episode_h5(
    frames: "EpisodeFrames",
    path: str | Path,
    *,
    fps: float | None = None,
    frame_skip: int = 1,
    include_orientation: bool = True,
    color_by: str | None = "speed",
    drop_inactive: bool = True,
    include_goal_markers: bool = True,
    metadata: dict | None = None,
    compression: str | None = "gzip",
) -> Path:
    """Write an :class:`crowdrl_env.visualiser.EpisodeFrames` to a Kinora HDF5 file.

    Parameters
    ----------
    frames
        Episode snapshot data from
        :func:`crowdrl_env.visualiser.collect_episode_frames`.
    path
        Output ``.h5`` path.
    fps
        Playback frame rate. Default ``1 / frames.dt`` (real-time, every sim frame).
        Scaled by ``1 / frame_skip`` when subsampling.
    frame_skip
        Keep every *frame_skip*-th simulation frame (default 1 = all). Frames are
        re-indexed to a contiguous 0-based axis so playback stays real-time.
    include_orientation
        Carry ``torso_angle`` / ``head_angle`` as extra ``trajectory`` columns
        (harmless to Kinora's required-field check; useful for heading analysis).
    color_by
        Per-agent colour channel (``position_data``). ``"speed"`` (default) colours
        by speed as a fraction of preferred speed; ``None`` omits the channel.
    drop_inactive
        Drop rows for agents that have reached their goal / gone inactive, so they
        exit the trajectory like real pedestrians.
    include_goal_markers
        Write per-agent static metadata (body dims, goal, reached-goal) into
        ``personal_details``.
    metadata
        Extra provenance merged into the root ``metadata`` JSON blob.
    compression
        h5py compression for the trajectory datasets.
    """
    positions = np.asarray(frames.positions, dtype=np.float64)
    n_frames, n_agents = positions.shape[:2]
    dt = float(getattr(frames, "dt", 0.01))
    if fps is None:
        fps = 1.0 / dt

    # Derived speed must be computed on the dense arrays, BEFORE subsample/mask.
    speed = _dense_speed(positions, dt)
    torso = np.asarray(frames.torso_orientations, dtype=np.float64)
    head = np.asarray(frames.head_orientations, dtype=np.float64)
    active = None if frames.active_masks is None else np.asarray(frames.active_masks, dtype=bool)

    if frame_skip > 1:
        sl = slice(None, None, frame_skip)
        positions = positions[sl]
        speed = speed[sl]
        torso = torso[sl]
        head = head[sl]
        if active is not None:
            active = active[sl]
        fps = fps / frame_skip
        n_frames = positions.shape[0]

    extra_fields: dict[str, NDArray] = {}
    if include_orientation:
        extra_fields["torso_angle"] = torso
        extra_fields["head_angle"] = head

    color_scalar = None
    if color_by == "speed":
        color_scalar = _speed_color(speed, frames.preferred_speeds)
    elif color_by is not None:
        raise ValueError(f"Unsupported color_by={color_by!r} (expected 'speed' or None).")

    per_agent_meta: dict[str, NDArray] = {}
    if include_goal_markers:
        ids = np.arange(1, n_agents + 1, dtype=np.int64)
        goal = np.asarray(frames.goal_positions, dtype=np.float64)
        per_agent_meta = {
            "marker_id": ids,
            "shoulder_width": np.asarray(frames.shoulder_widths, dtype=np.float64),
            "chest_depth": np.asarray(frames.chest_depths, dtype=np.float64),
            "goal_x": goal[:, 0],
            "goal_y": goal[:, 1],
            "reached_goal": np.asarray(frames.reached_goal).astype(np.int64),
        }
        if frames.preferred_speeds is not None:
            per_agent_meta["preferred_speed"] = np.asarray(
                frames.preferred_speeds, dtype=np.float64
            )

    channels = OptionalChannels(
        z=0.0,
        extra_fields=extra_fields or None,
        color_scalar=color_scalar,
        per_agent_meta=per_agent_meta or None,
    )

    wkt = _resolve_wkt(getattr(frames, "polygon", None), getattr(frames, "walls", None))

    merged_metadata = {
        "n_agents": int(n_agents),
        "n_frames_emitted": int(n_frames),
        "dt": dt,
        "fps": float(fps),
        "source": _SOURCE,
    }
    if metadata:
        merged_metadata.update(metadata)

    run_name = frames.title or "crowdrl_episode"

    return write_trajectory_h5(
        path,
        positions,
        wkt,
        fps,
        active_mask=active,
        drop_inactive=drop_inactive,
        channels=channels,
        metadata=merged_metadata,
        compression=compression,
        run_name=run_name,
    )

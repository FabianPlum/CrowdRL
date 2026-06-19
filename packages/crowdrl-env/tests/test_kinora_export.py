"""Tests for the Kinora/pedpy HDF5 trajectory exporter.

Builds a small synthetic :class:`EpisodeFrames`, writes it, reads it back, and
asserts the exact pedpy/Kinora "ped data archive" contract (the same datasets and
attributes Kinora loads). When pedpy is importable, also round-trips through its
loaders -- the literal calls Kinora makes.
"""

from __future__ import annotations

import dataclasses
import json
import logging

import numpy as np
import pytest

pytest.importorskip("h5py")
import h5py  # noqa: E402
from shapely import wkt as shp_wkt  # noqa: E402
from shapely.geometry import box  # noqa: E402

from crowdrl_env.kinora_export import write_episode_h5  # noqa: E402
from crowdrl_env.visualiser import EpisodeFrames  # noqa: E402

N_FRAMES = 5
N_AGENTS = 3


def _make_frames() -> EpisodeFrames:
    """A 3-agent, 5-frame episode. Agent id=2 (index 1) goes inactive at frame 3.

    Geometry is a 10x10 room with a central square obstacle (one hole).
    """
    rng = np.random.default_rng(0)
    positions = np.cumsum(rng.uniform(-0.05, 0.05, size=(N_FRAMES, N_AGENTS, 2)), axis=0)
    positions += np.array([1.0, 1.0])  # keep inside the room

    torso = rng.uniform(-np.pi, np.pi, size=(N_FRAMES, N_AGENTS))
    head = rng.uniform(-np.pi, np.pi, size=(N_FRAMES, N_AGENTS))

    active = np.ones((N_FRAMES, N_AGENTS), dtype=bool)
    active[3:, 1] = False  # agent index 1 reaches its goal at frame 3

    polygon = box(0.0, 0.0, 10.0, 10.0).difference(box(4.0, 4.0, 6.0, 6.0))

    return EpisodeFrames(
        positions=positions,
        torso_orientations=torso,
        head_orientations=head,
        shoulder_widths=np.array([0.22, 0.20, 0.24]),
        chest_depths=np.array([0.12, 0.11, 0.13]),
        goal_positions=np.array([[9.0, 9.0], [8.0, 1.0], [1.0, 9.0]]),
        active_masks=active,
        reached_goal=np.array([False, True, False]),
        polygon=polygon,
        preferred_speeds=np.array([1.3, 1.4, 1.2]),
        dt=0.01,
        title="test_episode",
    )


def test_writes_pedpy_contract(tmp_path):
    frames = _make_frames()
    path = write_episode_h5(frames, tmp_path / "ep.h5")

    with h5py.File(path, "r") as f:
        # (1) Geometry: loads as a valid polygon with the one obstacle hole.
        poly = shp_wkt.loads(f.attrs["wkt_geometry"])
        assert poly.is_valid
        assert len(poly.interiors) == 1

        traj = f["trajectory"]
        names = set(traj.dtype.names)
        # (2) Required field subset + float fps attr (dt=0.01 -> 100 fps).
        assert {"id", "frame", "x", "y"} <= names
        assert "fps" in traj.attrs
        assert float(traj.attrs["fps"]) == pytest.approx(100.0)

        rows = traj[:]
        # (3) id 1-indexed, frame 0-indexed and contiguous.
        assert rows["id"].min() == 1
        assert rows["id"].max() == N_AGENTS
        assert rows["frame"].min() == 0
        assert set(np.unique(rows["frame"])) == set(range(N_FRAMES))

        # (4) drop_inactive (default): one row per active cell; agent 2 stops early.
        assert rows.shape[0] == int(frames.active_masks.sum()) == 13
        agent2 = rows[rows["id"] == 2]
        assert agent2["frame"].max() == 2  # no rows at/after frame 3
        # (5) an always-active agent keeps all frames.
        assert (rows["id"] == 1).sum() == N_FRAMES

        # (6) orientation carried as extra trajectory columns, values match source.
        assert {"torso_angle", "head_angle"} <= names
        r00 = rows[(rows["frame"] == 0) & (rows["id"] == 1)][0]
        assert r00["torso_angle"] == pytest.approx(frames.torso_orientations[0, 0])
        assert r00["head_angle"] == pytest.approx(frames.head_orientations[0, 0])

        # (7) z column present, all zero (CrowdRL is 2D).
        assert "z" in names
        assert np.all(rows["z"] == 0.0)

        # (8) personal_details: per-agent statics, id 1..N.
        pers = f["personal_details"][:]
        assert pers.shape == (N_AGENTS,)
        assert {"shoulder_width", "goal_x", "goal_y", "reached_goal"} <= set(pers.dtype.names)
        assert list(pers["id"]) == [1, 2, 3]
        assert pers["shoulder_width"][0] == pytest.approx(frames.shoulder_widths[0])

        # (9) position_data colour channel in [0, 1].
        colour = f["position_data"][:]
        assert set(colour.dtype.names) == {"id", "frame", "color"}
        assert colour["color"].min() >= 0.0
        assert colour["color"].max() <= 1.0

        # (10) provenance attrs + JSON metadata round-trip.
        assert f.attrs["archive_doi"] == "10.34735/ped.2020.3"
        assert f.attrs["file_version"] == "1.0.0"
        meta = json.loads(f.attrs["metadata"])
        assert meta["source"] == "crowdrl"
        assert meta["n_agents"] == N_AGENTS


def test_drop_inactive_false_is_dense(tmp_path):
    frames = _make_frames()
    path = write_episode_h5(frames, tmp_path / "dense.h5", drop_inactive=False)
    with h5py.File(path, "r") as f:
        rows = f["trajectory"][:]
    assert rows.shape[0] == N_FRAMES * N_AGENTS


def test_no_orientation_no_colour(tmp_path):
    frames = _make_frames()
    path = write_episode_h5(
        frames, tmp_path / "minimal.h5", include_orientation=False, color_by=None
    )
    with h5py.File(path, "r") as f:
        names = set(f["trajectory"].dtype.names)
        assert "torso_angle" not in names
        assert "position_data" not in f


def test_frame_skip_reindexes_and_scales_fps(tmp_path):
    frames = _make_frames()
    path = write_episode_h5(frames, tmp_path / "skip.h5", frame_skip=2)
    with h5py.File(path, "r") as f:
        rows = f["trajectory"][:]
        # frames 0,2,4 -> re-indexed to a contiguous 0,1,2 axis.
        assert set(np.unique(rows["frame"])) == {0, 1, 2}
        # fps scaled by 1/frame_skip (100 / 2).
        assert float(f["trajectory"].attrs["fps"]) == pytest.approx(50.0)


def test_bbox_fallback_when_polygon_missing(tmp_path, caplog):
    frames = _make_frames()
    walls = np.array(
        [
            [[0.0, 0.0], [10.0, 0.0]],
            [[10.0, 0.0], [10.0, 10.0]],
            [[10.0, 10.0], [0.0, 10.0]],
            [[0.0, 10.0], [0.0, 0.0]],
        ]
    )
    frames = dataclasses.replace(frames, polygon=None, walls=walls)
    with caplog.at_level(logging.WARNING):
        path = write_episode_h5(frames, tmp_path / "bbox.h5")
    assert "bounding-box" in caplog.text
    with h5py.File(path, "r") as f:
        poly = shp_wkt.loads(f.attrs["wkt_geometry"])
    assert poly.is_valid
    assert len(poly.interiors) == 0  # hole-free fallback


def test_empty_episode_raises(tmp_path):
    frames = _make_frames()
    frames = dataclasses.replace(frames, active_masks=np.zeros((N_FRAMES, N_AGENTS), bool))
    with pytest.raises(ValueError, match="empty trajectory"):
        write_episode_h5(frames, tmp_path / "empty.h5")


def test_pedpy_roundtrip_if_available(tmp_path):
    """When pedpy is importable, the file loads through the exact Kinora load path."""
    pedpy = pytest.importorskip("pedpy")
    frames = _make_frames()
    path = write_episode_h5(frames, tmp_path / "pedpy.h5")

    traj = pedpy.load_trajectory_from_ped_data_archive_hdf5(trajectory_file=path)
    assert float(traj.frame_rate) == pytest.approx(100.0)
    assert len(traj.data) == int(frames.active_masks.sum())

    area = pedpy.load_walkable_area_from_ped_data_archive_hdf5(trajectory_file=path)
    assert area is not None

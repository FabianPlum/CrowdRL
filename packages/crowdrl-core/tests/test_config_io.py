"""Tests for the lossless config <-> dict serialization boundary.

These functions are the payload format for the ONNX ``metadata_props``
embedding (issue #7): both the exporter and the deployment reader go through
them, so exactness here is what makes the embedded config trustworthy.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from crowdrl_core.action import ActionConfig
from crowdrl_core.config_io import (
    action_config_from_dict,
    action_config_to_dict,
    obs_config_from_dict,
    obs_config_to_dict,
)
from crowdrl_core.observation import ObsConfig
from crowdrl_core.sensing import RaycastConfig

# A deliberately non-default config exercising every schema axis the YAML
# contract cannot express: raycast geometry, 2-channel rays, and social K.
NON_DEFAULT_OBS = ObsConfig(
    k_neighbours=6,
    raycast=RaycastConfig(n_rays=32, fov_deg=170.0, max_range=8.0, two_channel=True),
    use_navmesh=True,
    use_goal_direction=False,
    use_temporal_memory=True,
    temporal_memory_window=25,
    temporal_memory_max_steps=3000,
    temporal_memory_dt=0.02,
    use_neighbor_memory=True,
    use_neighbor_vel_history=True,
)

NON_DEFAULT_ACTION = ActionConfig(
    max_forward_speed=1.8,
    max_backward_speed=0.4,
    max_heading_change=float(np.radians(4.8)),
    max_torso_change=float(np.radians(4.8)),
    speed_turn_coupling=True,
    turn_pivot_rate=float(np.radians(240.0)),
)


class TestObsConfigRoundTrip:
    def test_round_trip_is_exact(self):
        """Frozen-dataclass equality after to -> from, nested raycast included."""
        assert obs_config_from_dict(obs_config_to_dict(NON_DEFAULT_OBS)) == NON_DEFAULT_OBS

    def test_round_trip_through_json_text(self):
        """The dict must survive real JSON serialization -- that is how it is
        stored in ONNX metadata_props."""
        text = json.dumps(obs_config_to_dict(NON_DEFAULT_OBS))
        assert obs_config_from_dict(json.loads(text)) == NON_DEFAULT_OBS

    def test_obs_dim_preserved(self):
        """The consequence that matters: the rebuilt config derives the same
        observation width."""
        rebuilt = obs_config_from_dict(json.loads(json.dumps(obs_config_to_dict(NON_DEFAULT_OBS))))
        assert rebuilt.obs_dim == NON_DEFAULT_OBS.obs_dim

    def test_defaults_only_round_trips(self):
        assert obs_config_from_dict(obs_config_to_dict(ObsConfig())) == ObsConfig()

    def test_missing_keys_fill_from_defaults(self):
        """Forward compatibility: an artefact exported before a field existed
        reconstructs with that field at its (behaviour-preserving) default."""
        data = obs_config_to_dict(NON_DEFAULT_OBS)
        del data["use_neighbor_vel_history"]
        rebuilt = obs_config_from_dict(data)
        assert rebuilt.use_neighbor_vel_history is False
        assert rebuilt.raycast == NON_DEFAULT_OBS.raycast

    def test_unknown_top_level_key_raises(self):
        """Backward incompatibility must be loud: a newer exporter's field this
        core cannot reproduce."""
        data = obs_config_to_dict(NON_DEFAULT_OBS)
        data["use_quantum_sensing"] = True
        with pytest.raises(ValueError, match="use_quantum_sensing"):
            obs_config_from_dict(data)

    def test_unknown_raycast_key_raises(self):
        data = obs_config_to_dict(NON_DEFAULT_OBS)
        data["raycast"]["lidar_noise"] = 0.1
        with pytest.raises(ValueError, match="lidar_noise"):
            obs_config_from_dict(data)


class TestActionConfigRoundTrip:
    def test_round_trip_is_exact(self):
        assert (
            action_config_from_dict(action_config_to_dict(NON_DEFAULT_ACTION))
            == NON_DEFAULT_ACTION
        )

    def test_round_trip_through_json_text(self):
        """Raw radians survive JSON bit-exactly (floats are IEEE round-trip
        safe in json), so no deg<->rad translation error can creep in."""
        text = json.dumps(action_config_to_dict(NON_DEFAULT_ACTION))
        rebuilt = action_config_from_dict(json.loads(text))
        assert rebuilt == NON_DEFAULT_ACTION
        assert rebuilt.max_heading_change == NON_DEFAULT_ACTION.max_heading_change

    def test_head_limit_and_dt_travel(self):
        """head_limit and dt are part of the motion envelope; they must be in
        the payload even though the YAML schema never carried them."""
        data = action_config_to_dict(NON_DEFAULT_ACTION)
        assert "head_limit" in data
        assert "dt" in data

    def test_unknown_key_raises(self):
        data = action_config_to_dict(NON_DEFAULT_ACTION)
        data["max_jetpack_thrust"] = 9000.0
        with pytest.raises(ValueError, match="max_jetpack_thrust"):
            action_config_from_dict(data)

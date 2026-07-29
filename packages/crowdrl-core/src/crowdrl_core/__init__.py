"""crowdrl-core: shared geometry, perception, and action library for CrowdRL."""

from crowdrl_core.action import ActionConfig, ActionResult, interpret_action
from crowdrl_core.config_io import (
    action_config_from_dict,
    action_config_to_dict,
    obs_config_from_dict,
    obs_config_to_dict,
)
from crowdrl_core.navmesh import shortest_path
from crowdrl_core.observation import ObsConfig, build_observation, build_observations_batch
from crowdrl_core.sensing import RaycastConfig
from crowdrl_core.world_state import NavMesh, WorldState

__all__ = [
    "ActionConfig",
    "ActionResult",
    "NavMesh",
    "ObsConfig",
    "RaycastConfig",
    "WorldState",
    "action_config_from_dict",
    "action_config_to_dict",
    "build_observation",
    "build_observations_batch",
    "interpret_action",
    "obs_config_from_dict",
    "obs_config_to_dict",
    "shortest_path",
]

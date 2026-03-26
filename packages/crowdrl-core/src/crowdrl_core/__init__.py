"""crowdrl-core: shared geometry, perception, and action library for CrowdRL."""

from crowdrl_core.action import ActionConfig, ActionResult, interpret_action
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
    "build_observation",
    "build_observations_batch",
    "interpret_action",
    "shortest_path",
]

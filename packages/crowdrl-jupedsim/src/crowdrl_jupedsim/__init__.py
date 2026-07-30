"""crowdrl-jupedsim: JuPedSim integration adapter for learned CrowdRL policies.

Deploys a policy exported from training as a JuPedSim 2.0 operational model.
Depends only on crowdrl-core, onnxruntime and JuPedSim -- never on PyTorch,
crowdrl-env or crowdrl-train. The only artefact crossing from the training side
is an ``.onnx`` file.
"""

from typing import TYPE_CHECKING

from crowdrl_jupedsim.policy import (
    ConstantPolicy,
    OnnxPolicy,
    Policy,
    PolicyMetadata,
    resolve_configs,
    resolve_dynamics,
)

if TYPE_CHECKING:  # pragma: no cover
    from crowdrl_jupedsim.lockstep import LockstepPolicyModel, native_batch_step
    from crowdrl_jupedsim.model import CrowdRLAgentState, LearnedPolicyModel, TemporalMemory

__all__ = [
    "ConstantPolicy",
    "CrowdRLAgentState",
    "LearnedPolicyModel",
    "LockstepPolicyModel",
    "OnnxPolicy",
    "Policy",
    "PolicyMetadata",
    "TemporalMemory",
    "native_batch_step",
    "resolve_configs",
    "resolve_dynamics",
]

_MODEL_EXPORTS = {"CrowdRLAgentState", "LearnedPolicyModel", "TemporalMemory"}
_LOCKSTEP_EXPORTS = {"LockstepPolicyModel", "native_batch_step"}


def __getattr__(name: str):
    """Import the operational-model classes lazily (PEP 562).

    ``crowdrl_jupedsim.model`` (and ``.lockstep``, which builds on it)
    requires a JuPedSim 2.0 source build, which is provided out-of-band (see
    pyproject.toml). Deferring the import keeps the jupedsim-free surface --
    ``OnnxPolicy``, ``PolicyMetadata``, ``resolve_configs`` -- importable
    everywhere (CI has no jupedsim), while touching the model classes still
    raises the helpful install guidance from ``crowdrl_jupedsim.model``.
    """
    if name in _MODEL_EXPORTS:
        from crowdrl_jupedsim import model

        return getattr(model, name)
    if name in _LOCKSTEP_EXPORTS:
        from crowdrl_jupedsim import lockstep

        return getattr(lockstep, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Tests for adapter defaults that do not require a JuPedSim build."""

from __future__ import annotations

import subprocess
import sys


def test_lockstep_uses_agent_state_chest_depth_default():
    """The lockstep fallback must track the public agent-state default."""
    code = """
import sys
import types

jupedsim = types.ModuleType("jupedsim")
jupedsim.__path__ = []
models = types.ModuleType("jupedsim.models")
models.__path__ = []
custom_model = types.ModuleType("jupedsim.models.custom_model")
custom_model.CustomOperationalModel = type("CustomOperationalModel", (), {})
sys.modules.update({
    "jupedsim": jupedsim,
    "jupedsim.models": models,
    "jupedsim.models.custom_model": custom_model,
})

from crowdrl_jupedsim.lockstep import _Row
from crowdrl_jupedsim.model import CrowdRLAgentState, DEFAULT_CHEST_DEPTH

state = types.SimpleNamespace(position=(0.0, 0.0))
row = _Row(state, goal=(1.0, 0.0), buf_size=1)
state_default = CrowdRLAgentState.__dataclass_fields__["chest_depth"].default
assert row.chest == state_default == DEFAULT_CHEST_DEPTH
"""

    subprocess.run([sys.executable, "-c", code], check=True)

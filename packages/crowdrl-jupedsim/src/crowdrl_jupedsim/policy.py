"""Policy inference backends for the JuPedSim deployment adapter.

The adapter depends only on an exported ``.onnx`` artefact -- never on PyTorch,
crowdrl-env or crowdrl-train. ``Policy`` is the narrow seam between the
operational model and whatever produces actions, which keeps the adapter
testable without a trained checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class Policy(Protocol):
    """Maps a single observation vector to a raw action vector in [-1, 1]."""

    def __call__(self, obs: NDArray[np.float64]) -> NDArray[np.float64]: ...


class OnnxPolicy:
    """ONNX Runtime policy loaded from an exported CrowdRL checkpoint.

    The exported graph bakes in the frozen observation normalizer and the
    deterministic ``tanh(mean)`` action, so this wrapper is a thin call.

    The policy file is loaded by path and never bundled with the package --
    baseline weights live outside the repository.
    """

    def __init__(self, onnx_path: str | Path, providers: list[str] | None = None) -> None:
        import onnxruntime as ort

        self.path = Path(onnx_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"ONNX policy not found: {self.path}")

        self._session = ort.InferenceSession(
            str(self.path), providers=providers or ["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

        # Static obs width from the graph, when the exporter pinned it. Used to
        # fail loudly on an ObsConfig/checkpoint mismatch instead of silently
        # feeding the policy a differently-shaped world.
        shape = self._session.get_inputs()[0].shape
        last = shape[-1] if shape else None
        self.obs_dim: int | None = last if isinstance(last, int) else None

    def __call__(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        batched = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        out = self._session.run(None, {self._input_name: batched})[0]
        return np.asarray(out, dtype=np.float64).reshape(-1)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"OnnxPolicy(path={self.path.name!r}, obs_dim={self.obs_dim})"


class ConstantPolicy:
    """Returns a fixed action regardless of observation.

    Exists so the adapter's control flow (WorldState assembly, observation
    construction, action interpretation, integration) can be exercised without
    a trained checkpoint.
    """

    def __init__(self, action: NDArray[np.float64] | list[float]) -> None:
        self._action = np.asarray(action, dtype=np.float64)

    def __call__(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._action.copy()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ConstantPolicy(action={self._action.tolist()})"

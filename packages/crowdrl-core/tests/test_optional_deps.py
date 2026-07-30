"""Guards the deployment dependency footprint of crowdrl-core.

The JuPedSim-side install ships an ``.onnx`` plus crowdrl-core and onnxruntime,
and must not drag in the training stack. ``triangle`` is a compiled
triangulation library needed only to *build* a navmesh from a polygon;
deployment consumes an existing geometry through WorldState, so the shared
observation path has to stay importable without it.

A top-level ``import triangle`` anywhere in that path would reintroduce the
dependency silently -- the tests would still pass in this repo, because the dev
environment installs the ``[geometry]`` extra. Hence the subprocess: it checks
what the import actually pulls in, rather than what happens to be in
``sys.modules`` already.
"""

import subprocess
import sys
import textwrap

import pytest

# Modules a deployment install is expected to import without the [geometry] extra.
DEPLOYMENT_MODULES = [
    "crowdrl_core.world_state",
    "crowdrl_core.observation",
    "crowdrl_core.sensing",
    "crowdrl_core.action",
    "crowdrl_core.navmesh",
    "crowdrl_core.geometry",
]


def _import_pulls_in(module: str, package: str) -> bool:
    """Whether importing ``module`` in a fresh interpreter imports ``package``."""
    script = textwrap.dedent(f"""
        import sys
        import {module}  # noqa: F401
        print("YES" if {package!r} in sys.modules else "NO")
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    return result.stdout.strip().endswith("YES")


def test_the_detector_can_actually_detect():
    """Positive control. Without this, a broken detector that always reports
    'not imported' would make every guard below pass vacuously."""
    assert _import_pulls_in("crowdrl_core.geometry", "shapely"), (
        "geometry genuinely imports shapely, so the detector should see it"
    )


@pytest.mark.parametrize("module", DEPLOYMENT_MODULES)
def test_deployment_modules_do_not_import_triangle(module):
    assert not _import_pulls_in(module, "triangle"), (
        f"{module} imports 'triangle' at module level. That is an optional "
        "dependency (crowdrl-core[geometry]) and would be missing from a "
        "JuPedSim deployment install. Import it lazily inside the function "
        "that triangulates, as triangulate_polygon does."
    )


def test_triangulation_still_works_when_the_extra_is_installed():
    """The lazy import must actually resolve, not just defer the failure."""
    import shapely

    from crowdrl_core.geometry import triangulate_polygon

    triangles = triangulate_polygon(shapely.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]))
    assert len(triangles) >= 2, "a square should triangulate into at least two triangles"
    assert all(t.shape == (3, 2) for t in triangles)

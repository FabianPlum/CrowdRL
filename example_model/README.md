# Example model

`policy_r0400.onnx` -- the current best trained policy, shipped as a
**self-describing artefact** (metadata schema v2): its resolved
`ObsConfig`/`ActionConfig`, its trained dynamics (`desired_velocity_weight`,
speed clamp, contact constants) and provenance are embedded in the ONNX
`metadata_props` (issue #7), so the JuPedSim adapter configures itself --
observations, action limits AND physics -- from the file alone:

```python
from crowdrl_jupedsim import LearnedPolicyModel, OnnxPolicy

model = LearnedPolicyModel(OnnxPolicy("example_model/policy_r0400.onnx"))
# no ObsConfig / ActionConfig needed -- and a wrong explicit one would raise
```

Provenance (also readable via `OnnxPolicy(...).metadata.provenance`):

- run: `results_exp_nogoaldir_stable_bigrooms_density_v4`
- checkpoint: `checkpoint_rollout_0400.pt`
- observation: 89D -- `use_goal_direction=False` (navigates by the routed
  waypoint alone), navmesh block, 6D temporal memory
- action: speed/turn-coupled interpreter (240 deg/s pivot), dt=0.01
- dynamics: `desired_velocity_weight=0.8`, speed clamp 3.0 m/s, contact
  stiffness 30000 / damping 500 -- the values the checkpoint trained under.
  Self-configured runs therefore use the TRAINED dynamics; at those, the
  checkpoint deterministically loses one agent per corner/bottleneck
  scenario to its wall-facing absorbing state (a known policy trait, see
  plan/lockstep_parity_analysis.md).

Regenerated (bit-exact vs. the run's original `policy_r0400.onnx`) via:

```bash
uv run python scripts/reexport_onnx.py results_exp_nogoaldir_stable_bigrooms_density_v4 \
    checkpoint_rollout_0400.pt --output example_model/policy_r0400.onnx \
    --verify-against results_exp_nogoaldir_stable_bigrooms_density_v4/policy_r0400.onnx
```

Used by `tests/test_e2e_jupedsim_trained_policy.py` (the self-configured
scenarios run whenever a JuPedSim 2.0 build is on `sys.path`; no environment
variables required).

# Example model

`policy_r0125.onnx` -- the current best trained policy, shipped as a
**self-describing artefact** (metadata schema v2): its resolved
`ObsConfig`/`ActionConfig`, its trained dynamics (`desired_velocity_weight`,
speed clamp, contact constants) and provenance are embedded in the ONNX
`metadata_props` (issue #7), so the JuPedSim adapter configures itself --
observations, action limits AND physics -- from the file alone:

```python
from crowdrl_jupedsim import LearnedPolicyModel, OnnxPolicy

model = LearnedPolicyModel(OnnxPolicy("example_model/policy_r0125.onnx"))
# no ObsConfig / ActionConfig needed -- and a wrong explicit one would raise
```

Provenance (also readable via `OnnxPolicy(...).metadata.provenance`):

- run: `results_exp_jps_routing_ft_r0400` -- a 2-GPU fine-tune of the previous
  best checkpoint (`exp_nogoaldir_stable_bigrooms_density_v4` r0400) under the
  **JuPedSim routing contract** (`use_jupedsim_style_routing: true`: waypoints
  served router-style at the fixed 0.2 m portal inset, `path_deviation` pinned
  to 0.0 -- the signal the deployed `LearnedPolicyModel` actually feeds)
- checkpoint: `checkpoint_rollout_0125.pt` (episode 8745)
- observation: 89D -- `use_goal_direction=False` (navigates by the routed
  waypoint alone), navmesh block, 6D temporal memory
- action: speed/turn-coupled interpreter (240 deg/s pivot), dt=0.01
- dynamics: `desired_velocity_weight=0.8`, speed clamp 3.0 m/s, contact
  stiffness 30000 / damping 500 -- the values the checkpoint trained under, so
  self-configured runs use the TRAINED dynamics. `desired_velocity_weight` is
  read from this run's `config_resolved.yaml` (shipped alongside); the other
  three are not expressible in the YAML schema and were asserted at export
  time (see `dynamics_provenance` in the artefact's provenance).

## Why r0125 and not the final rollout

The fine-tune ran to ~575 of 600 rollouts and was stopped early: from roughly
rollout 150 onward the fixed eval suite regressed, and it never recovered.
Collision rate kept falling while freeze/stuck fractions climbed -- the policy
traded goal completion for collision avoidance and became over-conservative at
high density. Every checkpoint's scorecard was written during the run, so the
artefact is the best-scoring checkpoint rather than the last one.

`scorecard_r0125.json` (shipped) against `scorecard_r0400.json` (the previous
best, kept as the comparison baseline):

| metric | r0400 baseline | **r0125 (shipped)** |
|---|---|---|
| goal_rate | 0.954 | **0.975** |
| agent_collision_rate | 0.116 | **0.093** |
| stuck_agent_frac | 0.159 | **0.032** |
| freeze_rate | 0.099 | **0.084** |
| path_efficiency | 0.917 | **0.920** |

Better on every axis. The high-density scenarios (`composed_hi` at 60-100
agents) remain the weakest and are where the later checkpoints lost ground --
see `plan/CrowdRL_Project_Plan_v9.md` for what that implies for retraining.

Regenerated via:

```bash
uv run python scripts/reexport_onnx.py results_exp_jps_routing_ft_r0400 \
    checkpoint_rollout_0125.pt --output example_model/policy_r0125.onnx \
    --max-velocity-magnitude 3.0 --contact-stiffness 30000 --contact-damping 500
```

The three dynamics flags are required: the script refuses to stamp values the
run's YAML does not record as though they were trained (they would otherwise
come from whatever `CrowdEnvConfig` defaults to at re-export time). Torch
parity against the checkpoint is verified at export.

## Deployment baselines

Measured with this artefact through `LearnedPolicyModel` on a JuPedSim 2.0
source build (see `plan/CrowdRL_Project_Plan_v9.md` for the build recipe):

| scenario | result |
|---|---|
| Corner, 4 agents (jupedsim#1625 geometry) | 4/4 exit, 9.7 s sim |
| Bottleneck, 12 agents, 1.4 m aperture | 12/12 exit, 7.1 s sim |

Pinned as tests in `tests/test_e2e_jupedsim_trained_policy.py` (they run
whenever a JuPedSim 2.0 build is on `sys.path`; no environment variables
required) and demonstrated in `examples/10_jupedsim_learned_model.ipynb`.

Not shipped, kept local (see `.gitignore`): `checkpoint_rollout_0400.pt` (the
baseline this run initialised from) and `viz_r0125_tier3B.mp4` (a regenerable
training-time render).

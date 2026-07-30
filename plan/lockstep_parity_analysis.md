# Byte-exact deployment parity -- analysis and design (2026-07-29)

Goal: make a JuPedSim-hosted run of a CrowdRL policy reproduce the
CrowdRL-native simulation **byte-identically**. This documents the divergence
analysis between the two pipelines, the measurements, and the resulting
design (`crowdrl_jupedsim.lockstep.LockstepPolicyModel`), which achieves it.

## Divergence channel inventory (measured)

Micro-ablations on identical inputs, corner geometry, the shipped
`example_model/policy_r0400.onnx` (89D nogoaldir + navmesh + temporal):

| # | Channel | Measured delta | Byte-parity consequence |
|---|---------|----------------|-------------------------|
| 1 | `build_observation` (per-agent) vs `build_observations_batch` | 3.7e-15 | float reassociation: the SAME builder must be used on both sides |
| 2 | ONNX Runtime batch(N) vs per-row(1) inference | 0.0 exactly | none on this machine; still batch identically for structure |
| 3 | `interpret_action` vs `interpret_actions_batch` | 0.0 exactly | use the batch interpreter on both sides anyway |
| 4 | Wall segments: `extract_wall_segments(polygon)` vs per-agent `line_segments_in_range` | equal min-distances, different set/order | carry the identical full segment array |
| 5 | **Waypoint source: navmesh funnel vs router `next_target`** | **up to 116 deg direction; p_dev 0.29-0.46 vs 0.0** | **dominant channel; must compute the nav block from the same funnel** |
| 6 | Physics application: per-agent (ego row) vs synchronous batch | neighbour velocities pre- vs post-filter in the damping term | run the physics once, batched |
| 7 | Exit removal timing | JuPedSim's exit stage removes agents **2 iterations after** they enter the exit area | apply native removal inside the model |
| 8 | **Heading anchoring: torso-derived (training) vs free-integrated (interactive adapter)** | **unbounded drift between heading and torso** | **interactive-model-only; fixed 2026-07-30 -- anchor to torso** |

Channel 8 was found in the 2026-07-29 end-of-session review and never
affected lockstep. Both training engines pass `torso_orientations` as *both*
`current_headings` and `current_torsos` (`crowd_env.py:334-341`;
`native_batch_step`, which is why byte-parity held and why the byte-parity
test could not have caught this). Heading is therefore *derived* in training:
re-anchored to the previous torso every step, so the commanded velocity
direction never sits more than one per-step delta (~4.8 deg for the shipped
artefact) from the previous torso. `LearnedPolicyModel` instead fed its own
persisted `state.heading` back in and stored `result.new_heading`, letting
the two free-integrate apart without bound and systematically changing the
action->motion mapping vs training. `CrowdRLAgentState.heading` is now
documented as output-only.

Measured effect of closing channel 8 (shipped artefact, trained dynamics
w=0.8, `LearnedPolicyModel`, contact physics on):

| Scenario | Before | After |
|----------|--------|-------|
| Corner, 4 agents | 3/4 exit; 1 pinned at (11.8, 1.7) for all 4000 steps | **4/4 exit**, steps 738/821/903/979 (9.8 s) |
| Bottleneck, 12 agents, 1.4 m | 11/12 exit; 1 pinned before the neck; min pairwise 0.350 m | **12/12 exit** in 7.4 s; min pairwise 0.410 m |

This **retracts the "policy absorbing state" attribution** for the lost
agent in both e2e scenarios and in notebook 10's parity section. The
wall-facing state itself is real (identical obs and action across engines at
the captured state), but what was *driving* agents into it was channel 8, an
adapter defect. No committed scenario reaches it now, so nothing in the repo
currently demonstrates an absorbing state -- treat it as an open question,
not an established policy property, until a scenario reproduces it under
torso-anchored headings.

Channel 5 deserves emphasis: along the corner approach corridor the native
policy sees `path_deviation` of 0.29-0.46 -- **+3 to +4 sigma** under the
baked normalizer statistics -- while the interactive adapter feeds 0.0, and
the waypoint direction disagrees by tens of degrees (funnel apex vs router
portal point). This is not a perturbation; it is a different navigation
signal, and it explains the deployment-side weaving and the earlier
freeze-divergence observations. No post-processing of the router output
(clearance push-off, LOS promotion) reproduces the funnel signal.

## First-divergence experiments

With channels 1-6 equalised (prototype), the corner run at trained dynamics
(w=0.8) was **byte-identical for 849 consecutive steps**; the first
difference was channel 7 alone -- every exit landed exactly +2 steps later
under JuPedSim (its exit stage takes 2 iterations to actually remove an
agent). With native removal applied inside the model (freeze the row the
step it lands in an exit; exclude it from all subsequent batches; keep
serving the frozen state until JuPedSim catches up):

- **Corner, 1082 steps: byte-identical** (`np.array_equal` on every position
  of every agent at every step), identical exit steps {850, 923, 1012, 1082}.
- **Bottleneck, 12 agents, 400 contact-heavy steps: byte-identical**
  (collisions, contact forces, wall projection all exercised).

## Design: `LockstepPolicyModel`

Key insight: JuPedSim's compute-then-apply pass gives every callback the
same pre-step snapshot, which is exactly the synchronous world the native
loop steps. So the model computes **the entire native batched step once per
iteration** on the first callback of a pass (pass boundary = an agent id
seen twice) and serves each agent its precomputed row:

- roster = ego + `other_agents_in_range(geometry-diagonal radius)`, sorted
  by id (= spawn order = native array order);
- `native_batch_step(...)` -- the single shared step function (exported from
  `crowdrl_jupedsim.lockstep`): observations -> one batched ONNX call ->
  batch interpreters -> velocity filter -> contact accelerations -> clamp ->
  integration -> body-clearance wall projection, in the exact
  `CrowdEnv.step` order. The byte-parity test's reference loop calls the
  same function, making identity a structural property of shared code
  rather than a maintenance promise;
- navmesh built from `walkable_geometry` at construction (funnel waypoints +
  true `path_deviation`; requires the `crowdrl-core[geometry]` extra, i.e.
  `triangle`, at deployment -- accepted for validation mode);
- native removal semantics via `exit_geometries` (channel 7);
- temporal-memory ring buffers maintained batch-side per the CrowdEnv
  contract, world-level pass counter as the step count.

Verification lives in `tests/test_lockstep_byte_parity.py`: a corridor run
that completes (byte-equal trajectories AND equal exit steps) and a corner
segment (funnel corners + wall projection), both compared with
`np.array_equal` -- no tolerances.

## Division of labour going forward

- **`LearnedPolicyModel`** stays the interactive/production adapter: router
  waypoint, per-agent callbacks, no `triangle` dependency, tolerant of
  arbitrary journeys. Route-level faithful, not byte-exact -- channels 6
  (neighbour one-step staleness in the contact damping) and 7 (the exit lag
  leaking removed agents into neighbours' observations) remain step-level
  semantic differences by design, and no amount of retraining removes them.
  Optional `waypoint_clearance` mitigates the router's corner-targeting (off
  by default).
- **`LockstepPolicyModel`** is the validation instrument: IAS-7-style
  comparisons, regression baselines, and any claim of the form "JuPedSim
  reproduces the CrowdRL result" should run through it.

## Caveats and follow-ups

1. Byte-identity is machine-scoped: it also relies on ONNX Runtime being
   run-to-run deterministic for the same session/hardware (measured exact
   here, including batch-vs-single). Cross-machine runs should expect
   ULP-level drift.
2. Dynamics parameters (`desired_velocity_weight`, contact constants,
   `max_velocity_magnitude`): **resolved -- metadata schema v2 records them**
   (`crowdrl.dynamics`) and both models self-configure, raising on
   explicit-vs-recorded disagreement. En route we found deployment had been
   clamping at 5.0 m/s vs the training default 3.0 (the YAML schema cannot
   express these fields at all). v1 artefacts still require hand-matching.
3. Fixed per-agent final goals are assumed (single exit-stage journeys).
4. Cost: ~2 navmesh path queries per agent per step (funnel waypoint +
   path deviation), like the native numpy env. Fine for validation scales;
   a shared-path cache is the obvious optimisation if needed.
5. Upstream questions to co-draft when the time comes: the 2-iteration exit
   removal lag (bookkeeping quirk or intended?), radius-aware routing, and
   the dt convention -- all now with concrete measurements attached.

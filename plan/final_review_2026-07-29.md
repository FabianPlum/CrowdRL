# Final deep review -- 2026-07-29 session (feat/jupedsim-integration)

End-of-session adversarial review of everything committed today
(`a263280..65f9dc9`, 24 commits: issue #7 metadata schema v1+v2, the
jupedsim 2.0 adapter port, contact physics, the shipped example model,
notebook 10, and `LockstepPolicyModel` byte-parity). Method: four
independent review passes (physics/math parity vs the training envs;
jupedsim-internals assumptions verified against the C++/Python source at
`49e3ddebd`; the ONNX metadata contract; observation math + test/notebook
honesty), each instructed to refute rather than confirm, plus empirical
probes (CrowdEnv-vs-`native_batch_step` step comparison, jupedsim runtime
behaviour, ONNX metadata dumps). Every finding below was then re-verified
against the source before inclusion. **This review changed no code**; the
jupedsim tree was confirmed untouched (no tracked file modified).

## Verdict

The central engineering claims of the day survived adversarial
re-verification:

- `native_batch_step` reproduces the real `CrowdEnv.step` **byte-identically**
  (independent probe: contact-heavy steps, speed-turn coupling on, including
  a masked-out finisher in CrowdEnv vs a removed row in the batch).
- The lockstep design assumptions were proven from jupedsim source:
  compute-then-apply on one pre-step snapshot, deterministic insertion-stable
  callback order (std::deque + stable remove_if + monotonic never-reused
  ids), and the 2-iteration exit-removal lag is structural pipeline order
  (strategical marks before the operational step; erasure at the start of
  the next iterate), not incidental.
- The shipped `example_model/policy_r0400.onnx` metadata is correct
  end-to-end (schema 2; w=0.8 / clamp 3.0 / contact 30000/500 match what
  that run trained under; provenance chain verified).
- Observation rotation math is identical in all four nav branches
  (route/navmesh x single/batch; probed), branch precedence is consistent,
  temporal-memory indexing agrees across all four implementations, and every
  notebook-10 headline output matches its committed cells.

Two real defects were found that should be fixed before further validation
work, one boundary-semantics bug in lockstep, a set of silent-fallback and
wording issues, several latent pre-existing CrowdRL issues, and two genuine
upstream jupedsim bugs to co-draft. Nothing shipped is factually wrong as a
*measurement*; the defects below change what future runs would do.

## The two findings that matter

### 1. [HIGH] `LearnedPolicyModel` integrates a heading that training does not have

`packages/crowdrl-jupedsim/src/crowdrl_jupedsim/model.py:474,531` -- the
interactive adapter feeds `current_heading=state.heading` into
`interpret_action` and persists `result.new_heading` back into the agent
state, so heading free-integrates independently of the torso.

Both training engines do something different: `crowd_env.py:334-341` and
the torch twin pass `torso_orientations` **twice** (as `current_headings`
AND `current_torsos`). In training, heading is *derived* -- re-anchored to
the previous torso every step -- and the commanded velocity direction can
never sit more than one per-step delta (~4.8 deg for the shipped artefact)
from the previous torso. In the interactive adapter, heading and torso
drift apart without bound, systematically changing the action->motion
mapping vs training. This channel appears nowhere in the
`plan/lockstep_parity_analysis.md` inventory (channels 1-7) nor in
model.py's deviation comment (which names only neighbour staleness).

`native_batch_step` (`lockstep.py:96-103`) passes torso twice, exactly like
training -- which is *why* byte-parity holds and why the byte-parity test
could never catch this: the deviant is the interactive model, and the test's
write-only `heading` array in the reference loop is correct (heading is not
persistent state in training).

Fix (one argument): `current_heading=state.torso_angle`; keep
`state.heading` as cosmetic output only. Then re-baseline the e2e scenarios
and notebook 10 (the interactive corner/bottleneck numbers 3/4 and 11/12
were measured WITH the deviation and may shift), and add the channel to the
parity-doc inventory.

### 2. [HIGH] `reexport_onnx.py` stamps guessed dynamics as "trained"

`scripts/reexport_onnx.py:69-74` sources the schema-v2 dynamics block from
`build_env_config(config_resolved.yaml)` -- but the YAML schema can express
only `desired_velocity_weight` (`train_mappo.py:189` is the sole dynamics
key parsed; `cfg_dict_from_env_config` emits only it). The other three
fields (`max_velocity_magnitude`, `contact_stiffness`, `contact_damping`)
always come from **present-day `CrowdEnvConfig` defaults at reexport time**,
yet get certified in the artefact as the run's trained physics.

Concretely wrong for older runs: before 540ccb0 (2026-05-26) the clamp was
a different formulation (effectively 4.0 m/s, and its YAML key
`max_speed_multiplier` is silently ignored by today's parser); before
9ab84a5 a YAML without `desired_velocity_weight` meant the then-default
0.8, which reexport today fills as 0.05. Re-exporting such a run produces a
schema-v2 artefact whose "trained dynamics" are fabricated -- exactly the
drift the v2 guarantee exists to prevent.

The shipped `policy_r0400.onnx` is correct **by date, not by mechanism**:
that run post-dates the refactor, its YAML pins `desired_velocity_weight:
0.8`, and the other three defaults have not changed since -- chain
verified. Fix: refuse (or require an explicit `--dynamics`) for fields the
YAML does not record, or at minimum print per-field provenance
(read-from-YAML vs assumed-default) and stamp it into provenance metadata.

## Medium findings

- **[MEDIUM] Lockstep exit predicate differs from jupedsim on the boundary.**
  `lockstep.py:347` freezes on shapely `contains` (strict interior);
  jupedsim marks exits boundary-INCLUSIVE (CGAL `bounded_side !=
  ON_UNBOUNDED_SIDE`, `libsimulator/src/Polygon.cpp:43-47`). Probe-confirmed
  with a position exactly on the exit edge: jupedsim marks it one iteration
  before shapely `contains` is true. Desync scenario: boundary landing at
  step k, pushed back out at k+1 -> jupedsim erases at k+2 an agent lockstep
  never froze -> the roster handler silently drops the row (no `exit_steps`
  entry) while the native reference keeps simulating it. Measure-zero under
  float dynamics but reachable; shapely `covers` matches jupedsim's
  predicate exactly.
- **[MEDIUM] The legacy dynamics fallback is completely silent.** A v1
  artefact self-configures obs/action without any warning, then runs
  `w=0.05, clamp=5.0` (`policy.py:313-351`) -- vs the trained `0.8/3.0` of
  every current best run; the w difference alone is a ~30x change in the
  velocity-response time constant. No warning anywhere names dynamics (the
  resolve_configs "cannot be verified" warning fires only for no-metadata
  artefacts and covers configs only). Related wording bug: docstrings call
  the 5.0 clamp "the crowdrl default", but the env default is 3.0 -- 5.0 is
  the pre-v2 *adapter* constant, kept deliberately so v1 deployments do not
  change silently. Fix: `warnings.warn` listing every field resolved from
  `_DYNAMICS_DEFAULTS` when the policy is metadata-capable; correct the
  docstrings.
- **[MEDIUM] Lockstep pass detection breaks on wholesale roster
  replacement.** Empirically confirmed: let the full roster exit, then
  `add_agent` + `iterate` -> `KeyError` at `lockstep.py:370`. Safe today
  only because jupedsim's agent container is a deque with order-stable
  removal, so any survivor fires first and triggers the pass. The
  limitation is undocumented. Clean fix discovered in review:
  `simulation.iteration_count()` is readable during callbacks (no
  `ThrowIfIterating` guard; increments only at the very end of `Iterate()`)
  -- keying passes on it eliminates the id-repeat heuristic and this
  failure class entirely (requires handing the model the Simulation handle
  after construction).
- **[MEDIUM] Notebook 10 closing note overstates.** Cell 16 claims that
  with retraining items 1-3 "the remaining engine differences are the
  measured sub-ULP float effects". Wrong on two counts: channels 6
  (neighbour one-step staleness in the per-agent contact damping) and 7
  (the 2-iteration exit lag leaking exited agents into neighbours'
  observations) remain step-level semantic differences of the interactive
  model after retraining; and the measured 3.7e-15 obs-builder reassociation
  is ~1 ulp, not sub-ULP. Reword.
- **[MEDIUM] e2e determinism margins are zero-slack with the caveat one hop
  away.** `tests/test_e2e_jupedsim_trained_policy.py` asserts exactly the
  observed counts (observed 3/4 asserts >=3; observed 11/12 asserts >=11)
  and says "deterministically loses one agent" without the machine-scope
  qualifier that lives in lockstep.py/the parity doc. Cross-machine ULP
  drift flipping one marginal agent into the absorbing state fails these
  tests with no hint at the point of failure. Fix: add the ONNX-Runtime
  machine-scope caveat to the module/class docstrings.

## Minor findings

- `config_io.validate_dynamics_dict` accepts NaN/inf/negatives/bools;
  a NaN defeats `resolve_dynamics` mismatch detection (`abs(x - nan) > tol`
  is False), so a disagreeing explicit value passes silently. Add
  `math.isfinite`.
- A dynamics payload that is valid JSON but not an object (e.g. `[]`)
  escapes as a raw `AttributeError` instead of the clean "unreadable"
  ValueError (`policy.py:108` -- the except net is too narrow).
- The `1e-12` dynamics mismatch tolerance is sub-ULP at contact-stiffness
  scale (`math.ulp(30000.0)=3.6e-12`): two adjacent doubles raise as a
  mismatch. Use a relative tolerance.
- `LearnedPolicyModel` has no sim-dt vs `ActionConfig.dt` guard (lockstep
  warns; the interactive model silently rescales the motion envelope at a
  non-trained dt).
- `_route_waypoint` has no fallback if `next_target` were ever None
  (unreachable through the real API -- it is a C++ value type -- but the
  failure would be an obscure shape error in `build_world_state`).
- e2e wall-clearance/containment checks subsample ~200 of up to 6000
  positions; brief excursions between samples are invisible (the pairwise
  spacing check, by contrast, runs every step).
- Deployment default body dims: chest 0.15 sits +2 sigma above the training
  spawner's mean (0.12 +/- 0.015); shoulder/mass/preferred-speed match. The
  byte tests hardcode the same 0.15, so training-typical bodies are never
  exercised.
- Exit-goal convention trap: the byte-parity reference builds its goal from
  shapely's **area centroid** while jupedsim's `final_target` is the
  **vertex average** (`Polygon.cpp:49-56`). These coincide for rectangles
  (all current tests) and diverge for irregular exits -- parity would break
  with a different goal, not a protocol bug. Document or compute the goal
  the jupedsim way.
- `other_agents_in_range` excludes "self" by exact **position** equality,
  not id (`EnvironmentQuery.hpp:41-48`): two agents spawned at the identical
  point are mutually invisible (breaks lockstep's full-roster guarantee and
  the interactive model's KNN). Degenerate spawns only;
  `check_model_constraint` is the sanctioned place to reject them.

## Test-honesty hardening (assertions weaker than their docstrings)

- `assert_byte_identical` compares `min(len, len)` steps and never asserts
  the lengths match; `TestCornerSegment` asserts nothing else, so a jps side
  stopping early would pass on the compared prefix. The corridor test pins
  exit equality but not completion (1 of 3 exiting passes). Add length and
  full-completion assertions.
- The e2e straggler skip (`len(traj) == steps`) is exact (verified), but
  the corner/aperture route tests pass vacuously on an all-freeze regression
  when run in isolation -- assert at least one non-skipped trajectory.
- The e2e comment "(a policy property, not an adapter defect --
  byte-identical across engines)" rests on an uncommitted absorbing-state
  probe; the committed native/lockstep corner runs exit 4/4, so nothing in
  the repo demonstrates it. Commit the captured-state probe as a test or
  reword the comment.
- `example_model/README.md` says "bit-exact" but the committed
  `--verify-against` gate enforces max|diff| <= 1e-5 over 256 random inputs
  (the actual measured diff was 0.0). Tighten the gate or soften the word.
- Byte-parity test and notebook reach into `model._frozen`; add a public
  frozen-ids accessor.

## Pre-existing CrowdRL issues surfaced (not from today's commits)

- `crowd_env.py:243`: `world.preferred_speeds` is populated only inside the
  `use_temporal_memory` branch -- temporal-OFF configs feed the constant
  1.34 into the ego obs while the torch training twin always feeds the
  sampled per-agent speeds. A real train/eval twin inconsistency (moot for
  the shipped temporal-on artefact).
- There is no full-step numpy-vs-torch trajectory equivalence test; the
  twins are equivalence-tested at component level (atol 1e-4) and the
  navmesh signal is asserted equal only at spawn (they intentionally differ
  off-route: numpy re-funnels, torch follows the stored path cursor).
  Worth adding before IAS-7 validation leans on the numpy env as "the"
  reference.
- `observation.py:626-659`: the temporal block's offset advance sits inside
  the populated-guard; a world with temporal enabled but unpopulated memory
  AND populated neighbor blocks would misalign the A+ block into the
  temporal slots. No current producer creates that combination.
- The batch obs builder sanitizes with `nan_to_num`; the single builder
  (used by the interactive adapter) does not.
- Lockstep never populates neighbor-memory state, so its byte-exactness
  claim is scoped to configs without A+/A++ neighbor features -- currently
  undeclared in lockstep.py (model.py at least carries a scope note).

## Upstream jupedsim -- to co-draft together (never post directly)

New, found in this review (both verified against source and probed):

1. **Bug**: the sim-side `Agent.next_target` property
   (`python_modules/jupedsim/jupedsim/agent.py:173`) calls
   `self.__resolve().next_destination`, but the binding exposes
   `next_target` (`python_bindings_jupedsim/agent.cpp:36-37`) ->
   `AttributeError` for anyone reading `sim.agent(id).next_target`. (Our
   adapters use the transient callback path, which works.)
2. **Bug**: `environment_query.py` declares
   `line_segments_in_grid_cell_distance` and `intersects_any` with no
   corresponding C++ binding registered (only 4 methods are bound). Unused
   by us.
3. **Refinement of the corner root-cause**: the router DOES inset funnel
   waypoints -- by a **fixed 0.2 m** along the portal edge
   (`RoutingEngine.cpp:330-331`; our measured corner waypoint sat at
   exactly 0.2000 m). The problem is that 0.2 is not radius-aware and sits
   inside our 0.225 m body radius. The upstream ask is therefore
   "radius-aware/configurable waypoint inset", not "add an inset".
   **Followed up 2026-07-30**: configurability is now ruled out from source --
   agent size is private to the operational model, the router is passed only a
   position, and the A* explicitly assumes point-size agents. Full evidence
   and the co-draft design sketch live in `plan/lockstep_parity_analysis.md`,
   section "Why the router inset cannot be configured".
4. Existing items, now with mechanism attached: the 2-iteration exit
   removal lag is structural pipeline order (`Stage.cpp:116-123` marks in
   the strategical phase; `AgentRemovalSystem` erases at the start of the
   NEXT iterate) -- is the lag intended?; the dt convention question.

## Re-verified sound (what we now trust, with evidence)

- Step-order/operand parity `native_batch_step` == `CrowdEnv.step`,
  empirically byte-identical on contact-heavy steps, including
  masked-finisher vs removed-row equivalence (KNN/raycast/collision/wall
  code all skip inactive agents identically).
- Temporal-memory indexing consistent across CrowdEnv, lockstep, the
  interactive adapter, and the test reference (write at pre-step count slot,
  read at post-step count).
- Rotation math identical in all four nav branches (probed to ~1e-16);
  route branch strictly wins over navmesh in both builders; no producer
  sets both sources.
- `enforce_wall_boundaries` clearance = per-agent max(shoulder, chest) in
  both numpy and torch; the e2e 0.225 threshold asserts exactly the
  enforced quantity with a rounding tolerance.
- dt = 0.01 consistently across CrowdEnvConfig, ActionConfig, ObsConfig
  temporal, torch EnvConfig, the tests, and the artefact metadata.
- resolve_configs/resolve_dynamics decision tables behave as specified in
  every quadrant; the reader hard-fails on partial/corrupt/unknown/
  contradictory metadata; all three producers (train_mappo, reexport,
  notebook 06) pass complete four-field dynamics from the live env config,
  and the torch training env consumes exactly those fields.
- Compute-then-apply verified in `OperationalDecisionSystem.hpp:38-48`
  (callbacks read the pre-step generation; swap after the loop); env-query
  returns copies of pre-step agents; callback order is insertion-stable and
  ids are never reused -- lockstep's sorted-by-id roster equals native
  array order (float summation order preserved).
- `CrowdRLAgentState` fully satisfies jupedsim's custom-state contract
  (position-only; jupedsim writes nothing back; same-instance returns
  rejected -- `dataclasses.replace` always complies). Bypassing the router
  corrupts no stage bookkeeping (position-driven) and no writers.
- Notebook 10's committed outputs match every headline claim (dynamics
  dict, 3/4, 11/12, byte-identical over 1082 steps, identical exit steps);
  the file is 100% ASCII; the w=0.8 -> tau ~6 ms arithmetic is correct.
- Both repos' integrity: no tracked jupedsim file modified; CrowdRL tree
  clean apart from unrelated untracked files.

## Next steps (ordered)

1. Fix the interactive heading anchoring (`current_heading=state.torso_angle`),
   document the channel, re-baseline e2e + notebook 10.
2. Harden `reexport_onnx.py` dynamics provenance (refuse or `--dynamics`
   for YAML-unrecorded fields; stamp per-field provenance).
3. Lockstep correctness batch: `covers` instead of `contains` for exit
   freeze; pass detection via `iteration_count()`; declare the
   fixed-roster + no-A+/A++ scope in the docstring.
4. Warn on the legacy dynamics fallback for metadata-capable policies;
   fix the "crowdrl default" docstrings; add the isfinite/JSON-shape/
   relative-tolerance fixes in config_io/policy.
5. Test hardening: byte-parity length+completion assertions; e2e
   machine-scope caveats + at-least-one-examined assertions; commit the
   absorbing-state probe (or reword the comment); public frozen accessor;
   align README wording with the verify gate.
6. Reword notebook 10 cell 16 (channels 6/7 persist for the interactive
   model; "~1 ulp", not "sub-ULP").
7. Add a dt guard to `LearnedPolicyModel` (mirror lockstep's warning).
8. YAML lossy-gap guard in `cfg_dict_from_env_config` (long-standing,
   confirmed still absent -- docstring-only today).
9. Co-draft the upstream items above with the jupedsim dev team.
10. Retraining roadmap (unchanged, now better justified): router-like
    waypoint randomization (train against fixed-0.2 m-style insets),
    no_p_dev ablation, absorbing-state robustness.
11. Pre-existing CrowdRL cleanups when convenient: preferred_speeds for
    temporal-off configs; a full-step numpy-vs-torch trajectory test;
    single-builder NaN policy.

Standing items unchanged: dt-validation upstream issue rests (user call);
`plan/CrowdRL_Project_Plan_v5.docx` staleness question;
`memory_experiments` has 1 unpushed local commit.

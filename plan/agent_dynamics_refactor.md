# Agent Dynamics Refactor: From Kinematic Targets to Bounded Accelerations

> **Implementation status (verified 2026-08-03) -- this doc is half shipped, half design.**
>
> - **Layer 1 (parameter recalibration): SHIPPED.** The recalibrated caps are now the
>   dataclass defaults -- heading `0.020` rad/step (115 deg/s), torso `0.010` (57 deg/s),
>   head `0.030` (172 deg/s), `desired_velocity_weight` 0.05 (renamed from
>   `velocity_damping`), smoothness weights rescaled. `crowdrl_core/action.py` and
>   `crowdrl_env/{crowd_env,reward}.py` cite this file by name at those defaults. See the
>   2026-05-26 entry in `plan/CrowdRL_Project_Plan_v10.md`.
> - **Layer 2 (second-order action semantics -- accelerations, explicit yaw-rate state,
>   3D non-holonomic action space, `dynamics_mode` gate): DESIGNED, NEVER BUILT.**
>   `dynamics_mode`, `second_order`, `yaw_rate` and `longitudinal_speed` appear nowhere in
>   `packages/`, `configs/` or `train_mappo.py`. **Section 4 below is the only record of
>   that design.**
> - **What shipped instead of Layer 2:** the *speed-turn coupling* clamp
>   (`ActionConfig.speed_turn_coupling`, heading/torso deltas additionally limited to
>   `min(turn_pivot_rate, turn_lat_accel / v) * dt`). It expresses Layer 2's intent --
>   you must slow down to turn sharply -- as a Layer-1 constraint, without the state-space
>   change. The shipped r0125 run trains with it on at a 240 deg/s pivot rate.
>
> Read Sections 1-3 as history (the diagnosis was correct and led to the shipped fix), and
> Section 4 as a live, unbuilt proposal.

**Status:** Draft for internal discussion
**Author:** Plum (with assistance)
**Scope:** `crowdrl-core/action.py`, `crowdrl-torch/{action,step,types}.py`, `crowdrl-env/crowd_env.py`, observation builders, reward weights
**Trigger:** Goal-completion rates are now solid across all geometry tiers since memory and progress-awareness additions, but motion is visibly erratic. Reward smoothness terms are not the bottleneck; the action space and its physical realisation are.

---

## 0. Executive Summary

The current action interpreter maps a 4D continuous policy output directly to kinematic targets (desired speed, heading change, torso angle change, head angle change). The per-step rate caps were chosen for a much larger timestep than the current `dt = 0.01 s`, so at simulation rate they correspond to angular velocities one to two orders of magnitude beyond human capability. The first-order velocity filter is also misnamed and run with a time constant of roughly 12 ms, which is essentially "no filter" at 100 Hz. The Tier 2 smoothness reward weights are too small to influence learning against a `+10` goal bonus.

This document proposes a two-layer fix:

**Layer 1** (parameter recalibration): a single-commit retune of rate caps, velocity filter constant, and smoothness weights. Same action semantics, same code paths. Smooths motion without architectural change. Cheap to validate.

**Layer 2** (action space redesign): reinterpret the 4D policy output as accelerations rather than targets, add angular velocities to agent state and observation, and introduce a speed-coupled yaw envelope. This is the structural change that produces human-shaped trajectories as a property of the dynamics rather than a learned objective.

Layer 1 is a sanity check. Layer 2 is the version that should appear in the paper.

---

## 1. Diagnostic: Current State Against Human Envelope

### 1.1 Per-axis rate caps at `dt = 0.01 s`

From `packages/crowdrl-torch/src/crowdrl_torch/types.py` (EnvConfig defaults) and `packages/crowdrl-core/src/crowdrl_core/action.py` (ActionConfig defaults):

| Axis | Cap per step | Implied rate | Human walking | Human standing |
|---|---|---|---|---|
| Heading change | π/12 = 15° | 1500°/s | ~120°/s | ~360°/s |
| Torso change | π/12 = 15° | 1500°/s | ~90°/s | ~360°/s |
| Head change | π/3 = 60° | 6000°/s | ~180°/s sustained | saccadic peaks higher |
| Speed | [0, 1.5] m/s direct set | up to 150 m/s² implied | ~2 m/s² normal, ~3 m/s² emergency |

`docs/environment_mechanics.md` lines 824 to 826 print the 1500°/s and 6000°/s figures verbatim in the parameter table. The numbers were chosen for a coarser timestep and never revisited after `dt` was reduced.

### 1.2 Velocity filter naming and direction

In both `crowdrl_env/crowd_env.py` (line 259) and `crowdrl_torch/step.py` (line 60), the velocity update is:

```python
new_v = velocity_damping * desired_v + (1 - velocity_damping) * old_v
```

With `velocity_damping = 0.8`, the new velocity is 80 percent the new desired velocity and 20 percent the previous velocity. This is a first-order low-pass with time constant approximately `dt / velocity_damping = 12.5 ms`. The Helbing-Molnár social force model uses a relaxation time τ around 500 ms; this filter is roughly 40 times faster.

The name is also backwards relative to the formula. Higher `velocity_damping` means *less* smoothing, not more. The doc at `docs/agent_pipeline.md` line 515 has this inverted as a recommendation:

> "Increase velocity damping. Raising velocity_damping from 0.8 toward 0.9 or 0.95 increases inertia."

That direction reduces inertia, not increases it. The variable should be renamed (`desired_velocity_weight` or `velocity_blend`) or the formula inverted to match the name. Either renaming is a one-line semantic clarification and will save downstream confusion.

### 1.3 Smoothness reward weights versus task reward

From `crowdrl_env/reward.py` lines 75 to 83 and `EnvConfig` defaults:

| Term | Weight | Notes |
|---|---|---|
| `goal_bonus` | +10.0 | Sparse, terminal |
| `collision_penalty` | -1.0 | Per-step on contact |
| `progress_weight` | +1.0 | Per-step on goal distance reduction |
| `jerk_penalty_weight` | -1e-6 | Comment notes jerk scales 1/dt², so weight was kept small |
| `angular_accel_penalty_weight` | -1e-4 | |
| `speed_deviation_weight` | -1e-3 | |
| `action_rate_weight` | 0.0 | Disabled |

The comment next to `jerk_penalty_weight` ("jerk scales as 1/dt², kept small") shows that the weight was reduced specifically because magnitudes are huge. This is the wrong way round. If jerk magnitudes are enormous, that is the symptom of the action space being unconstrained, not a reason to ignore them. Either the action space prevents large jerk (Layer 2) or the penalty is large enough to dominate (still not ideal because the policy will then trade collisions for smoothness, but at least it sees the signal).

### 1.4 The heading/torso passing pattern

In both `crowdrl_torch/step.py` lines 49 to 55 and `crowdrl_env/crowd_env.py` lines 248 to 254:

```python
batch_result = interpret_actions_batch(
    actions,
    state.torso_orientations,   # passed as current_headings
    state.torso_orientations,   # passed as current_torsos
    state.head_orientations,
    cfg.action,
)
```

`TorchWorldState` has no `headings` field. `current_headings` is therefore always reset to the torso angle each step. The `new_headings` returned by `interpret_action` is used to compute `desired_velocity` and is then discarded; it is not written back to state.

Effective semantics: `action[1]` is "instantaneous velocity direction offset from torso this step, up to ±15°", not heading in any persistent sense. The policy can produce a velocity vector misaligned from body orientation by up to 15 degrees on any step, with no continuity constraint between steps. This is a contributor to the visible jitter and should be decided explicitly: either heading is body-aligned and `action[1]` should not exist as a separate axis, or heading is a real state variable and needs to be stored and integrated like the others.

The Layer 2 refactor resolves this by collapsing the ambiguity: there is one body orientation (torso), one heading equal to the direction of the velocity vector (derived, not commanded), and the angular controls act on torso and head only.

---

## 2. Biomechanical Reference Values

All values are for free-flow pedestrian locomotion in an evacuation or transit context. Sources are mainstream pedestrian-dynamics literature and biomechanics textbooks.

### 2.1 Linear motion

| Quantity | Value | Source / rationale |
|---|---|---|
| Preferred walking speed | 1.34 m/s, σ ≈ 0.37 | Bohannon 1997 (meta-analysis) |
| Maximum comfortable walking | ~2.0 m/s | Bohannon 1997, Knoblauch 1996 |
| Forward acceleration (normal) | 0.5 to 1.0 m/s² | Various gait studies |
| Forward acceleration (emergency) | up to ~3 m/s² | Helbing social force τ ≈ 0.5 s gives ~3 m/s² effective |
| Forward deceleration | 3 to 5 m/s² | Higher than acceleration due to mechanical braking via stance leg |

### 2.2 Yaw (heading change of body)

| Quantity | Value | Notes |
|---|---|---|
| Yaw rate, walking, comfortable | 30 to 60 °/s | Hicheur et al. 2007 |
| Yaw rate, walking, aggressive | up to ~120 °/s | Same reference; observed in tight cornering |
| Yaw rate, stationary | up to ~360 °/s | Pivot turn in place |
| Coupling | Decreasing in forward speed | Hicheur et al. 2007 quantifies a roughly inverse relationship between forward velocity and admissible yaw rate |

The speed-yaw coupling is the single most important biomechanical constraint absent from the current code. Humans cannot turn sharply at full pace. Encoding this constraint is what produces the canonical "decelerate, turn, accelerate" trajectory shape that distinguishes pedestrian motion from particles in a social force field.

### 2.3 Torso rotation independent of feet

Torso can twist relative to hips by approximately ±30° during normal gait. Angular velocity of torso independent of feet is bounded by hip joint constraints, typically 60 to 90 °/s in sustained motion. For shoulder-pass behaviour in bottlenecks (Helbing's "zipper" phenomenon), pedestrians rotate torso by 10 to 30 degrees in roughly 0.3 to 0.5 seconds, giving angular velocities in the 30 to 60 °/s range.

### 2.4 Head rotation relative to torso

Head can rotate ±90° relative to torso (anatomical limit, already encoded). Sustained head angular velocity in active scanning is around 120 to 180 °/s. Saccadic gaze shifts use eye motion primarily, and only slower head motion follows; for our purposes 180 °/s is a reasonable upper bound on head-only angular velocity.

### 2.5 Linear acceleration coupling

Pedestrian linear acceleration is also coupled to current speed, but more weakly than yaw rate is. The constraint is mostly mechanical: from standstill, you can spend the full leg-extension cycle on forward thrust; near max speed, leg recovery time constrains how much additional thrust you can deliver. A simple linear envelope is adequate for our purposes.

---

## 3. Layer 1: Parameter Recalibration

### 3.1 Goal

Bring the per-step rate caps into the human envelope at `dt = 0.01 s`, engage the velocity filter meaningfully, and raise smoothness reward weights to the order where they can compete with the goal bonus. No semantics change. No new state. Single-commit change.

This is a baseline calibration, not the final design. The purpose is to confirm that (a) the visible jitter is genuinely caused by the action space and not by the policy itself, and (b) Layer 2 builds on top of a working baseline rather than fighting a broken one.

### 3.2 Proposed parameter values

In `EnvConfig` (`crowdrl_torch/types.py`) and `ActionConfig` (`crowdrl_core/action.py`) and the matching CPU config:

```python
# Action rate caps at dt = 0.01 s, biomechanically grounded
max_speed:           1.5       # unchanged; preferred ~1.34
max_heading_change:  0.020     # 1.15° per step → ~115°/s, walking yaw envelope
max_torso_change:    0.010     # 0.57° per step → ~57°/s, slower than heading (hips constrain)
max_head_change:     0.030     # 1.72° per step → ~172°/s, head moves fastest
head_limit:          π/2       # unchanged; anatomical clamp

# Velocity filter
velocity_damping:    0.05      # τ ≈ 200 ms; rename considered (see 3.3)
# OR for Helbing-like:
# velocity_damping:  0.02      # τ ≈ 500 ms
```

Derivation: the per-step cap times the sampling rate gives the implied angular velocity. The choices above target the middle of the walking envelope, leaving room for the policy to learn aggressive behaviour up to the upper bound without exceeding human kinematics.

The relative ordering matters: head faster than heading, torso slower than heading. This biases the policy toward "head scans, body commits", which is the observed pattern in real pedestrians and is currently impossible to learn because all three move at the same speed.

### 3.3 Optional rename of `velocity_damping`

The name is currently misleading. Two equivalent fixes:

**Option A** (minimal change): rename to `desired_velocity_weight` everywhere, keeping the formula.

**Option B** (matches the name): change the formula to
```python
new_v = (1 - velocity_damping) * desired_v + velocity_damping * old_v
```
and flip the default value accordingly (`velocity_damping = 0.95` then means heavy damping, intuitive).

Pick one and apply consistently across `crowdrl_env/crowd_env.py`, `crowdrl_torch/step.py`, both action.py files if they reference it, and `docs/{agent_pipeline,environment_mechanics}.md`. Update the agent_pipeline.md recommendation block at line 515 to reflect the correct direction.

### 3.4 Reward weight rebalance

Current Tier 2 weights are too small by 3 to 4 orders of magnitude relative to task reward. The comment in `reward.py` line 76 ("kept small because jerk scales 1/dt²") inverted the correct response. Proposed weights:

```python
jerk_penalty_weight:           -1e-4   # was -1e-6; up 100x
angular_accel_penalty_weight:  -1e-2   # was -1e-4; up 100x
speed_deviation_weight:        -1e-1   # was -1e-3; up 100x
action_rate_weight:            -1e-2   # was 0.0; enable
```

These are starting points for a sweep, not final values. A short sweep across [0.5x, 1x, 2x, 5x] of these should identify the right operating point. The criterion is that smoothness penalties produce visible behavioural change without dominating goal completion.

Note: after Layer 2, smoothness penalties will likely become much less important (perhaps removable) because the action space itself prevents the bad behaviour. Layer 1's reward rebalance is therefore expected to be partially or fully reversed in Layer 2.

### 3.5 Files to change

| File | Lines | Change |
|---|---|---|
| `packages/crowdrl-core/src/crowdrl_core/action.py` | 30-40 | Update `ActionConfig` defaults |
| `packages/crowdrl-env/src/crowdrl_env/crowd_env.py` | 85-95 | Update `CrowdEnvConfig` to match |
| `packages/crowdrl-env/src/crowdrl_env/reward.py` | 67-83 | Update reward weights |
| `packages/crowdrl-torch/src/crowdrl_torch/types.py` | 100-135 | Update `EnvConfig` defaults |
| `docs/agent_pipeline.md` | 158-161, 515 | Update parameter table; fix damping direction note |
| `docs/environment_mechanics.md` | 813-826 | Update parameter table and implied-rate comments |

If renaming `velocity_damping`, add the corresponding changes in `step.py` lines 60-63 and `crowd_env.py` lines 258-261.

### 3.6 Tests to update

- `packages/crowdrl-core/tests/test_action.py`: update any tests that hardcode the old per-step caps. Add a test that asserts the implied angular velocities at `dt = 0.01 s` are below 200 °/s for each axis.
- `packages/crowdrl-torch/tests/test_equivalence.py`: should pass unchanged if both CPU and GPU configs are updated symmetrically. Run this test specifically to confirm.

### 3.7 Validation plan

Train one policy with Layer 1 parameters using the existing curriculum and memory/progress-awareness architecture. Compare against the current policy on three observables:

1. **Velocity autocorrelation function** computed over agent trajectories. Layer 1 should produce smoother decay with longer correlation timescales, closer to PeTrack measurements on IAS-7 experimental data.
2. **Empirical jerk distribution** across all active agents over an evaluation episode. Layer 1 should reduce the right tail substantially.
3. **Goal completion rate** at the hardest tier. Layer 1 should not degrade completion significantly. If it does, the rate caps are too tight, or the policy needs more training steps to adapt; both are diagnostically informative.

If all three improve and completion is preserved, Layer 1 is a clean baseline. If only smoothness improves and completion degrades, the rate caps are too restrictive given the policy's current state; either relax slightly or accept and proceed to Layer 2 where the policy gets richer state to plan with.

### 3.8 Expected outcomes

Visible jitter substantially reduced. Trajectories look like pedestrians, not particles. Some loss of agility in tight spaces, but goal completion preserved within a few percent. This is the baseline for the structural redesign.

---

## 4. Layer 2: Second-Order Action Semantics

### 4.1 Motivation

The current action interpreter treats the policy output as kinematic targets. Each step, the agent declares what its speed and orientation will be, and the simulator makes it so (modulo a too-weak filter and contact forces). Under this model the policy has authority over kinematic state directly; there is no physical body in the loop.

The "driving a body" framing in conversations with collaborators and reviewers describes a different model. The body has mass, inertia, angular momentum, and biomechanical limits. The policy operates the controls (accelerations, torques) that influence body state. The body integrates over those controls subject to its own constraints.

This is the right model for three reasons:

1. **Discoverability of behaviour.** With a constrained body, behaviours like "slow before a sharp turn" or "torso-rotate to fit through a bottleneck" emerge from the policy maximising reward subject to physical constraints. Under direct kinematic control these behaviours have to be learned despite being not necessary.

2. **Defensibility.** "Action space corresponds to longitudinal and angular accelerations on a body with kinematic envelope from Hicheur 2007" is a one-sentence justification a reviewer accepts. "We have rate-limited actions plus a jerk penalty with weight 1e-4" is not.

3. **Sim-to-real and JuPedSim deployment.** JuPedSim's existing operational models (social force, generalized centrifugal force) are all second-order in velocity. A CrowdRL policy that outputs accelerations slots in alongside them naturally. A policy that sets velocities directly does not match the abstraction.

### 4.2 Action space redefinition

The policy continues to output `a ∈ [-1, 1]^4` from a Tanh-Gaussian distribution. The interpretation changes:

| Index | Old meaning | New meaning |
|---|---|---|
| `a[0]` | Desired speed (mapped to [0, max_speed]) | Longitudinal acceleration along torso direction (mapped to [-max_long_decel, +max_long_accel]) |
| `a[1]` | Heading change this step | Yaw acceleration of torso (subject to speed-coupled envelope) |
| `a[2]` | Torso angle change | Removed. Torso angle is the agent's body orientation; "yaw" is its rate of change |
| `a[3]` | Head angle change relative to torso | Head angular acceleration relative to torso (clamped to anatomical range on integration) |

The new action space is 3D, not 4D. Action[2] disappears because the previous action[1]/action[2] split was the source of the heading/torso confusion. There is one body orientation. The direction of motion is derived from the velocity vector, which integrates over longitudinal acceleration along the body axis plus contact forces.

If a strafe degree of freedom is desired (sidestepping), it can be reintroduced as a separate lateral acceleration component, giving 4D again. This is worth a deliberate design decision rather than carrying through the existing ambiguity. For pedestrian dynamics in JuPedSim contexts, holonomic 2D motion (forward and sideways acceleration) is appropriate; for an embodied model closer to legged locomotion, non-holonomic (forward only) is more faithful. The bottleneck pass behaviour you care about needs torso rotation, not lateral acceleration, so non-holonomic is the cleaner default. Recommendation: start non-holonomic, 3D action space.

### 4.3 State extension

Add to `WorldState` (and `TorchWorldState`):

```python
longitudinal_speed:   float    # current scalar forward speed along torso direction
yaw_rate:             float    # angular velocity of torso
head_rate:            float    # angular velocity of head relative to torso
```

The 2D `velocities` vector becomes a derived quantity: `velocities = longitudinal_speed * [cos(torso_orientation), sin(torso_orientation)]` plus any residual from contact forces. Contact forces continue to add to the 2D velocity vector directly as before; this is the only mechanism by which motion can be non-aligned with body orientation.

Internal state extension: 3 new scalars per agent. Storage cost is negligible.

### 4.4 Integration step

In pseudocode for the new dynamics step (replaces lines 49 to 64 of `crowdrl_torch/step.py` and analogous block in `crowd_env.py`):

```python
# 1. Policy actions -> raw accelerations
long_accel = a[0] * MAX_LONG_ACCEL          # ~2.0 m/s²
yaw_accel  = a[1] * MAX_YAW_ACCEL           # ~5.0 rad/s²
head_accel = a[2] * MAX_HEAD_ACCEL          # ~10.0 rad/s²

# 2. Integrate angular velocities, clamp to biomechanical envelope
new_yaw_rate = state.yaw_rate + yaw_accel * dt
new_yaw_rate = clamp(new_yaw_rate, -yaw_rate_max(state.longitudinal_speed),
                                   +yaw_rate_max(state.longitudinal_speed))

new_head_rate = state.head_rate + head_accel * dt
new_head_rate = clamp(new_head_rate, -HEAD_RATE_MAX, +HEAD_RATE_MAX)

# 3. Integrate scalar speed, clamp to physical range
new_speed = state.longitudinal_speed + long_accel * dt
new_speed = clamp(new_speed, 0.0, V_MAX)

# 4. Integrate angles
new_torso_orientation = state.torso_orientation + new_yaw_rate * dt
new_head_rel_torso    = state.head_rel_torso + new_head_rate * dt
new_head_rel_torso    = clamp(new_head_rel_torso, -HEAD_LIMIT, +HEAD_LIMIT)
# Reset head_rate to zero if clamped to anatomical limit (joint stop)
new_head_rate = where(at_clamp_boundary, 0.0, new_head_rate)

# 5. Derive desired velocity vector (body-aligned)
desired_velocity = new_speed * [cos(new_torso_orientation), sin(new_torso_orientation)]

# 6. Velocity filter as before (with rebalanced or eliminated damping)
new_velocity = filter_weight * desired_velocity + (1 - filter_weight) * state.velocity

# 7. Contact forces and wall enforcement unchanged
# 8. Position update unchanged
```

The clamp on `new_head_rate` to zero at the anatomical boundary is a small but important detail: it prevents the integrator from accumulating phantom angular velocity into the joint stop, which would otherwise produce a spring-back artifact when the head moves away from the limit.

### 4.5 Speed-yaw coupling

The function `yaw_rate_max(speed)` is the centrepiece. Concrete form:

```python
def yaw_rate_max(speed, v_max=2.0,
                 omega_stationary=2*pi,    # 360°/s when stationary
                 omega_walking=pi/2):       # 90°/s at v_max
    alpha = clip(speed / v_max, 0.0, 1.0)
    return omega_stationary * (1 - alpha) + omega_walking * alpha
```

This is a linear envelope between stationary (fast spin) and full-speed (slow turn). Other shapes (exponential, sigmoidal) are defensible; the qualitative property of monotone decrease in admissible yaw rate as forward speed grows is what matters, and the literature supports a roughly linear or weakly concave decrease across the comfortable walking range.

The same idea applies in principle to longitudinal acceleration, but the effect is much weaker for pedestrians. Recommendation: do not couple longitudinal acceleration to current speed in the first version, keep `MAX_LONG_ACCEL` constant. Reconsider if validation against IAS-7 data shows the coupling is needed.

### 4.6 Observation space changes

Ego state expands from 7 to 10 dimensions to expose the new angular velocity state to the actor:

| Index | Quantity | Was |
|---|---|---|
| 0-1 | Goal direction (egocentric) | unchanged |
| 2-3 | Velocity vector (egocentric) | unchanged |
| 4 | Longitudinal speed | NEW (or replace 2-3 if scalar suffices) |
| 5 | Yaw rate | NEW |
| 6 | Head rate (relative to torso) | NEW |
| 7 | Heading (relative to goal direction) | unchanged at old index 4 |
| 8 | Torso orientation | unchanged at old index 5 |
| 9 | Head angle relative to torso | unchanged at old index 6 |

Without these the policy is blind to its own momentum. The critic, in the centralised input, can see the same plus other agents' yaw rates if desired; this is a future tunable.

Observation dim goes from 79 (or 98 with optional channels) to 82 (or 101). Update `EnvConfig.obs_dim` accordingly.

### 4.7 Files to change

| File | Approximate scope |
|---|---|
| `packages/crowdrl-core/src/crowdrl_core/action.py` | Rewrite `interpret_action` and `interpret_actions_batch`. Add `IntegratedActionState` (or extend `ActionResult`) carrying new angular velocities. |
| `packages/crowdrl-core/src/crowdrl_core/world_state.py` | Add `longitudinal_speeds`, `yaw_rates`, `head_rates` fields. |
| `packages/crowdrl-core/src/crowdrl_core/observation.py` | Extend ego state from 7 to 10 dims. Update obs_dim calculation. |
| `packages/crowdrl-env/src/crowdrl_env/crowd_env.py` | Update step to use new action interpreter signature. Pass and update angular velocity state. |
| `packages/crowdrl-torch/src/crowdrl_torch/action.py` | Mirror CPU changes. |
| `packages/crowdrl-torch/src/crowdrl_torch/types.py` | Add `yaw_rates`, `head_rates`, `longitudinal_speeds` to `TorchWorldState`. Update `EnvConfig`. |
| `packages/crowdrl-torch/src/crowdrl_torch/step.py` | Update step to call new action interpreter, integrate angular velocities, apply speed-yaw coupling. |
| `packages/crowdrl-torch/src/crowdrl_torch/observation.py` | Mirror CPU obs builder changes. |
| `packages/crowdrl-train/src/crowdrl_train/mappo.py` | Verify obs_dim is read from config rather than hardcoded. |
| Tests across all packages | Update or add. |

### 4.8 Backward compatibility / migration

Gate the new dynamics behind a config flag during transition:

```python
# EnvConfig
dynamics_mode: str = "kinematic"   # "kinematic" (Layer 1) or "second_order" (Layer 2)
```

Both action interpreters live side by side, dispatched on `cfg.dynamics_mode`. Layer 1 trained policies remain runnable. The flag is removed once Layer 2 is validated. This avoids a hard cut and lets you compare policies trained under each regime in the same evaluation harness.

### 4.9 CPU / GPU parity

Both `crowdrl_core/action.py` and `crowdrl_torch/action.py` implement the same logic. The existing equivalence test in `packages/crowdrl-torch/tests/test_equivalence.py` should be extended to cover the new dynamics path. The two implementations must produce identical observations and identical agent states given identical actions, to within floating point tolerance. This is non-negotiable; it is the train/deploy parity invariant.

### 4.10 ONNX export and JuPedSim adapter

The ONNX export wraps the actor only; the actor takes observations and produces actions. Layer 2 changes both the observation shape (79 to 82) and the action interpretation (the meaning of `a[0..3]`, and the dimensionality if dropping to 3D). The ONNX model itself does not change structurally; only its input shape changes.

The JuPedSim adapter (when started) needs to know to:

1. Build observations using the new ego state layout.
2. Maintain `longitudinal_speed`, `yaw_rate`, `head_rate` as private state alongside the existing torso/head angles, since JuPedSim agents do not natively carry these.
3. Apply the new action interpreter to update these private state variables, then expose the resulting `desired_velocity` to JuPedSim's simulation loop as before.

The contract between training and deployment is preserved as long as the `WorldState` population in the adapter sets all new fields correctly. Document this explicitly when `crowdrl-jupedsim` work starts.

---

## 5. Training and Curriculum Considerations

### 5.1 Exploration noise

PPO/MAPPO learn a Gaussian over action space. With the new action semantics being accelerations, the policy's natural exploration scale is now an acceleration, which has different units and characteristic scale from the old "set target" interpretation. The default `log_std_init` may need retuning. Suggested starting point: `log_std_init = -1.0` (σ ≈ 0.37 in normalised action units), monitor entropy during early training.

### 5.2 Curriculum impact

Layer 2 changes what the policy has to learn. Under Layer 1 a policy has to discover that smooth motion is rewarded; under Layer 2 smooth motion is forced and the policy has to discover when to apply which acceleration.

Expectation: Layer 2 policies will look worse early in training (the policy has to learn to plan over momentum) but will reach better terminal behaviour. The curriculum may need an additional Tier 0.5 stage with extra-low density and large goals to bootstrap the basic "apply forward acceleration, accumulate speed, integrate to goal" loop before density and obstacles enter the picture.

### 5.3 Warm-starting

Layer 1 policies are not architecturally transferable to Layer 2 because the action distribution mean is interpreted differently. Cold-start Layer 2 from scratch. This is a small cost relative to total training budget on JURECA / JUSUF and avoids a hidden bias from the Layer 1 reward landscape.

### 5.4 Reward weights under Layer 2

Most smoothness penalties become redundant once the action space prevents large jerk. Suggested defaults for Layer 2:

```python
jerk_penalty_weight:           0.0     # action space prevents pathological jerk
angular_accel_penalty_weight:  0.0     # bounded by integration limits
speed_deviation_weight:        -1e-2   # keep, biases toward preferred speed
action_rate_weight:            0.0     # disabled
```

The `speed_deviation_weight` is the only smoothness term worth keeping because it pulls the policy toward biomechanically realistic walking speeds rather than always running at max. Reconsider after running validation.

---

## 6. Validation Observables

For both Layer 1 and Layer 2, the same set of measurements applies. Compare new policies against the current policy on:

### 6.1 Trajectory smoothness

- **Empirical jerk distribution** over all active agents in evaluation episodes. Plot histogram, compare to PeTrack-derived distribution from IAS-7 experiments.
- **Velocity autocorrelation function** averaged over agents. Decay timescale should match or approach experimental values (typically several hundred ms).
- **Angular acceleration distribution** of torso. Should be unimodal around zero with thin tails.

### 6.2 Speed-turning coupling (Layer 2 specifically)

Plot agent speed against curvature (turning radius) for trajectory segments. Real pedestrians show a clear negative correlation: tight turns happen at low speed. The current policy should show weak or no correlation; Layer 2 should show a clear negative slope matching Hicheur 2007 data.

### 6.3 Bottleneck behaviour

In a Tier 3+ scenario with a narrow passage, measure:

- Distribution of torso orientations relative to passage direction at entry.
- Time-to-cross and flow rate.
- Compare to RiMEA bottleneck test data if available.

Layer 2 should produce visible torso rotation at bottleneck entry without explicit reward for shoulder-pass behaviour.

### 6.4 Aggregate flow validation

- Fundamental diagram (density vs flow) in corridor scenarios. Should match IAS-7 corridor experiment data.
- Lane formation in counterflow scenarios. Layer 2 policies should form lanes at densities and timescales consistent with experimental observations.

---

## 7. Risks and Open Questions

**Risk:** Layer 2 retraining may not converge as easily as the current setup. The action space is harder to plan over (true second-order dynamics). Mitigation: start with a softer biomechanical envelope (looser yaw coupling) and tighten over training. This is a curriculum on the dynamics itself.

**Risk:** The speed-yaw coupling may make some narrow passages physically unsolvable. Mitigation: validate solvability at episode generation time using the actual dynamics (treat the navmesh A* as a lower bound and verify a kinematically feasible trajectory exists). This may be heavy; alternative is to soft-bound coupling (penalty rather than clamp) initially.

**Open question:** Should longitudinal acceleration also be speed-coupled? Literature is weaker here. Recommend starting without, revisiting if validation against IAS-7 data shows the gap.

**Open question:** Should `head_rate` be exposed in the K-NN social block (other agents' head rates)? Current design only exposes torso orientation, not its rate. There is a behavioural argument that pedestrians read intent from others' head and body motion, not just position; the K-NN block could carry yaw rate of neighbours at the cost of 8 extra obs dims. Defer until basic Layer 2 works.

**Open question:** ONNX dynamic shape support. If `obs_dim` changes between training and deployment, the exported model must support either dynamic input shape or be exported per obs_dim variant. Verify with a unit test on the existing export pipeline.

---

## 8. Recommended Execution Order

1. **Layer 1 implementation** (1 day). Parameter retune, optional rename, smoothness weight rebalance.
2. **Layer 1 validation run** (1 to 2 days on JURECA). Single policy, current curriculum. Compare smoothness metrics and completion rate.
3. **Decision point.** If Layer 1 produces visibly improved motion with preserved completion, proceed. If not, diagnose whether the rate caps were too tight or the policy needs more steps to adapt.
4. **Layer 2 design freeze** (0.5 day). Confirm 3D vs 4D action space (non-holonomic vs lateral component), confirm observation layout, confirm speed-yaw coupling shape.
5. **Layer 2 implementation** (3 to 5 days). All files in section 4.7. Pay special attention to CPU/GPU parity.
6. **Layer 2 unit and equivalence tests** (1 day). Existing test_equivalence.py is the critical gate.
7. **Layer 2 training run** (3 to 7 days on JURECA, cold start). Curriculum and hyperparameters as in section 5.
8. **Validation against IAS-7 data** (concurrent with training, completed after). Section 6 observables.
9. **Paper-ready writeup** of the dynamics design as a methods section, with the speed-yaw coupling as the key methodological contribution and the emergent behavioural validation as the result.

Total elapsed time roughly 2 to 3 weeks of focused work, gated by compute availability. The implementation cost is small; the validation cost is most of the time budget.

---

## Appendix A: References for biomechanical values

- Bohannon RW (1997). "Comfortable and maximum walking speed of adults aged 20-79 years: reference values and determinants." Age and Ageing 26(1):15-19.
- Hicheur H, Vieilledent S, Richardson MJE, Flash T, Berthoz A (2007). "Velocity and curvature in human locomotion along complex curved paths: a comparison with hand movements." Experimental Brain Research 162:145-154.
- Knoblauch RL, Pietrucha MT, Nitzburg M (1996). "Field studies of pedestrian walking speed and start-up time." Transportation Research Record 1538:27-38.
- Helbing D, Molnár P (1995). "Social force model for pedestrian dynamics." Physical Review E 51(5):4282-4286.

## Appendix B: Glossary of terms used

- **First-order dynamics**: action sets a kinematic target (position, velocity, angle); next state is the target (modulo a filter).
- **Second-order dynamics**: action sets an acceleration; next state integrates over previous state plus acceleration.
- **Yaw rate**: angular velocity of body around the vertical axis.
- **Holonomic motion**: lateral and forward motion are independently controllable.
- **Non-holonomic motion**: motion is only along the body's forward axis (plus rotation); pedestrians and bicycles are non-holonomic to first approximation.
- **Speed-yaw coupling**: the empirical observation that maximum admissible turning rate decreases as forward speed increases.

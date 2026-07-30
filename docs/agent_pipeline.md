# CrowdRL Agent Pipeline

How observations are built, how the shallow neural network output dictates agent
movement, which physics models constrain agents, and how interactions / collisions
are detected, resolved, and penalised.

---

## 1. Perception: building the observation vector (80-D base, up to 129-D)

Every timestep, each agent receives an **egocentric** observation assembled in
`crowdrl_core/observation.py` from a `WorldState` struct. Everything is rotated
into the agent's own torso-heading frame via a 2-D rotation matrix built from
`-ego_heading`.

| Component | Dims | Details |
|-----------|------|---------|
| **Ego state** | 8 | Goal direction (2, unit vector), velocity (2, ego frame), scalar speed (1), preferred speed (1, raw m/s), torso angle (1, always 0 in ego frame), head angle relative to torso (1, wrapped to [-pi, pi]) |
| **Social** | 56 | 8 nearest neighbours x 7: relative position (2), relative velocity (2), body orientation relative to ego (1), shoulder width (1), chest depth (1). Zero-padded when fewer than 8 neighbours exist |
| **Raycasts** | 16 | 16 rays over 200 deg FOV anchored to the **head** (not torso). Max range 5 m. Each ray yields a normalised distance in [0, 1] where 1.0 = max range / no hit. Optional 2-channel mode adds a hit-type channel |
| **Navmesh** *(optional)* | 3 | Next-waypoint direction in ego frame (2) + path deviation from A\* shortest path (1). With `use_jupedsim_style_routing` the waypoint is served router-style at JuPedSim's fixed 0.2 m portal inset and `path_deviation` is pinned to 0.0 |
| **Temporal memory** *(optional)* | 6 | Own-trajectory summary: displacement-from-spawn, cumulative path, path efficiency, elapsed fraction, windowed displacement + goal-progress (window W=50) |
| **Neighbour velocity history** *(optional)* | K x 2 = 16 | Per tracked neighbour: velocity change over the last W_n=5 steps in ego frame (acceleration proxy). Needs the persistent neighbour-ID tracker |
| **Neighbour trajectory** *(optional)* | K x 3 = 24 | Per tracked neighbour: its own path-efficiency + windowed displacement/goal-progress. Off in the current run |

**Total**: `8 + (8 x 7) + 16 = 80` base (single-channel rays, all optional blocks off). The optional blocks add +3 (navmesh), +6 (temporal memory), +16 (neighbour velocity history), +24 (neighbour trajectory); 2-channel rays add a further +16. The **production config** (shipped `example_model/policy_r0125.onnx`) enables navmesh + temporal memory -> **obs_dim = 89**, with `use_goal_direction = false` (goal-direction slots kept but zeroed -- the agent navigates by the routed waypoint alone) and `use_jupedsim_style_routing = true` (router-style waypoint at JuPedSim's fixed 0.2 m portal inset, `path_deviation` pinned to 0.0). The earlier Layer 1 v2 line (navmesh + temporal memory + neighbour velocity history) was 105D; full instrumentation reaches 129.

Key implementation details:
- Goal direction is safe-divided (zero vector if at goal).
- Batch path (`build_observations_batch`) is fully vectorised for training throughput.
- Social sensing uses `argpartition` (O(N) per row) for KNN, not full sort.
- Ray-wall intersection: standard parametric ray-segment test.
- Ray-agent intersection: transform ray into ellipse-local frame (where the
  ellipse becomes a unit circle), solve the standard quadratic.
- Disabled channels keep their slots: `use_goal_direction=False` zeroes the two
  goal-direction entries and jupedsim-style routing zeroes `path_deviation`, so
  obs_dim never changes. Principle: do not train on a signal deployment cannot
  supply.
- The batch builder sanitises its output with `nan_to_num`; the per-agent
  builder (the one the JuPedSim adapter calls) does not -- a known open
  asymmetry (see plan v9 carried items).

---

## 2. The neural network: a shallow actor-critic

Defined in `crowdrl_train/networks.py`. Both actor and critic are **separate**
2-hidden-layer MLPs (no shared trunk, per Andrychowicz et al. 2021):

```
Actor:   obs_dim --> [256, tanh] --> [256, tanh] --> 4  (action means)
Critic:  obs_dim --> [256, tanh] --> [256, tanh] --> 1  (state value)
```

Hidden width is (256, 256) by default (`NetworkConfig`); the shipped r0125 run
trains (512, 512). `obs_dim` is 80 for the bare default observation, 83 with
the training driver's defaults (navmesh signals on), and 89 in the production
configuration (navmesh + temporal memory); see Section 1.

### Policy distribution

The actor outputs 4 means (mu). A **state-independent** learnable `log_std`
parameter (initialised to `log(0.5) ~ -0.693`, clamped to [-5, 0]) defines a
diagonal Gaussian over a pre-squash variable: a raw sample `u ~ N(mu, sigma^2)`
is passed through `tanh` to produce the action in [-1, 1]. The policy is
**tanh-squashed, not clipped**: `log_prob` is the Gaussian log-density of the
raw sample plus the tanh change-of-variables correction (via the numerically
stable `2 * (log 2 - u - softplus(-2u))` identity), and the rollout buffer
stores the raw pre-squash sample so the PPO ratio re-evaluates the same
density. The deterministic (eval / ONNX export) action is `tanh(mu)`. This
replaced the earlier clip-only scheme (Huang et al. 2022, detail #27) in the
2026-06 stabilisation campaign, together with truncation-aware GAE.

### Initialisation

- **Orthogonal init** via numpy QR decomposition.
- Hidden layer gain: `sqrt(2)`.
- Actor output gain: `0.01` (near-zero initial actions).
- Critic output gain: `1.0`.
- All biases: zero.

### Parameter sharing

All agents use the same network weights. Heterogeneity (body size, preferred
speed) enters through the observation vector, not through separate networks.

---

## 3. Action interpretation: from 4 scalars to movement

The 4-D output in [-1, 1] is mapped by `crowdrl_core/action.py`:

| Output | Raw range | Physical quantity | Default range |
|--------|-----------|-------------------|---------------|
| `a[0]` | [-1, 1] | Desired speed | [-0.5, +2.0] m/s asymmetric (linear: `-0.5 + (a+1)/2 * 2.5`; negative = backing up) |
| `a[1]` | [-1, 1] | Heading change (velocity direction) | +/-1.15 deg/step (115 deg/s) |
| `a[2]` | [-1, 1] | Torso orientation change | +/-0.57 deg/step (57 deg/s) |
| `a[3]` | [-1, 1] | Head orientation change | +/-1.72 deg/step (172 deg/s), hard-clamped to +/-90 deg from torso |

The speed remap is one straight line over [-0.5, +2.0], so a zero action
commands +0.75 m/s -- standing still requires a[0] = -0.6.

### Speed-turn coupling (optional; ON in the shipped run)

`ActionConfig.speed_turn_coupling` (default **False**) adds a speed-dependent
turn envelope on top of the flat caps: the per-step heading and torso deltas
are additionally clamped to `min(turn_pivot_rate, turn_lat_accel / v) * dt`
(defaults `turn_pivot_rate` = 120 deg/s, `turn_lat_accel` = 2.0 m/s^2; the
head channel is never coupled). The effective cap is
`min(flat cap, coupled cap)`, so with stock flat caps the coupling only binds
above ~1.0 m/s. The shipped r0125 run trains with coupling ON
(`turn_pivot_rate_deg: 240`, `turn_lat_accel: 2.0`) and raises the flat
heading/torso caps to 4.8 deg/step so the coupled envelope is the real
governor: ~2.4 deg/step near standstill, ~1.15 deg/step at 1 m/s,
~0.57 deg/step at 2 m/s. Agents must slow down to turn sharply -- the
anti-ice-skating constraint expressed as physics rather than reward.

**Important nuance**: In `CrowdEnv.step()` (crowd_env.py:338-354), both
`current_headings` and `current_torsos` are passed as
`self._world.torso_orientations`, and the returned `new_headings` is discarded
once the desired velocity is computed. Heading is **not stored** as a state
variable -- it is re-anchored to the current torso orientation every step, gets
a delta applied, and is used purely to compute the desired velocity direction.
It can therefore never sit more than one per-step heading delta away from the
torso, making the torso rate the true steering bottleneck. The JuPedSim
adapter anchors identically (divergence channel 8, fixed 2026-07-30):

```
new_heading = current_torso_orientation + a[1] * 0.020   # a[1] * max_heading_change (rad)
desired_velocity = desired_speed * [cos(new_heading), sin(new_heading)]
```

Only the torso and head orientations are written back into `WorldState`. The
"heading" is therefore a per-step velocity-direction command rather than a
persistent state.

Head and torso are **independently actuated**:
- The torso rotates the collision ellipse.
- The head steers where the 16 raycasts point.
- This lets an agent look around a corner while walking straight.

---

## 4. Physics model: semi-implicit Euler with velocity damping

The dynamics are first-order (kinematic + forces), not second-order Newtonian.
The full sequence in `CrowdEnv.step()` (crowd_env.py:327-400) is:

### Step 1: Velocity blending (exponential filter)

```
v_new = w * v_desired + (1 - w) * v_old      # w = desired_velocity_weight
```

This provides inertia: agents cannot instantly change direction. The config
default is `w = 0.05` -- only 5% of the network's desired velocity applied per
step, a first-order low-pass with tau ~200 ms at dt=0.01s (the Layer 1
recalibration; it was 0.8 before). **The shipped r0125 line trains at
`w = 0.8`** (tau ~12 ms, filter nearly transparent); its inertia comes from
the speed-turn coupling envelope and contact physics instead. The trained
value travels in the ONNX `crowdrl.dynamics` metadata (schema v2), so
deployment self-configures to match the run -- `w` is a per-run dynamics
choice, not a constant.

### Step 2: Contact force impulse

```
v += contact_forces * dt      (dt = 0.01 s)
```

Forces from both agent-agent collisions and wall repulsion are applied as
velocity impulses. `contact_forces` is already in acceleration units: core
computes forces in Newtons and divides by per-agent mass before returning
(the `# implicit unit mass` comment at the call site is stale).

### Step 3: Speed clamping

```
max_vel = 3.0 m/s        # max_velocity_magnitude (sits above max_forward_speed = 2.0)
if ||v|| > max_vel:
    v *= max_vel / ||v||
```

Prevents contact forces from launching agents at unrealistic speeds while still
allowing brief above-preferred-speed bursts (e.g., being pushed by a crowd).

### Step 4: Position update (explicit Euler)

```
positions += velocities * dt
```

### Step 5: Wall boundary enforcement

Hard constraint: any agent that has penetrated the walkable polygon boundary is
projected back inside with body clearance, and its velocity component into the
wall is cancelled. The returned wall-contact mask feeds the
`wall_collision_penalty` reward term. See section 5.

### Physics parameters

Forces are computed in Newtons and divided by per-agent mass to produce
accelerations: `v += (F / mass) * dt`. Agent masses are sampled from
N(80, 15) kg at spawn (clamped >= 40 kg), so lighter agents are pushed
harder than heavier ones -- matching real crowd dynamics.

| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| `dt` | 0.01 | s | `CrowdEnvConfig.dt` |
| `desired_velocity_weight` | 0.05 (shipped run: 0.8) | -- | `CrowdEnvConfig.desired_velocity_weight`; per-run dynamics choice, travels in ONNX metadata |
| `contact_stiffness` | 30,000 | N / overlap | `CrowdEnvConfig.contact_stiffness` |
| `contact_damping` | 500 | N*s/m | `CrowdEnvConfig.contact_damping` |
| `max_velocity_magnitude` | 3.0 | m/s | `CrowdEnvConfig.max_velocity_magnitude` (hard velocity clamp, safety against contact-force blowup) |
| `agent mass` | ~80 | kg | `SpawnConfig.mass_mean` |

---

## 5. Collision detection and response

Defined in `crowdrl_core/collision.py`. Agents are **ellipses** (not circles),
parameterised by `shoulder_width` (lateral semi-axis) and `chest_depth` (forward
semi-axis), rotated by `torso_orientation`.

### 5.1 Agent-agent collision detection

**Broad phase**: Pairwise squared-distance pre-filter. Only test pairs where
`dist^2 < (radius_i + radius_j)^2 * 4`.

**Narrow phase** (boundary-distance approach): For each candidate pair (i, j):

1. Compute the direction vector from i to j.
2. Find the boundary point of ellipse i that is closest to j along this direction (in i's rotated local frame).
3. Find the boundary point of ellipse j closest to i the same way.
4. Sum the "boundary reach" of both ellipses. If the sum exceeds the centre-to-centre distance, the ellipses overlap.
5. Overlap = penetration_depth / sum_of_reaches (normalised to [0, 1]).

This properly detects edge-on collisions that the previous centre-in-ellipse
proxy missed, and gives a smooth gradient for the spring-damper model.

### 5.2 Agent-agent contact forces (spring-damper)

For each colliding pair (i, j):

```
normal = (pos_j - pos_i) / ||pos_j - pos_i||
rel_vel_normal = dot(vel_i - vel_j, normal)

force_N = 30000 * overlap + 500 * max(rel_vel_normal, 0)
          ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          spring (N)         damping (N, only when approaching)

accel_i = -force_N / mass_i * normal   (pushed away from j)
accel_j = +force_N / mass_j * normal   (pushed away from i)
```

Forces are in Newtons and divided by per-agent mass to yield accelerations.
Lighter agents (40 kg) are pushed ~2x harder than heavier ones (80 kg).
Forces are accumulated with `np.add.at` to correctly handle agents in
multiple simultaneous collisions.

### 5.3 Wall repulsion forces (exponential)

Smooth exponential repulsion following JuPedSim's BoundaryRepulsion model:

```
f_mag = 400.0 * exp((agent_radius - dist_to_wall) / 0.3)
accel  = f_mag / mass
```

- `wall_strength = 400.0 N` (amplitude)
- `wall_range = 0.3 m` (length scale)

This provides a continuous gradient that ramps up sharply within ~30 cm of a
wall, giving the policy a smooth signal to learn from (unlike hard boundary
clipping, which has zero gradient).

Forces are computed against **all wall segments** (vectorised via batch
point-to-segment nearest) with no distance cutoff, then summed per agent --
the exponential makes far segments negligible. `wall_strength` / `wall_range`
are core-level defaults not exposed on `CrowdEnvConfig`; the torch training
env duplicates the same 400 N / 0.3 m values in its own `EnvConfig`.

### 5.4 Hard wall boundary enforcement

After physics integration, `enforce_wall_boundaries()` acts as a safety net:

1. **Fast pre-filter**: Skip agents far from all wall segments (vectorised
   distance check). Also flags agents moving fast enough to have crossed a
   boundary in one step. The pre-filter is skipped entirely when the polygon
   has holes -- every active agent then takes the detailed path.
2. **Detailed check**: For agents near walls, uses Shapely polygon containment.
   If outside or too close to boundary:
   - Find nearest point on polygon boundary.
   - Compute inward normal.
   - Project agent to `nearest_point + radius * inward_normal`.
   - Cancel velocity component into wall: `v += max(dot(v, -inward), 0) * inward`.

The function returns a per-agent wall-contact mask, consumed by the
`wall_collision_penalty` reward term.

---

## 6. Reward function (three tiers)

Computed in `crowdrl_env/reward.py`. A `RewardState` object tracks previous-step
quantities for temporal derivatives (velocities, accelerations, headings).

### Tier 1: Sparse task rewards

| Signal | Value | Condition |
|--------|-------|-----------|
| Goal reached | +10.0 | `||pos - goal|| < 0.5 m` |
| Collision (agent-agent) | -1.0 per step | While overlapping another agent |
| Agent proximity (graded ramp) | -0.005 .. -0.0001 per step | Graded linear ramp between contact (`r_i + r_j`) and `personal_space_radius` (1.0 m absolute). Each agent pays the most-negative per-pair penalty from any neighbour in the zone |
| Wall proximity | -0.1 per step | `dist_to_wall < 1.5 * agent_radius` |
| Wall collision | -1.0 per step | While `enforce_wall_boundaries` reports hard wall contact (`wall_collision_penalty`; 0.0 disables; shipped run -0.5) |
| Timeout | -5.0 | Episode reaches max_steps; also paid by agents removed by the optional stuck-termination check (default off) |
| Existence | -0.01 per step | While agent is active (time pressure) |

The **agent proximity penalty** is a reward signal (not a physics force) that
teaches the policy to maintain personal space. Unlike JuPedSim's Social Force
Model which prescribes exponential repulsion as a world force, CrowdRL lets
the policy discover its own avoidance strategy through this tunable reward
term. See Project Plan v9, Section 3.2, and Section 8.3 below for the
current graded-ramp form.

**Impact-speed (velocity-weighted) variants.** `use_velocity_weighted_collision`
and `use_velocity_weighted_proximity` (both default off, both ON in the shipped
run) scale the collision / proximity penalties by approach speed:
`multiplier = max(floor + scale * closing_speed, 0)`, with the closing speed
taken against the worst offending neighbour and capped at 10 m/s (defaults:
collision floor 0.5 / scale 0.5, proximity floor 0.25 / scale 0.5). The
wall-collision penalty is scaled by the agent's own pre-contact speed when the
collision weighting is on. `collision_penalty_cap` (default 0.0 = off; shipped
run -2.0) is a discount-only floor on the scaled per-step collision penalty.
Net effect: standing in contact is cheap, plowing in at speed is expensive --
the retune that un-froze moderate-density scenarios.

### Tier 2: Progress shaping (potential-based)

```
reward += progress_weight * (prev_distance - current_distance)
```

The distance is the **navmesh remaining-path length** (route-aware;
straight-line only as a fallback when no navmesh is present), so progress
along the routed path is rewarded even where it points away from the goal as
the crow flies. Positive when advancing, negative when retreating;
potential-based either way, so it does not introduce spurious optima. Default
`progress_weight = 1.0` (shipped run: 2.0).

### Tier 3: Smoothness priors

| Penalty | Weight | What it discourages |
|---------|--------|---------------------|
| Jerk (change in acceleration) | -1e-5 * \|\|da/dt\|\| | Sudden acceleration changes |
| Angular acceleration | -1e-2 * \|d_omega/dt\| | Rapid heading oscillations |
| Speed deviation | -5e-3 * \|v - v_preferred\| | Deviating from natural walking speed |
| Action rate | -1e-2 * \|\|a_t - a_{t-1}\|\| | Chattering/oscillating policy outputs (enabled in Layer 1) |

Jerk and angular acceleration require two steps of history to compute
(acceleration needs previous velocity, jerk needs previous acceleration).
Action rate needs one step of history. The angular-acceleration penalty is
measured on the torso orientation (the heading is torso-anchored). The
smoothness weights are kept deliberately small so they regularise without
dominating the progress and collision signals in congested scenarios.

Note the shipped r0125 run disables the whole smoothness block
(`use_smoothness: false`, `speed_deviation_weight: 0.0`) and regularises via
the action-rate penalty alone (-0.001). It consequently ignores its
preferred-speed observation and cruises near the 2.0 m/s ceiling -- notebook
11 clamps the commanded speed as an interim shim; retraining with the
speed-deviation term on and randomised preferred speeds is the planned fix.

---

## 7. The full loop (single timestep)

```
observations (N, 89)             80D base + 3D navmesh + 6D temporal memory
       |                         (production config; goal-direction slots zeroed,
       |                          path_deviation pinned under jps-style routing)
       v
  Actor network: obs -> mu (4) + sigma
  Sample u ~ N(mu, sigma^2), action = tanh(u)
       |
       v
  interpret_actions_batch()
  a[0] -> desired speed ([-0.5, +2.0] m/s asymmetric)
  a[1] -> heading change (re-anchored to torso each step -> velocity direction)
  a[2] -> torso change (rotates collision ellipse)
  a[3] -> head change (steers raycasts, clamped +/-90 from torso)
  (speed-turn coupling, if enabled: heading/torso deltas additionally
   clamped to min(turn_pivot_rate, turn_lat_accel / v) * dt)
       |
       v
  Velocity blending: v = w * v_desired + (1 - w) * v_old
       |
       v
  detect_collisions(): pairwise ellipse boundary-distance test
       |
       v
  compute_contact_forces():
    - Agent-agent: spring-damper (k=30kN, c=500N*s/m), F/mass -> accel
    - Walls: exponential repulsion (400N, range=0.3m), F/mass -> accel
       |
       v
  v += accel * dt,  clamp ||v|| <= 3.0 m/s
  pos += v * dt
  enforce_wall_boundaries()  -> wall-contact mask
       |
       v
  compute_rewards():
    goal +10 / collision -1 (optionally impact-speed-scaled, capped)
    wall proximity -0.1 / wall collision -1
    agent proximity: graded ramp -0.005..-0.0001 (worst neighbour,
      per-pair contact r_i+r_j, absolute 1.0 m personal_space_radius,
      optionally closing-speed-scaled)
    existence -0.01 / progress (navmesh path distance) / smoothness
    action rate
       |
       v
  Update active mask (goal reached / optional stuck termination),
  check timeout truncation
       |
       v
  Rebuild observations from updated WorldState
```

Critic evaluates V(s) from the same observation for GAE advantage
estimation (gamma=0.99, lambda=0.95; truncation-aware -- truncated segments
bootstrap from the critic at the post-step observation). PPO updates run up
to 10 full-batch epochs per rollout (n_minibatches=1) with clip ratio
epsilon=0.2, entropy bonus 0.01 (shipped run 0.003), and KL early stopping:
stop when the approximate KL -- averaged across DDP ranks -- exceeds
1.5 * target_kl (target_kl = 0.02).

---

## 8. Implemented improvements (formerly "gaps")

### 8.1 Wall proximity penalty -- IMPLEMENTED

Flat penalty while an agent is within `threshold * body_radius` of a wall --
a binary band, not a distance-graded signal (the smooth gradient near walls
comes from the physics-side exponential repulsion). It fires before hard wall
contact; hard contact itself is penalised separately by
`wall_collision_penalty` (-1.0 default).

```python
wall_proximity = min_wall_distance < (agent_radius * 1.5)
rewards[wall_proximity & active_mask] += -0.1
```

Configurable via `RewardConfig.wall_proximity_penalty` (default -0.1) and
`RewardConfig.wall_proximity_threshold` (default 1.5x agent radius).

### 8.2 Smoothness improvements -- IMPLEMENTED

**A. Action rate penalty** -- Penalises frame-to-frame changes in the raw policy
output. Configured via `RewardConfig.action_rate_weight` (default 0.0 -- disabled).
Targets the network's output before the nonlinear action interpretation.

**B. Biomechanical orientation limits** -- Per-step rate caps were re-grounded
in the human walking envelope in the agent-dynamics refactor (Layer 1): heading
0.020 rad/step (115 deg/s), torso 0.010 rad/step (57 deg/s), head 0.030 rad/step
(172 deg/s). These replaced the earlier pi/12 (1500 deg/s) heading/torso and
pi/3 (6000 deg/s) head caps, which were far above human capability. See
`plan/agent_dynamics_refactor.md` and Project Plan v9, Section 3.3.

### 8.3 Agent proximity penalty (graded linear ramp) -- IMPLEMENTED

Reward-side social-distance signal. This is a reward signal (not a physics
force) that teaches the policy to maintain personal space. Unlike JuPedSim's
deterministic repulsion forces, this lets the policy learn its own avoidance
strategy.

The current form is a **graded linear ramp** over the center-to-center
distance to the worst-offending neighbour, not a binary threshold:

```python
# Pairwise center-to-center distance (E, N, N) or (N, N)
pair_dist    = ||pos_i - pos_j||
pair_contact = agent_radius_i + agent_radius_j      # per-pair contact distance

# Linear ramp: 1 at contact -> 0 at personal_space_radius
t = clip((pair_dist - pair_contact) /
         (personal_space_radius - pair_contact), 0, 1)
pair_penalty = (1 - t) * agent_proximity_penalty_near \
             +      t  * agent_proximity_penalty_far

# Each agent receives the most-negative per-pair penalty (worst neighbour).
rewards[i] += min_j pair_penalty[i, j]   # self and inactive pairs masked
```

Configurable via three `RewardConfig` / `EnvConfig` fields:

| Field | Default | Meaning |
|-------|---------|---------|
| `agent_proximity_penalty_near` | -0.005 | Per-pair penalty at contact distance `r_i + r_j` |
| `agent_proximity_penalty_far`  | -0.0001 | Per-pair penalty right at `personal_space_radius` |
| `personal_space_radius`        | 1.0 m | Absolute centre-to-centre cutoff (not body-relative) |

The aggregation is `min` over neighbours, so an agent inside a crowd is
penalised by its single nearest neighbour rather than by a sum over
density. The previous implementation had a flat `-0.005` inside a
`2.0 * agent_radius` threshold; the graded ramp supplies a continuous
gradient from 1 m all the way down to contact. The pair-distance
computation now lives inside `compute_rewards` itself (CPU and torch paths);
the `agent_distances` parameter has been removed from the function
signature.

### 8.4 Mass-based inertia -- IMPLEMENTED

Contact forces are now computed in Newtons and divided by per-agent mass
(F=ma) to produce accelerations. Agent masses are sampled from N(80, 15) kg
at spawn. This means lighter agents are pushed harder and heavier agents
resist more, matching real crowd dynamics.

### 8.5 Boundary-distance overlap detection -- IMPLEMENTED

The previous centre-in-ellipse algebraic proxy missed edge-on collisions
where ellipse boundaries overlapped but neither centre was inside the other
ellipse. The new boundary-distance method computes the closest boundary
points of each ellipse along the line connecting their centres, detecting
overlap when the sum of boundary reaches exceeds the centre distance.

### 8.6 Rollout collector: cross-collect episode carry-over -- IMPLEMENTED

Previously, both `RolloutCollector` (CPU subproc path) and
`TorchRolloutCollector` (GPU path) called `env.reset_all()` at the start of
every `collect()`. Any in-flight episode was discarded, and the recorded
episode statistics biased toward episodes short enough to finish inside one
rollout.

The fix is a **persistent episode state** that spans multiple `collect()`
calls:

- The initial reset happens **lazily** on the very first `collect()`.
- Subsequent calls reuse the existing env + episode tracking state, so an
  episode that straddles a collect boundary counts the full episode reward
  across both rollouts.
- The first segment in a new collect is treated as "segment 0" (possibly a
  carry-over), and GAE handles it as a regular trailing-incomplete segment
  bootstrapped from the critic at the segment's last observation.
- The trailing-incomplete-segment bootstrap now uses the **post-step**
  observation (`s_T`), not the already-normalised `s_{T-1}` that used to
  sit in the buffer. The post-step obs is normalised exactly once at the
  end of `collect()`.
- The torch collector slices each segment over the full `max_agents` axis
  and relies on the per-step `active_mask` to select real agents, rather
  than inferring `n_agents` from the first step (which breaks when a
  carry-over segment has terminated agents scattered across the row).

The GPU `BatchedTorchEnv` also grew an `env_tiers: list[str]` field that
records each env's current geometry tier name (e.g. `"TIER_3B"`) on every
reset, so per-tier episode statistics can be attached as
`ep_dict["geometry_tier"]` without adding a new collective.

### 8.7 Single-node multi-GPU training (DDP) -- IMPLEMENTED

Added a DD-PPO-style single-node multi-GPU path (Wijmans et al. 2019)
living in the new `crowdrl_torch/distributed.py` module, with gradient
sync and normaliser sync hooks wired into `crowdrl_train.mappo`.

| Helper | Role |
|--------|------|
| `init_distributed(backend="nccl")` | Reads `RANK` / `LOCAL_RANK` / `WORLD_SIZE` from `torchrun`, sets the CUDA device, returns `(rank, world_size, device)` |
| `cleanup_distributed()` | Destroys the process group |
| `is_distributed` / `is_main_rank` / `get_rank` / `get_world_size` | Rank queries (fall back to single-process values) |
| `allreduce_gradients(model)` | Flattens every `.grad` into one buffer, issues a single `all_reduce(SUM)`, divides by world size, unflattens |
| `TorchRunningNormalizer.sync_across_ranks()` | Merges obs-normaliser statistics via parallel Welford (weighted mean + variance) |
| `sync_reward_normalizer(rnorm, device)` | Same parallel Welford merge for the reward normalizer's return-variance tracker, plus averaged running return |
| `gather_episode_stats(local)` | `all_gather_object` episode dicts to rank 0 |
| `broadcast_curriculum_state(mgr)` | After rank 0 decides phase advancement, broadcast the new state to all ranks |
| `distributed_seed(base)` / `seed_everything` | Per-rank seed helpers |

`MAPPOUpdater` now accepts a `distributed: bool | None` flag (auto-detected
from `torch.distributed.is_initialized()` by default). When distributed,
every `actor_loss.backward()` / `critic_loss.backward()` is followed by an
`allreduce_gradients(...)` call before the optimiser step, making the
effective batch `local_batch * world_size` without any learning-rate
scaling (matching CleanRL's convention).

**KL early-stopping fix.** Under DDP, each rank's minibatch produces a
different local approximate KL. If each rank early-stops independently,
one rank can exit the epoch loop while another is still issuing gradient
all-reduces, which deadlocks NCCL on mismatched collectives. `MAPPOUpdater`
now averages the KL tensor across ranks (`all_reduce(SUM)` / `world_size`)
inside the loop and uses the **global** KL for the early-stop decision, so
all ranks agree. Regression tests live in
`packages/crowdrl-train/tests/test_mappo.py` (two subprocess-based tests
that spin up a `world_size=1` gloo group and spy on the KL collective).

Launch pattern:

```
torchrun --standalone --nproc_per_node=N train_mappo.py
```

Full design rationale, synchronisation table and launch script are in
`plan/ddp_single_node.md`.

### 8.8 Export wrapper device isolation -- IMPLEMENTED

`PolicyForExport` (in `crowdrl_train/export.py`) now **deep-copies** the
actor's `feature_net` and `action_mean` before wrapping them. Previously it
held references, so a downstream `wrapper.cpu()` would silently move the
original actor's parameters to CPU -- breaking any subsequent GPU operation
on the training model. Regression tests in
`packages/crowdrl-train/tests/test_export.py` verify that `export_onnx`
leaves the source actor on its original device.

### 8.9 Speed-turn coupling -- IMPLEMENTED

Optional speed-dependent turn envelope in the action interpreter (core, env
and torch paths): heading/torso deltas clamped to
`min(turn_pivot_rate, turn_lat_accel / v) * dt` when
`ActionConfig.speed_turn_coupling` is on (default off; ON at a 240 deg/s
pivot rate in the shipped run). See Section 3.

### 8.10 Impact-speed reward weighting + wall-collision penalty -- IMPLEMENTED

Velocity-weighted collision and proximity penalties with a discount-only
`collision_penalty_cap`, plus a separate `wall_collision_penalty` on the hard
wall-contact mask. The weighting is off by default and ON in the shipped run.
See Section 6.

### 8.11 Path-based progress + optional stuck termination -- IMPLEMENTED

The progress potential switched from straight-line goal distance to the
navmesh remaining-path length (straight-line fallback), removing the
"progress points through walls" pathology in Tier 2+ geometries. An optional
stuck-termination check (`stuck_termination_enabled`, default off; window 300
steps, threshold 0.2 m of path progress) removes agents that stop making
path progress, applying the timeout penalty.

### 8.12 tanh-squashed policy + truncation-aware GAE -- IMPLEMENTED

The actor squashes samples through tanh with the change-of-variables
log-prob correction (Section 2); GAE bootstraps truncated (time-limit)
segments from the critic at the post-step observation instead of treating
them as terminal. Both landed in the 2026-06 stabilisation campaign.

### 8.13 Normalizer count caps + NaN tripwires -- IMPLEMENTED

Both running normalizers (torch obs normalizer, numpy reward/return
normalizer) cap their merged sample count at 1e8. Uncapped, the DDP sync
re-summed merged counts geometrically until `var * count` overflowed to inf
and normalized values went NaN -- the r355 (obs) and r360 (reward) collapses.
Non-finite sample rows are dropped before they can poison the statistics, and
`CROWDRL_NAN_TRIPWIRE=1` enables staged pre/post-forward checks that dump
state to /tmp and exit at the first corruption.

### 8.14 Remaining potential improvements

**C. Decrease desired_velocity_weight -- IMPLEMENTED (Layer 1).** The default
dropped from 0.8 to 0.05 in the agent-dynamics refactor, adding genuine
first-order inertia (95% carry-over, tau ~200 ms at dt=0.01s). (Renamed from
`velocity_damping`; the old name suggested the opposite direction.) Note the
shipped r0125 line nonetheless trains at 0.8 and records it in the ONNX
dynamics metadata.

**D. Temporal action smoothing** -- Low-pass filter on the policy output:
`smoothed = alpha * raw + (1 - alpha) * prev`. Guarantees smooth trajectories
at the cost of reduced agility. Not yet needed given action rate penalty results.

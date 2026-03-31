# CrowdRL Agent Pipeline

How observations are built, how the shallow neural network output dictates agent
movement, which physics models constrain agents, and how interactions / collisions
are detected, resolved, and penalised.

---

## 1. Perception: building the 79-D observation vector

Every timestep, each agent receives an **egocentric** observation assembled in
`crowdrl_core/observation.py` from a `WorldState` struct. Everything is rotated
into the agent's own torso-heading frame via a 2-D rotation matrix built from
`-ego_heading`.

| Component | Dims | Details |
|-----------|------|---------|
| **Ego state** | 7 | Goal direction (2, unit vector), velocity (2, ego frame), scalar speed (1), torso angle (1, always 0 in ego frame), head angle relative to torso (1, wrapped to [-pi, pi]) |
| **Social** | 56 | 8 nearest neighbours x 7: relative position (2), relative velocity (2), body orientation relative to ego (1), shoulder width (1), chest depth (1). Zero-padded when fewer than 8 neighbours exist |
| **Raycasts** | 16 | 16 rays over 200 deg FOV anchored to the **head** (not torso). Max range 5 m. Each ray yields a normalised distance in [0, 1] where 1.0 = max range / no hit. Optional 2-channel mode adds a hit-type channel |
| **Navmesh** *(optional)* | 3 | Next-waypoint direction in ego frame (2) + path deviation from A\* shortest path (1) |

**Total**: `7 + (8 x 7) + 16 = 79` (default, single-channel rays, navmesh off).

Key implementation details:
- Goal direction is safe-divided (zero vector if at goal).
- Batch path (`build_observations_batch`) is fully vectorised for training throughput.
- Social sensing uses `argpartition` (O(N) per row) for KNN, not full sort.
- Ray-wall intersection: standard parametric ray-segment test.
- Ray-agent intersection: transform ray into ellipse-local frame (where the
  ellipse becomes a unit circle), solve the standard quadratic.

---

## 2. The neural network: a shallow actor-critic

Defined in `crowdrl_train/networks.py`. Both actor and critic are **separate**
2-hidden-layer MLPs (no shared trunk, per Andrychowicz et al. 2021):

```
Actor:   79 --> [256, tanh] --> [256, tanh] --> 4  (action means)
Critic:  79 --> [256, tanh] --> [256, tanh] --> 1  (state value)
```

### Policy distribution

The actor outputs 4 means (mu). A **state-independent** learnable `log_std`
parameter (initialised to `log(0.5) ~ -0.693`) defines a diagonal Gaussian.
Actions are sampled from `N(mu, sigma^2)`, then clipped to [-1, 1] for the
environment. Importantly, `log_prob` is computed from the **unclipped** sample
(Huang et al. 2022, detail #27).

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
| `a[0]` | [-1, 1] | Desired speed | 0 to 1.5 m/s (linear: `(a+1)/2 * 1.5`) |
| `a[1]` | [-1, 1] | Heading change (velocity direction) | +/-45 deg/step |
| `a[2]` | [-1, 1] | Torso orientation change | +/-30 deg/step |
| `a[3]` | [-1, 1] | Head orientation change | +/-60 deg/step, hard-clamped to +/-90 deg from torso |

**Important nuance**: In `CrowdEnv.step()` (line 204-210), both `current_headings`
and `current_torsos` are initialised from `self._world.torso_orientations`.
Heading is **not stored** as a separate state variable -- it starts from the
current torso orientation each step, gets a delta applied, and is used purely to
compute the desired velocity direction vector:

```
new_heading = current_torso_orientation + a[1] * pi/4
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
The full sequence in `CrowdEnv.step()` (lines 200-258) is:

### Step 1: Velocity blending (exponential filter)

```
v_new = 0.8 * v_desired + 0.2 * v_old
```

This provides inertia: agents cannot instantly change direction. The damping
factor of 0.8 means 80% of the network's desired velocity is applied each step,
with 20% carry-over.

### Step 2: Contact force impulse

```
v += contact_forces * dt      (dt = 0.01 s)
```

Forces from both agent-agent collisions and wall repulsion are applied as
velocity impulses.

### Step 3: Speed clamping

```
max_vel = 2.0 * 1.5 = 3.0 m/s
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
wall is cancelled. See section 5.

### Physics parameters

All "forces" in this pipeline are accelerations (m/s^2) under an implicit
unit-mass convention: `v += force * dt` with no mass division. This follows
the Social Force Model tradition (Helbing & Molnar 1995).

| Parameter | Value | Source |
|-----------|-------|--------|
| `dt` | 0.01 s (10 ms) | `CrowdEnvConfig.dt` |
| `velocity_damping` | 0.8 | `CrowdEnvConfig.velocity_damping` |
| `contact_stiffness` | 2000 m/s^2 / overlap | `CrowdEnvConfig.contact_stiffness` |
| `contact_damping` | 50 1/s | `CrowdEnvConfig.contact_damping` |
| `max_speed_multiplier` | 2.0 | `CrowdEnvConfig.max_speed_multiplier` |

---

## 5. Collision detection and response

Defined in `crowdrl_core/collision.py`. Agents are **ellipses** (not circles),
parameterised by `shoulder_width` (lateral semi-axis) and `chest_depth` (forward
semi-axis), rotated by `torso_orientation`.

### 5.1 Agent-agent collision detection

**Broad phase**: Pairwise squared-distance pre-filter. Only test pairs where
`dist^2 < (radius_i + radius_j)^2 * 4`.

**Narrow phase** (algebraic distance approach): For each candidate pair (i, j):

1. Transform j's centre into i's ellipse frame via 2-D rotation by `-angle_i`.
2. Compute algebraic distance: `(x/depth_i)^2 + (y/width_i)^2`.
3. Also compute i in j's frame the same way.
4. Take the minimum. If `min < 1.0`, overlap = `1 - sqrt(min)`.

This is approximate (exact ellipse-ellipse intersection is expensive) but gives
a smooth, differentiable-enough gradient for the spring-damper model.

### 5.2 Agent-agent contact forces (spring-damper)

For each colliding pair (i, j):

```
normal = (pos_j - pos_i) / ||pos_j - pos_i||
rel_vel_normal = dot(vel_i - vel_j, normal)

force_mag = 2000 * overlap + 50 * max(rel_vel_normal, 0)
            ^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            spring term      damping (only when approaching)

force_on_i = -force_mag * normal   (pushed away from j)
force_on_j = +force_mag * normal   (pushed away from i)
```

Forces are accumulated with `np.add.at` to correctly handle agents in multiple
simultaneous collisions.

### 5.3 Wall repulsion forces (exponential)

Smooth exponential repulsion following JuPedSim's BoundaryRepulsion model:

```
f_mag = 5.0 * exp((agent_radius - dist_to_wall) / 0.3)
```

- `wall_strength = 5.0` (amplitude)
- `wall_range = 0.3 m` (length scale)

This provides a continuous gradient that ramps up sharply within ~30 cm of a
wall, giving the policy a smooth signal to learn from (unlike hard boundary
clipping, which has zero gradient).

Forces are computed against **all wall segments** (vectorised via batch
point-to-segment nearest), then summed per agent.

### 5.4 Hard wall boundary enforcement

After physics integration, `enforce_wall_boundaries()` acts as a safety net:

1. **Fast pre-filter**: Skip agents far from all wall segments (vectorised
   distance check). Also flags agents moving fast enough to have crossed a
   boundary in one step.
2. **Detailed check**: For agents near walls, uses Shapely polygon containment.
   If outside or too close to boundary:
   - Find nearest point on polygon boundary.
   - Compute inward normal.
   - Project agent to `nearest_point + radius * inward_normal`.
   - Cancel velocity component into wall: `v += max(dot(v, -inward), 0) * inward`.

---

## 6. Reward function (three tiers)

Computed in `crowdrl_env/reward.py`. A `RewardState` object tracks previous-step
quantities for temporal derivatives (velocities, accelerations, headings).

### Tier 1: Sparse task rewards

| Signal | Value | Condition |
|--------|-------|-----------|
| Goal reached | +10.0 | `||pos - goal|| < 0.5 m` |
| Collision (agent-agent) | -1.0 per step | While overlapping another agent |
| Timeout | -5.0 | Episode reaches 5000 steps |

### Tier 2: Progress shaping (potential-based)

```
reward += 0.1 * (prev_goal_distance - current_goal_distance)
```

Positive when moving toward goal, negative when moving away. Potential-based so
it does not introduce spurious optima.

### Tier 3: Smoothness priors

| Penalty | Weight | What it discourages |
|---------|--------|---------------------|
| Jerk (change in acceleration) | -0.01 * \|\|da/dt\|\| | Sudden acceleration changes |
| Angular acceleration | -0.005 * \|d_omega/dt\| | Rapid heading oscillations |
| Speed deviation | -0.1 * \|v - v_preferred\| | Deviating from natural walking speed |

These require two steps of history to compute (acceleration needs previous
velocity, jerk needs previous acceleration).

---

## 7. The full loop (single timestep)

```
observations (N, 79)
       |
       v
  Actor network: obs -> mu (4) + sigma
  Sample action ~ N(mu, sigma^2), clamp to [-1, 1]
       |
       v
  interpret_actions_batch()
  a[0] -> desired speed (0 to 1.5 m/s)
  a[1] -> heading change (current_torso +/- 45 deg -> velocity direction)
  a[2] -> torso change (+/- 30 deg -> rotates collision ellipse)
  a[3] -> head change (+/- 60 deg -> steers raycasts, clamped +/-90 from torso)
       |
       v
  Velocity blending: v = 0.8 * v_desired + 0.2 * v_old
       |
       v
  detect_collisions(): pairwise ellipse overlap test
       |
       v
  compute_contact_forces():
    - Agent-agent: spring-damper (k=2000, c=50)
    - Walls: exponential repulsion (strength=5.0, range=0.3m)
       |
       v
  v += forces * dt,  clamp ||v|| <= 3.0 m/s
  pos += v * dt
  enforce_wall_boundaries()
       |
       v
  compute_rewards():
    goal +10 / agent-collision -1 / progress / smoothness
       |
       v
  Update active mask, check termination/truncation
       |
       v
  Rebuild observations from updated WorldState
```

Critic evaluates V(s) from the same 79-D observation for GAE advantage
estimation (gamma=0.99, lambda=0.95). PPO updates run 10 epochs per rollout
with clip ratio epsilon=0.2 and entropy bonus 0.01.

---

## 8. Current gaps and proposed improvements

### 8.1 Wall collision penalty (not currently penalised)

**Current state**: The `collision_mask` passed to `compute_rewards` is built
exclusively from agent-agent collisions (`crowd_env.py` lines 232-238). Wall
contacts are resolved only through physics (exponential repulsion + hard boundary
enforcement) but agents receive **no explicit reward penalty** for pressing
against or colliding with walls.

This means the policy has no direct incentive to avoid walls beyond the implicit
signal from the exponential repulsion force slowing it down (which indirectly
penalises via the speed-deviation and progress penalties). This is a weak,
indirect signal.

**Proposed fix**: Introduce a `wall_collision_penalty` in `RewardConfig` and
build a `wall_contact_mask` from the wall boundary enforcement step. Two options:

1. **Distance-based (smooth)**: Penalise agents whose distance to the nearest
   wall segment is below a threshold (e.g., `1.5 * agent_radius`). This
   provides a learnable gradient before hard contact.

   ```python
   wall_proximity = min_wall_distance < (agent_radius * 1.5)
   rewards[wall_proximity & active_mask] += wall_proximity_penalty  # e.g. -0.3
   ```

2. **Binary (hard contact only)**: Flag agents that were actually projected by
   `enforce_wall_boundaries()` and penalise them the same way as agent-agent
   collisions.

Option 1 is preferred as it gives a smoother learning signal. The penalty weight
should be lower than agent-agent collision penalty (-1.0) since wall-hugging is
less dangerous than interpenetration, something like -0.2 to -0.5.

### 8.2 Enforcing smoother agent movements

The current Tier 3 smoothness penalties (jerk, angular acceleration, speed
deviation) are a good start. Additional mechanisms to consider:

**A. Action rate penalty (simplest, highest impact)**

Penalise large *changes* in the raw policy output between consecutive steps.
This directly discourages the actor from producing oscillating or chattering
actions, which is the root cause of jerky movement.

```python
action_change = actions_t - actions_{t-1}
action_rate_penalty = -weight * ||action_change||
```

This is more direct than penalising jerk (which is a second derivative of
position) because it targets the policy output before it passes through the
nonlinear action interpretation and velocity blending.

**B. Reduce maximum per-step orientation changes**

The current limits (45 deg/step heading, 30 deg/step torso) are generous at
dt=0.01s. At 100 Hz this allows up to 4500 deg/s heading rotation, far beyond
human capability (~120 deg/s peak). Tightening these:
- `max_heading_change`: pi/4 -> pi/12 (~15 deg/step, ~1500 deg/s -- still generous)
- `max_torso_change`: pi/6 -> pi/12

This constrains the action space mechanically rather than relying on penalties.

**C. Increase velocity damping**

Raising `velocity_damping` from 0.8 toward 0.9 or 0.95 increases inertia,
making agents heavier-feeling and naturally smoother. However, this reduces
responsiveness and may slow training convergence.

**D. Temporal action smoothing (exponential moving average)**

Apply the same damping concept to the raw action output:

```python
smoothed_action = alpha * raw_action + (1 - alpha) * prev_action
```

This is equivalent to a low-pass filter on the policy output and guarantees
smooth trajectories regardless of what the network produces.

**E. Low-pass filter on heading**

Instead of applying the heading change directly, filter it:

```python
new_heading = current + damping * (desired_change)
```

This prevents instantaneous heading reversals.

### 8.3 Recommended priority

1. **Wall collision penalty** (section 8.1, option 1) -- quick win, fixes a real
   gap in the reward signal.
2. **Action rate penalty** (section 8.2.A) -- direct, effective, one new config
   field + ~5 lines of code in reward computation.
3. **Tighten orientation limits** (section 8.2.B) -- mechanical constraint, zero
   additional compute cost.
4. **Temporal action smoothing** (section 8.2.D) -- if the above are
   insufficient, this guarantees smoothness at the cost of reduced agility.

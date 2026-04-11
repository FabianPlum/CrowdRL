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

Forces are computed in Newtons and divided by per-agent mass to produce
accelerations: `v += (F / mass) * dt`. Agent masses are sampled from
N(80, 15) kg at spawn (clamped >= 40 kg), so lighter agents are pushed
harder than heavier ones -- matching real crowd dynamics.

| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| `dt` | 0.01 | s | `CrowdEnvConfig.dt` |
| `velocity_damping` | 0.8 | -- | `CrowdEnvConfig.velocity_damping` |
| `contact_stiffness` | 30,000 | N / overlap | `CrowdEnvConfig.contact_stiffness` |
| `contact_damping` | 500 | N*s/m | `CrowdEnvConfig.contact_damping` |
| `max_speed_multiplier` | 2.0 | -- | `CrowdEnvConfig.max_speed_multiplier` |
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
| Agent proximity (graded ramp) | -0.005 .. -0.0001 per step | Graded linear ramp between contact (`r_i + r_j`) and `personal_space_radius` (1.0 m absolute). Each agent pays the most-negative per-pair penalty from any neighbour in the zone |
| Wall proximity | -0.1 per step | `dist_to_wall < 1.5 * agent_radius` |
| Timeout | -5.0 | Episode reaches max_steps |
| Existence | -0.01 per step | While agent is active (time pressure) |

The **agent proximity penalty** is a reward signal (not a physics force) that
teaches the policy to maintain personal space. Unlike JuPedSim's Social Force
Model which prescribes exponential repulsion as a world force, CrowdRL lets
the policy discover its own avoidance strategy through this tunable reward
term. See Project Plan v6, Section 3.2, and Section 8.3 below for the
current graded-ramp form.

### Tier 2: Progress shaping (potential-based)

```
reward += progress_weight * (prev_goal_distance - current_goal_distance)
```

Positive when moving toward goal, negative when moving away. Potential-based so
it does not introduce spurious optima. Default `progress_weight = 1.0`.

### Tier 3: Smoothness priors

| Penalty | Weight | What it discourages |
|---------|--------|---------------------|
| Jerk (change in acceleration) | -1e-6 * \|\|da/dt\|\| | Sudden acceleration changes |
| Angular acceleration | -1e-4 * \|d_omega/dt\| | Rapid heading oscillations |
| Speed deviation | -1e-3 * \|v - v_preferred\| | Deviating from natural walking speed |
| Action rate | 0.0 (disabled) * \|\|a_t - a_{t-1}\|\| | Chattering/oscillating policy outputs |

Jerk and angular acceleration require two steps of history to compute
(acceleration needs previous velocity, jerk needs previous acceleration).
Action rate needs one step of history. The smoothness weights are kept
deliberately small so they regularise without dominating the progress and
collision signals in congested scenarios.

---

## 7. The full loop (single timestep)

```
observations (N, 82)              79D base + 3D navmesh (when enabled)
       |
       v
  Actor network: obs -> mu (4) + sigma
  Sample action ~ N(mu, sigma^2), clamp to [-1, 1]
       |
       v
  interpret_actions_batch()
  a[0] -> desired speed (0 to 1.5 m/s)
  a[1] -> heading change (current_torso +/- 15 deg -> velocity direction)
  a[2] -> torso change (+/- 15 deg -> rotates collision ellipse)
  a[3] -> head change (+/- 60 deg -> steers raycasts, clamped +/-90 from torso)
       |
       v
  Velocity blending: v = 0.8 * v_desired + 0.2 * v_old
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
  enforce_wall_boundaries()
       |
       v
  compute_rewards():
    goal +10 / collision -1 / wall proximity -0.1
    agent proximity: graded ramp -0.005..-0.0001 (worst neighbour,
      per-pair contact r_i+r_j, absolute 1.0 m personal_space_radius)
    existence -0.01 / progress / smoothness / action rate
       |
       v
  Update active mask, check termination/truncation
       |
       v
  Rebuild observations from updated WorldState
```

Critic evaluates V(s) from the same observation for GAE advantage
estimation (gamma=0.99, lambda=0.95). PPO updates run 10 epochs per rollout
with clip ratio epsilon=0.2 and entropy bonus 0.01.

---

## 8. Implemented improvements (formerly "gaps")

### 8.1 Wall proximity penalty -- IMPLEMENTED

Distance-based smooth penalty when agents are within `threshold * body_radius`
of a wall. Provides a learnable gradient before hard wall contact.

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

**B. Tightened orientation limits** -- Heading and torso change limits reduced
from pi/4 and pi/6 to pi/12 each (~15 deg/step, ~1500 deg/s). Still above
human capability (~120 deg/s) but prevents physically impossible snap turns.

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

### 8.9 Remaining potential improvements

**C. Increase velocity damping** -- Raising `velocity_damping` from 0.8 toward
0.9 or 0.95 increases inertia. Trades responsiveness for smoother trajectories.

**D. Temporal action smoothing** -- Low-pass filter on the policy output:
`smoothed = alpha * raw + (1 - alpha) * prev`. Guarantees smooth trajectories
at the cost of reduced agility. Not yet needed given action rate penalty results.

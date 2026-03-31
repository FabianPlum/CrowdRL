# Environment Mechanics: Physics, Constraints, and Rewards

How the Gymnasium environment wraps the WorldState model, what physical rules
constrain agent movement, and how the reward function shapes implicitly learned
behaviour.

This document extends [agent_pipeline.md](agent_pipeline.md) with a focus on
**why** agents learn certain behaviours -- tracing each learned tendency back to
a specific constraint or reward signal.

---

## 1. The Gymnasium-WorldState integration

CrowdRL's environment is a standard
[Gymnasium](https://gymnasium.farama.org/) `Env` subclass
([crowd_env.py](../packages/crowdrl-env/src/crowdrl_env/crowd_env.py))
that internally manages all N agents and exposes batched arrays:

- **Observations**: `(n_agents, obs_dim)`
- **Actions**: `(n_agents, 4)`
- **Rewards**: `(n_agents,)`

All agents share one policy network (MAPPO with parameter sharing).
Heterogeneity enters through the observation, not through separate weights.

### 1.1 The WorldState contract

[WorldState](../packages/crowdrl-core/src/crowdrl_core/world_state.py) is a
flat dataclass that holds everything the perception system needs:

```
WorldState
  |
  |-- Agent arrays (per-agent, all shape (N, ...))
  |     positions          (N, 2)    metres
  |     velocities         (N, 2)    m/s
  |     torso_orientations (N,)      radians
  |     head_orientations  (N,)      radians, absolute
  |     shoulder_widths    (N,)      metres (ellipse semi-axis)
  |     chest_depths       (N,)      metres (ellipse semi-axis)
  |     goal_positions     (N, 2)    metres
  |
  |-- Geometry (shared, static within an episode)
  |     walkable_polygon   Shapely Polygon (exterior = boundary, holes = obstacles)
  |     wall_segments      (S, 2, 2)  precomputed line segments
  |     navmesh            NavMesh | None
  |
  |-- active_mask          (N,) bool | None
```

The critical design guarantee: **the observation builder and all sensing code
consume only WorldState**. They never know whether the state came from the
training environment or from a JuPedSim deployment adapter. If WorldState is
populated identically, observations are numerically identical -- this is the
sim-to-real transfer guarantee.

### 1.2 Episode lifecycle

```
reset()
  |
  v
generate_geometry()           -- procedural Shapely polygon (Tier 0-5)
  |
  v
build_navmesh()               -- constrained Delaunay triangulation + adjacency
extract_wall_segments()       -- polygon boundary -> line segments
  |
  v
spawn_agents()                -- rejection-sampled positions, heterogeneous bodies
  |                              shoulder_width ~ N(0.22, 0.02) m
  |                              chest_depth    ~ N(0.12, 0.015) m
  |                              preferred_speed ~ N(1.34, 0.26) m/s
  v
verify_solvability()          -- A* confirms every (spawn, goal) pair is reachable
  |                              with the agent's body clearance
  v
filter unsolvable agents      -- prune, regenerate, or strict (configurable)
  |
  v
Populate WorldState           -- validate array shapes
  |
  v
Build initial observations    -- build_observations_batch() -> (N, obs_dim)
  |
  v
Return (observations, info)
```

Solvability verification
([solvability.py](../packages/crowdrl-env/src/crowdrl_env/solvability.py))
runs A\* on the navmesh with per-agent portal-width checks, ensuring no agent
is given an impossible task. Three modes control what happens when unsolvable
agents are found:

| Mode | Behaviour |
|------|-----------|
| **Prune** | Remove unsolvable agents, keep the geometry |
| **Regenerate** | Discard the whole geometry if >30% agents are unsolvable |
| **Strict** | All agents must be solvable (for validation runs) |

### 1.3 The step loop

Each call to `step(actions)` in
[CrowdEnv.step()](../packages/crowdrl-env/src/crowdrl_env/crowd_env.py#L174)
runs the following pipeline:

```
raw actions (N, 4)             values in [-1, 1]
       |
       v
  1. interpret_actions_batch()     -> desired velocities + new orientations
       |
       v
  2. Velocity blending             v = 0.8 * v_desired + 0.2 * v_old
       |
       v
  3. detect_collisions()           -> list of (i, j, overlap) tuples
     compute_contact_forces()      -> agent-agent spring-damper + wall repulsion
       |
       v
  4. Physics integration
       v += forces * dt            apply impulse
       clamp ||v|| <= 3.0 m/s     prevent blow-up
       pos += v * dt               explicit Euler position update
       |
       v
  5. enforce_wall_boundaries()     hard projection back inside polygon
       |
       v
  6. compute_rewards()             sparse + shaped + smoothness signals
       |
       v
  7. Deactivate goal-reached agents, check timeout
       |
       v
  8. build_observations_batch()    -> (N, obs_dim) for next step
```

Steps 1-5 form the physics pipeline (Section A). Step 6 is the reward pipeline
(Section B). Both shape what the policy learns.

---

## Section A: Physics constraints on agent behaviour

Every physical constraint limits what actions the agent **can** take, which in
turn constrains what behaviours it **can** learn. These constraints act as an
inductive bias: the policy does not need to learn that walking through walls is
bad if the physics engine makes it impossible.

### A.1 Action interpretation: what the network controls

The 4D action output in [-1, 1] is mapped by
[interpret_actions_batch()](../packages/crowdrl-core/src/crowdrl_core/action.py#L162):

| Output | Maps to | Range | Physical meaning |
|--------|---------|-------|------------------|
| `a[0]` | Desired speed | 0 to 1.5 m/s | How fast the agent wants to walk |
| `a[1]` | Heading change | +/-15 deg/step | Which direction the velocity points |
| `a[2]` | Torso rotation | +/-15 deg/step | Rotates the collision ellipse |
| `a[3]` | Head rotation | +/-60 deg/step | Steers where the 16 raycasts point |

Speed is mapped linearly: `desired_speed = (a[0] + 1) / 2 * 1.5`, so an
output of -1 means standing still and +1 means maximum walking speed.

**What this constrains:**

- **No instant direction reversal.** At +/-15 deg per step (1500 deg/s at
  dt=0.01s), a 180-degree reversal takes at least 12 steps (0.12s). This is
  already more agile than real humans (~120 deg/s peak), but it prevents
  physically impossible teleportation-like turns.

- **Speed is bounded.** The network cannot request speeds above 1.5 m/s. The
  maximum desired speed corresponds roughly to brisk walking. Combined with
  velocity damping (A.2), the actual achieved speed is lower still.

- **Head and torso are independent.** The head can rotate up to 60 deg/step
  with a hard clamp at +/-90 deg from the torso
  ([action.py:114-116](../packages/crowdrl-core/src/crowdrl_core/action.py#L114)).
  This means an agent can look around a corner while walking straight -- or
  scan for gaps in a crowd while maintaining its walking direction.

**Heading is not persistent state.** The heading (velocity direction) starts
from the current torso orientation each step, receives a delta, and is used
only to compute the desired velocity vector. Only torso and head orientations
are written back into WorldState
([crowd_env.py:219-220](../packages/crowdrl-env/src/crowdrl_env/crowd_env.py#L219)).

### A.2 Velocity blending: simulated inertia

After the action interpreter produces a desired velocity, the environment
blends it with the agent's current velocity
([crowd_env.py:215-218](../packages/crowdrl-env/src/crowdrl_env/crowd_env.py#L215)):

```
v_new = 0.8 * v_desired + 0.2 * v_current
```

**What this constrains:**

- **No instant velocity changes.** Even if the network outputs maximum speed in
  a new direction, only 80% of that command takes effect immediately. The
  remaining 20% carries the previous velocity forward. This acts as an
  exponential low-pass filter on velocity.

- **Agents feel "heavy".** A standing agent that commands full speed ahead will
  reach 80% of desired speed in one step, 96% in two steps, and 99.2% in three
  steps. Conversely, stopping requires multiple steps of zero-speed commands.

- **Collision recovery is gradual.** After being knocked sideways by a contact
  force, the agent cannot instantly resume its desired trajectory. It must
  "steer back" over several steps, creating realistic-looking recovery
  behaviour.

### A.3 Collision detection: elliptical body model

Agents are not circles -- they are oriented ellipses parameterised by
`shoulder_width` (lateral semi-axis) and `chest_depth` (forward semi-axis),
rotated by `torso_orientation`. Detection is handled by
[detect_collisions()](../packages/crowdrl-core/src/crowdrl_core/collision.py#L69):

```
Broad phase:
  Pairwise squared-distance pre-filter
  Only test pairs where dist^2 < (r_i + r_j)^2 * 4

Narrow phase (per candidate pair):
  1. Transform j's centre into i's ellipse frame (rotate by -angle_i)
  2. Compute algebraic distance: (x/depth_i)^2 + (y/width_i)^2
  3. Repeat with i in j's frame
  4. Take the minimum; if < 1.0, overlap = 1 - sqrt(min)
```

**What this constrains:**

- **Body shape matters.** Wider agents (larger shoulders) clip into each other
  sooner when passing side-by-side. Deeper agents (larger chest) need more
  space when walking in single file. The policy receives body dimensions in its
  social observation (2 of the 7 per-neighbour features), so it can learn
  different passing strategies for different body types.

- **Torso rotation changes the collision shape.** Turning sideways makes the
  agent narrower in the direction of travel, which is exactly what real
  pedestrians do in tight gaps. The policy has an incentive to discover this
  because it reduces collision penalties.

- **Overlap is smooth, not binary.** The algebraic distance proxy yields a
  continuous overlap magnitude. Deeper penetration means larger forces (A.4),
  which means the policy experiences a gradient -- slightly brushing another
  agent is penalised less than a head-on collision.

### A.4 Contact forces: spring-damper model

When agents overlap,
[compute_contact_forces()](../packages/crowdrl-core/src/crowdrl_core/collision.py#L255)
computes a spring-damper response:

```
For each colliding pair (i, j):
  normal      = (pos_j - pos_i) / ||pos_j - pos_i||
  rel_vel_n   = dot(vel_i - vel_j, normal)

  accel = 2000 * overlap  +  50 * max(rel_vel_n, 0)
          ~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~
          spring term          damping (only when approaching)

  Agent i is pushed away from j (acceleration along -normal)
  Agent j is pushed away from i (acceleration along +normal)
```

**A note on units:** agents have no explicit mass. The "forces" returned by
`compute_contact_forces()` are applied directly as accelerations:
`v += accel * dt`. This is equivalent to assuming unit mass (1 kg) for all
agents, which is standard in pedestrian social-force models. The stiffness
constant of 2000 has units of m/s^2 per unit overlap (not N/m), and the
damping constant of 50 has units of 1/s (not N*s/m). The practical effect is
the same -- overlapping agents are pushed apart -- but there is no F=ma step
in the physics pipeline.

**What this constrains:**

- **Overlapping agents are forcefully separated.** A stiffness of 2000 m/s^2
  means even small overlaps produce large accelerations. An agent that walks
  into another agent will be pushed backwards, making it lose progress toward
  its goal.

- **Head-on collisions are worse than glancing contacts.** The damping term
  (`50 * max(rel_vel_n, 0)`) adds extra acceleration when agents approach each
  other. Two agents walking toward each other at 1 m/s each get a combined
  100 m/s^2 of damping acceleration on top of the spring term. This teaches
  the policy to avoid approaching trajectories.

- **One-sided damping prevents bouncing.** The `max(rel_vel_n, 0)` clamp means
  damping only applies when agents are approaching, not when separating. This
  prevents oscillatory bouncing that would occur with standard bidirectional
  damping.

- **Multiple simultaneous collisions accumulate.** Forces are summed with
  `np.add.at` (not simple indexing), so an agent surrounded by three others
  receives the net force from all three contacts. This correctly handles
  crowd-crush scenarios.

### A.5 Wall repulsion: exponential force field

Walls exert a smooth exponential repulsion acceleration
([collision.py:322-352](../packages/crowdrl-core/src/crowdrl_core/collision.py#L322)),
again with implicit unit mass:

```
For each agent, for each wall segment:
  dist   = distance to nearest point on segment
  radius = max(shoulder_width, chest_depth)

  f_mag  = 5.0 * exp((radius - dist) / 0.3)
```

The force direction points from the nearest wall point toward the agent.

**What this constrains:**

- **Walls have a "soft boundary" aura.** The exponential means repulsion is
  negligible beyond ~1m but ramps up sharply within 30cm. An agent 5cm from a
  wall feels ~38x the force of an agent at 30cm.

- **Smooth gradient for learning.** Unlike a hard wall constraint (which has
  zero gradient until contact), the exponential provides a continuous signal
  that the policy can learn from. The agent can "feel" the wall approaching
  before contact occurs.

- **All wall segments contribute.** In a corner, the agent feels repulsion from
  both walls simultaneously. This creates a natural tendency to walk down the
  centre of corridors.

### A.6 Speed clamping: velocity ceiling

After contact forces are applied as velocity impulses, the total speed is
clamped
([crowd_env.py:246-251](../packages/crowdrl-env/src/crowdrl_env/crowd_env.py#L246)):

```
max_vel = 2.0 * 1.5 = 3.0 m/s

if ||v|| > max_vel:
    v *= max_vel / ||v||    (rescale direction, cap magnitude)
```

**What this constrains:**

- **Contact forces cannot launch agents.** Without this clamp, a stiff spring
  (2000 m/s^2) on deep penetration could accelerate an agent to unrealistic speeds.
  The clamp allows brief bursts above the 1.5 m/s desired ceiling (e.g. being
  pushed by a crowd) but caps the maximum at 3.0 m/s.

- **Direction is preserved.** The clamping rescales the velocity vector without
  changing its direction. An agent being pushed sideways by a collision will
  continue moving sideways, just slower.

### A.7 Hard wall enforcement: the safety net

After the position update, any agent that has penetrated the walkable polygon
boundary is corrected by
[enforce_wall_boundaries()](../packages/crowdrl-core/src/crowdrl_core/collision.py#L362):

```
Fast pre-filter (vectorised):
  Skip agents far from walls (dist > 3x body radius)
  Always check agents moving fast (speed > 10x body radius per step)

Detailed check (Shapely, per-agent):
  if outside polygon OR too close to boundary:
    1. Find nearest point on polygon boundary
    2. Compute inward normal
    3. Project agent to: nearest_point + radius * inward_normal
    4. Cancel velocity component into wall
```

**What this constrains:**

- **No agent can ever exist outside the walkable polygon.** This is a hard
  constraint, not a soft penalty. Even if the exponential repulsion (A.5) is
  insufficient (e.g. due to very high contact forces), this step catches the
  violation and corrects it.

- **Velocity into walls is cancelled.** The agent does not "bounce" -- the wall
  simply absorbs its inward momentum. It can still slide along the wall (the
  tangential component is preserved).

- **Fast agents get extra scrutiny.** The pre-filter explicitly checks agents
  with high speed (`speed > 10 * radius`), because a single-step position
  update could teleport them through a thin wall. This prevents tunnelling
  artifacts.

### A.8 Summary: the physics inductive bias

Together, these constraints create an inductive bias that significantly narrows
what the policy can learn:

```
+-- Action limits (A.1) -----> No teleportation, bounded turn rates
|
+-- Velocity blending (A.2) -> Smooth trajectories, gradual acceleration
|
+-- Elliptical bodies (A.3) -> Body shape affects passability
|
+-- Spring-damper (A.4) -----> Overlapping = getting pushed apart
|
+-- Wall repulsion (A.5) ----> "Feel" walls before contact
|
+-- Speed clamp (A.6) -------> No unrealistic velocities
|
+-- Hard walls (A.7) --------> Cannot leave walkable area, ever
```

The policy never needs to learn "don't walk through walls" because the physics
makes it impossible. Instead, it learns *how to avoid the consequences* of
these constraints: the velocity loss from wall contact, the force impulse from
agent collisions, and the wasted timesteps from being pushed off course.

---

## Section B: Reward signals and what they incentivise

The reward function
([reward.py](../packages/crowdrl-env/src/crowdrl_env/reward.py)) is computed by
[compute_rewards()](../packages/crowdrl-env/src/crowdrl_env/reward.py#L112)
every timestep. It combines sparse task rewards, shaped progress signals, and
smoothness priors to guide the policy toward human-like pedestrian behaviour.

### B.1 Tier 1: Sparse task rewards

These are the primary objectives. They define *what* the agent should do.

#### Goal reaching: +10.0

```
if ||position - goal|| < 0.5 m AND agent is active:
    reward += 10.0
    agent is deactivated
```

- **Incentivises**: navigating to the assigned goal position.
- **One-time**: the bonus fires once, then the agent stops receiving
  observations and rewards.
- **Implicit learning**: since deactivation stops the existence penalty (B.3),
  reaching the goal also ends the steady drain of -0.01/step. Over a 5000-step
  episode, the cumulative existence penalty is -50 -- far outweighing the +10
  goal bonus. This creates strong pressure to reach the goal *quickly*.

#### Agent-agent collision: -1.0 per step

```
if agent is overlapping any other agent AND both are active:
    reward += -1.0    (applied every step the overlap persists)
```

- **Incentivises**: maintaining personal space, yielding, and path planning
  that avoids other agents.
- **Per-step, not per-event**: an agent stuck in a collision for 10 steps
  receives -10.0 total, equal to the goal bonus. This makes prolonged
  collisions extremely costly.
- **Symmetric**: both agents in a collision are penalised equally, even if one
  is "at fault". This prevents exploitation where one agent learns to push
  others out of the way.
- **Implicit learning**: combined with the spring-damper force (A.4), the agent
  learns that collisions both hurt (reward) and push it off course (physics).
  The double signal reinforces avoidance.

#### Timeout: -5.0

```
if episode reaches 5000 steps (50 seconds at dt=0.01):
    reward += -5.0 for all still-active agents
    all agents are truncated
```

- **Incentivises**: not dawdling, not getting permanently stuck.
- **Implicit learning**: combined with the existence penalty, the agent learns
  that standing still or taking detours is costly. But the timeout penalty alone
  is weaker than the goal bonus (+10.0), so the agent is better off reaching
  the goal even with a few collisions along the way.

#### Wall proximity: -0.3 per step

```
threshold = agent_radius * 1.5    (e.g. 0.22m * 1.5 = 0.33m)

if distance_to_nearest_wall < threshold AND agent is active:
    reward += -0.3
```

- **Incentivises**: keeping distance from walls, walking in open space.
- **Complements wall repulsion (A.5)**: the exponential force pushes the agent
  away physically, while this penalty teaches the agent to *avoid approaching*
  walls in the first place. The physics acts reactively; the reward acts
  proactively.
- **Lower than collision penalty**: -0.3 vs -1.0 reflects that wall-hugging is
  less dangerous than agent-agent interpenetration. An agent in a narrow
  corridor cannot avoid triggering this penalty, so making it too large would
  create unlearnable situations.

### B.2 Shaped progress reward

#### Progress: +0.1 * distance_closed

```
progress = previous_goal_distance - current_goal_distance
reward += 0.1 * progress
```

- **Incentivises**: moving toward the goal, not just reaching it.
- **Potential-based**: the reward depends on the *change* in distance, not the
  absolute distance. Moving 1m closer yields +0.1 regardless of how far away
  the goal is. Moving 1m farther yields -0.1.
- **Implicit learning**: this is the primary signal that teaches the agent
  *direction*. Without it, the +10.0 goal bonus is too sparse -- the agent
  would need to stumble upon the goal by random exploration before learning
  anything. The progress reward provides a continuous gradient toward the goal.
- **No spurious optima**: because it is potential-based (difference of a
  potential function `phi(s) = -distance_to_goal`), it provably does not
  change the optimal policy under the MDP. It only speeds up learning by
  providing denser signal.

### B.3 Existence penalty: -0.01 per step

```
if agent is active:
    reward += -0.01
```

- **Incentivises**: reaching the goal as quickly as possible.
- **Time pressure**: over 5000 steps, this accumulates to -50.0 -- five times
  the goal bonus. An agent that reaches the goal in 100 steps pays -1.0 in
  existence penalty; one that takes 4000 steps pays -40.0. This creates a
  strong gradient toward faster navigation.
- **Implicit learning**: this penalty interacts with every other signal. Taking
  a detour to avoid a collision costs existence penalty for the extra steps.
  Waiting for another agent to pass costs existence penalty for each step of
  waiting. The policy must trade off collision risk against time cost.

### B.4 Tier 2: Smoothness priors

These penalties discourage physically unrealistic or visually jarring movement.
They are gated behind `use_smoothness = True` and require two steps of history
(the first step after reset has no previous data).

#### Jerk penalty: -0.01 * ||jerk||

```
acceleration_t      = (velocity_t - velocity_{t-1}) / dt
jerk_t              = (acceleration_t - acceleration_{t-1}) / dt
reward             += -0.01 * ||jerk_t||
```

- **Incentivises**: smooth acceleration profiles (gradual speed changes).
- **Discourages**: sudden starts, stops, or direction reversals.
- **Implicit learning**: jerk is the third derivative of position. High jerk
  means the agent is changing its acceleration, which in pedestrian dynamics
  corresponds to jerky, unnatural gait. The penalty teaches the policy to
  "ease into" velocity changes rather than snapping to them.
- **Note**: requires **two** previous steps of history (velocity for
  acceleration, acceleration for jerk), so this penalty only activates from
  step 3 onward.

#### Angular acceleration: -0.005 * |angular_accel|

```
heading_change_t    = heading_t - heading_{t-1}          (normalised to [-pi, pi])
angular_vel_t       = heading_change_t / dt
angular_accel_t     = |angular_vel_t - angular_vel_{t-1}|
reward             += -0.005 * angular_accel_t
```

- **Incentivises**: smooth turns (constant angular velocity rather than
  stop-and-go rotation).
- **Discourages**: oscillating heading ("wiggling"), which is a common failure
  mode where the policy rapidly alternates left and right heading changes.
- **Implicit learning**: without this penalty, a policy trained only on
  progress reward tends to produce oscillatory heading as it continually
  overcorrects toward the goal. The angular acceleration penalty acts as a
  stabiliser, teaching the policy to commit to a heading and maintain it.

#### Speed deviation: -0.1 * |speed - preferred_speed|

```
speed        = ||velocity||
preferred    = per-agent preferred speed (sampled at spawn, ~1.34 m/s)
reward      += -0.1 * |speed - preferred|
```

- **Incentivises**: walking at the agent's preferred speed (not too fast, not
  too slow).
- **Per-agent calibration**: each agent has its own preferred speed drawn from
  N(1.34, 0.26) at spawn. The policy learns to modulate its speed output to
  match, even though it does not directly observe the preferred speed value.
  The speed appears implicitly through this penalty signal.
- **Implicit learning**: this is the primary "naturalism" signal. Real
  pedestrians have a strong tendency to maintain a consistent walking speed
  ([Weidmann 1993](https://doi.org/10.3929/ethz-a-000555647)). The penalty
  teaches the policy to avoid both dawdling (low speed, also penalised by
  existence penalty) and rushing (high speed, which increases collision risk).

### B.5 Optional signals (disabled by default)

#### Action rate penalty (weight = 0.0, disabled)

```
action_change  = ||actions_t - actions_{t-1}||
reward        += weight * action_change
```

When enabled with a negative weight (e.g. -0.05), this directly penalises
large changes in the raw policy output between consecutive steps. It is more
direct than the jerk penalty because it targets the network's output before
it passes through the nonlinear action interpretation and velocity blending.

#### Inverse distance to goal (weight = 0.0, disabled)

```
reward += weight * 1 / (distance_to_goal + 1)
```

A continuous proximity signal that rewards being close to the goal at every
step (not just when making progress). Disabled by default because the
progress reward (B.2) is sufficient and this signal is not potential-based,
which can introduce local optima.

### B.6 Reward budget: what dominates learning

To understand what the policy prioritises, consider the approximate magnitude
of each signal over a typical 500-step episode (5 seconds) where the agent
starts 10m from its goal:

| Signal | Per-step | Typical episode total | Notes |
|--------|----------|----------------------|-------|
| Goal bonus | one-time +10.0 | +10.0 | If reached |
| Progress | ~+0.02/step | +2.0 | 10m at ~0.1 weight |
| Existence | -0.01/step | -5.0 | 500 steps |
| Speed deviation | ~-0.03/step | -15.0 | |speed - 1.34| ~0.3 avg |
| Collision | -1.0/step | -10.0 | If stuck for 10 steps |
| Wall proximity | -0.3/step | -3.0 | ~10 steps near walls |
| Jerk | ~-0.001/step | -0.5 | Varies widely |
| Angular accel | ~-0.0005/step | -0.25 | Varies widely |

The speed deviation penalty is the largest continuous signal, which means the
policy is under strong pressure to maintain natural walking speed. The
collision penalty dominates during contact events. The progress reward and
goal bonus together provide the navigational objective.

### B.7 Emergent behaviours from reward-physics interaction

The combination of physical constraints (Section A) and reward signals
(Section B) produces emergent behaviours that are not explicitly programmed:

**Anticipatory yielding.** The agent learns to adjust its heading *before*
a collision occurs, because:
- The progress penalty for a small detour (-0.1/m) is far less than the
  collision penalty (-1.0/step for multiple steps)
- The wall repulsion force provides a gradient signal before contact
- Social observations show approaching agents' velocities, allowing prediction

**Corridor lane formation.** In bidirectional corridors, agents spontaneously
form lanes because:
- Head-on collisions produce the largest contact forces (A.4) and penalties
- Wall proximity penalty pushes agents away from walls
- The resulting equilibrium is two streams offset from centre

**Gap exploitation through torso rotation.** Agents learn to rotate their
torso when passing through narrow gaps because:
- The elliptical body model (A.3) means a rotated torso is narrower
- Collisions carry a -1.0/step penalty
- The torso rotation action is cheap (no direct penalty, only indirect
  angular acceleration cost of -0.005)

**Speed modulation near obstacles.** Agents slow down in congested areas
because:
- Collision penalty at higher speeds is costlier (more steps in contact due
  to larger contact forces needed to separate)
- Speed deviation penalty is symmetric: going too fast is penalised just as
  much as going too slow
- The existence penalty creates pressure to speed back up once the congestion
  clears

**Head scanning.** Agents learn to rotate their head independently of their
body because:
- The head controls where the 16 raycasts point (200-degree FOV)
- Raycasts provide the only information about walls and approaching agents
  from the side
- Head rotation has no collision consequence (only the torso defines the
  ellipse)
- Looking toward potential threats before they enter the body's path allows
  earlier avoidance manoeuvres

---

## Appendix: Physics and reward parameter reference

### Physics parameters ([CrowdEnvConfig](../packages/crowdrl-env/src/crowdrl_env/crowd_env.py#L40))

| Parameter | Default | Unit | Role |
|-----------|---------|------|------|
| `dt` | 0.01 | s | Simulation timestep |
| `velocity_damping` | 0.8 | -- | Blend factor: 0.8 = desired, 0.2 = carry-over |
| `contact_stiffness` | 2000 | m/s^2 / overlap | Agent-agent spring (implicit unit mass) |
| `contact_damping` | 50 | 1/s | Agent-agent approach damping (implicit unit mass) |
| `max_speed_multiplier` | 2.0 | -- | Speed clamp = 2.0 * 1.5 = 3.0 m/s |
| `max_steps` | 5000 | steps | Episode timeout (50 seconds) |

### Action limits ([ActionConfig](../packages/crowdrl-core/src/crowdrl_core/action.py#L27))

| Parameter | Default | Unit | Equivalent rate |
|-----------|---------|------|-----------------|
| `max_speed` | 1.5 | m/s | -- |
| `max_heading_change` | pi/12 | rad/step | 1500 deg/s |
| `max_torso_change` | pi/12 | rad/step | 1500 deg/s |
| `max_head_change` | pi/3 | rad/step | 6000 deg/s |
| `head_limit` | pi/2 | rad | +/-90 deg from torso |

### Wall forces (in [compute_contact_forces()](../packages/crowdrl-core/src/crowdrl_core/collision.py#L255))

| Parameter | Default | Unit |
|-----------|---------|------|
| `wall_strength` | 5.0 | m/s^2 |
| `wall_range` | 0.3 | m |

### Reward weights ([RewardConfig](../packages/crowdrl-env/src/crowdrl_env/reward.py#L21))

| Signal | Weight | Active by default |
|--------|--------|-------------------|
| `goal_bonus` | +10.0 | Yes |
| `collision_penalty` | -1.0 | Yes |
| `timeout_penalty` | -5.0 | Yes |
| `wall_proximity_penalty` | -0.3 | Yes |
| `wall_proximity_threshold` | 1.5x radius | Yes |
| `existence_penalty` | -0.01 | Yes |
| `progress_weight` | +0.1 | Yes |
| `jerk_penalty_weight` | -0.01 | Yes (Tier 2) |
| `angular_accel_penalty_weight` | -0.005 | Yes (Tier 2) |
| `speed_deviation_weight` | -0.1 | Yes (Tier 2) |
| `action_rate_weight` | 0.0 | No |
| `inverse_distance_weight` | 0.0 | No |

### Agent body dimensions ([SpawnConfig](../packages/crowdrl-env/src/crowdrl_env/spawner.py#L20))

| Parameter | Distribution | Notes |
|-----------|-------------|-------|
| `shoulder_width` | N(0.22, 0.02) m | Half-width; full shoulder ~0.44m |
| `chest_depth` | N(0.12, 0.015) m | Half-depth; full chest ~0.24m |
| `preferred_speed` | N(1.34, 0.26) m/s | Clamped to [0.5, 2.0] m/s |

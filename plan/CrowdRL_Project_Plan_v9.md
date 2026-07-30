**CrowdRL**

Learning Crowd Navigation Policies via

Multi-Agent Reinforcement Learning on Synthetic Arenas

Project Plan v9 — Draft for Internal Discussion

Dr. Fabian Plum

IAS-7 — Zivile Sicherheitsforschung

Forschungszentrum Jülich

July 2026

*Supersedes v8. The design sections (1–8) are unchanged except where reality
overtook them: Section 3.3 records the production observation line, Section 3.6
replaces the "availability caveat" with the deployment status now that the
adapter is built and the operational-model contract is verified against a local
2.0 build, and Section 4's M9 row is closed. The substantive addition is the
2026-07-30 progress-log entry: the JuPedSim deployment path is now the
canonically correct one, validated against the training engine at
millimetre-scale agreement, and the shipped example policy was trained under
the deployment routing contract.*

# 1. Executive Summary

**The Thesis.** Hand-crafted pedestrian locomotion models (social forces, velocity obstacles, generalized centrifugal force) are tuned to reproduce macroscopic observables—fundamental diagrams, flow rates—without explicit supervision on the trajectories that produce those observables. They are elegant, interpretable, and brittle: every new geometric scenario requires manual re-calibration, and they cannot represent body-level behaviours (shoulder-turning, gait adaptation, anticipatory yielding) that govern dynamics in the high-density regimes that matter most for safety.

**The Proposal.** Replace or augment the hand-crafted locomotion layer with navigation policies learned via multi-agent reinforcement learning (MARL) in procedurally generated synthetic environments. Train agents to reach goals in complex, randomised geometries. Supervise them with reward functions derived from IAS-7’s decades of controlled experimental trajectory data. Validate learned policies by testing whether they reproduce known emergent phenomena (lane formation, faster-is-slower effect, zipper merging) in scenarios never seen during training.

**The Unique Advantage.** IAS-7 is the only lab that simultaneously owns (a) a validated open-source pedestrian simulator (JuPedSim), (b) a large corpus of high-precision trajectory data from controlled experiments (PeTrack, 3D-motion-in-crowds), (c) procedural synthetic environment generation expertise (replicAnt/UE5), and (d) deep RL policy distillation experience (Plum). No robotics lab has the empirical crowd data; no crowd dynamics lab has the RL and synthetic-data pipeline.

# 2. Motivation: Why Learned Policies?

## 2.1 Limitations of the Status Quo

Current operational models in JuPedSim and competing simulators share three structural weaknesses:

- **Scenario-specific calibration.** Model parameters calibrated on unidirectional corridor flow do not transfer to T-junctions, stairwells, or counterflow without re-tuning. Each new geometry is a new calibration exercise.
- **Macroscopic-only supervision.** Models are validated against aggregate quantities (flow, density, speed). Two models can match the same fundamental diagram while producing qualitatively different individual trajectories—meaning the micro-level dynamics are under-constrained.
- **No body representation.** Agents are discs. Shoulder rotation, asymmetric collision profiles, gait adaptation, and postural anticipation are absent. These matter precisely in the high-density bottleneck regimes where safety-critical decisions occur.

## 2.2 What RL Buys Us

A learned policy trained across diverse procedural environments and supervised with trajectory-level data addresses all three limitations:

- **Generalisation by construction.** If training environments are sufficiently randomised (varying corridor widths, obstacle placements, crowd densities, goal distributions), the policy must learn general-purpose navigation rather than geometry-specific heuristics.
- **Trajectory-level supervision.** Reward functions can incorporate trajectory-matching terms from real experimental data, penalising not just collisions but unrealistic motion patterns.
- **Richer agent representation.** The observation and action spaces can be extended to include body orientation, enabling emergent shoulder-turning without additional modelling assumptions.

# 3. Technical Architecture

## 3.1 Overview

The system comprises four modules: (A) a procedural environment generator, (B) a multi-agent RL training loop, (C) a reward module with real-data integration, and (D) a validation and benchmarking suite. These are designed as loosely coupled components connected through standardised interfaces (Gymnasium API for the environment, standard trajectory file formats for data exchange).

## 3.2 Environment Generator (Module A)

The training environment is a lightweight 2D physics simulation—not the full UE5 rendering pipeline. Using UE5 in the inner training loop would impose a 100–1000× computational penalty with no benefit to the policy learning, since the observation space is vectorial, not visual. The environment generator is the foundation of the entire project and must be designed for long-term extensibility: clean interfaces, modular components, and the ability to ingest both procedurally generated and externally defined geometries.

**Geometry representation**

All geometries are represented as 2D Shapely Polygons with holes, following the same convention used by JuPedSim. The walkable area is a single simple polygon (non-self-intersecting, non-zero area); obstacles are encoded as holes within that polygon. This ensures full compatibility with JuPedSim’s geometry format, meaning any geometry that runs in JuPedSim can be loaded directly into the training environment, and vice versa. Walls are implicit: the boundary of the walkable polygon and the boundaries of obstacle holes are the wall segments against which raycasts intersect and agents collide.

**Geometry sources: procedural generation and external import**

The environment generator supports two geometry sources, both producing the same Shapely polygon output:

**(1) Procedural generator.** A modular, parameterised geometry generator that composes walkable areas from a vocabulary of primitives. The generator operates at multiple complexity tiers, selected per episode to ensure the policy encounters a broad curriculum:

- **Tier 0 — Open fields.** Large convex polygons with no internal obstacles. Randomised shape (rectangular, circular, irregular convex hull). Useful for training basic goal-reaching and collision avoidance without geometric constraints.
- **Tier 1 — Corridors and bottlenecks.** Straight corridors with randomised width, length, and bottleneck constrictions (single or double). The core scenario class for pedestrian dynamics research. Parameters: corridor width (0.8–5.0m), bottleneck aperture (0.6–2.0m), bottleneck position along corridor.
- **Tier 2 — Branching corridors and junctions.** T-junctions, L-bends, crossroads, and Y-branches composed by joining corridor primitives. Parameters: number of branches (2–4), junction angle, branch widths, asymmetric branch lengths. This tier introduces route-choice complexity and counterflow at junctions.
- **Tier 3 — Rooms with furniture and obstacles.** Rectangular or irregular rooms with randomly placed internal obstacles (columns, furniture-like polygonal blocks, barriers). Exit doors of randomised width and position. This introduces cluttered navigation requiring local obstacle avoidance combined with global goal-seeking.
- **Tier 4 — Building floors.** Full floor plans generated by composing rooms and corridors into connected layouts: rooms linked by hallways, multiple exits, stairwell zones (modelled as special exit/entry regions connecting floors). This tier targets evacuation scenarios and requires agents to navigate through multiple decision points.
- **Tier 5 — Multi-floor evacuation.** Multiple Tier 4 floor plans connected via stairwell transition zones. Each floor is a separate 2D environment; stairwells act as portals transferring agents between floors with an associated traversal cost/delay. This is the most complex scenario class and may only become relevant in later project phases.

The generator is designed as a composition system: higher tiers are built by combining lower-tier primitives. Adding a new obstacle type or room shape means registering a new primitive, not rewriting the generator. All generated geometries are output as Shapely Polygons, ensuring a single downstream interface regardless of complexity tier.

**(2) External import.** IAS-7 maintains a set of well-characterised test geometries used in controlled experiments (bottleneck setups at various widths, unidirectional and bidirectional corridors, corner geometries). These serve a dual purpose: as validation benchmarks and as seed geometries for the procedural generator to learn from. The importer reads JuPedSim-compatible geometry definitions (Shapely polygons or the equivalent coordinate lists) and wraps them in the same environment interface as procedurally generated scenes. This ensures that training on procedural geometries and validating on real experimental geometries uses exactly the same code path.

**Solvability verification and the navigation mesh router**

Procedurally generated geometries are not guaranteed to be solvable: random obstacle placement can create disconnected regions, dead-end pockets, or spawn positions from which an agent’s goal is unreachable. Training on unsolvable configurations wastes compute and corrupts reward signals. This is addressed by integrating a navigation mesh router into the environment generator pipeline, inspired by JuPedSim’s own wayfinding system.

JuPedSim computes shortest paths by triangulating the walkable area and measuring distances between triangle centroids. Our environment generator adopts the same approach: upon geometry creation, the walkable polygon is triangulated (constrained Delaunay triangulation via Shewchuk's Triangle library), producing a navigation mesh that guarantees full coverage of the walkable area -- matching JuPedSim's CGAL CDT approach. An A* search on the triangle adjacency graph then verifies, for every (spawn, goal) pair, whether a path exists. The verification step operates in three modes:

- **Prune mode:** Agents whose (spawn, goal) pair has no valid path are removed from the episode. The geometry is kept, and the remaining agents train normally. This is the default for geometries where most agents are solvable but a few spawn in awkward pockets.
- **Regenerate mode:** If more than a configurable fraction of agents are unsolvable (default: >30%), the entire geometry is discarded and regenerated with a new random seed. This prevents degenerate episodes where most agents are idle.
- **Strict mode:** All agents must be solvable, or the geometry is regenerated. Used for validation episodes where partial populations would bias metrics.

While the procedural generation policies should aim to produce solvable geometries by construction (e.g., ensuring all rooms have at least one doorway connected to the corridor graph), the A* verifier acts as a safety net that guarantees no training time is wasted on impossible configurations.

**The router as an agent information source**

Beyond solvability checking, the navigation mesh router serves a second, more important role: it provides optional trajectory-planning information to agents. Once the A* shortest path from an agent’s current position to its goal is computed, two signals become available for inclusion in the observation space:

- **Next-waypoint direction (2D):** The direction vector from the agent’s current position to the next waypoint along the shortest path. This replaces the naïve “direct line to goal” that would point through walls in complex geometries. It effectively gives the agent a coarse route plan without dictating the fine-grained trajectory.
- **Path deviation scalar (1D):** The perpendicular distance from the agent’s current position to the planned path, normalised by corridor width. This tells the agent how far it has drifted from its intended route—useful information when crowd pressure pushes agents off course.

These signals are optional additions to the observation vector. Including them represents a “signposted” agent that has a map and knows the building layout; excluding them represents a “naïve” agent navigating purely by local perception. This is itself an ablation axis: comparing signposted vs. naïve agents quantifies how much global route knowledge affects crowd-level dynamics—a question directly relevant to evacuation scenarios where some occupants know the building and others do not.

**Crowd composition**

Agent count, body size distribution (drawn from anthropometric data), desired speed distribution, and goal assignment are sampled per episode. This ensures the policy encounters heterogeneous crowds during training.

**Physics**

Collision detection uses axis-aligned ellipses (not discs) to capture shoulder width vs. chest depth asymmetry. Contact forces follow a simplified Hertzian model. This is computationally cheap but provides the body-orientation signal that disc models lack.

**Collision model design: physics vs. learned behaviour.** The environment's collision model is deliberately limited to physical constraints -- things that are true about the world regardless of how pedestrians choose to behave. This means: (a) contact forces that prevent two bodies from occupying the same space, (b) mass-based inertia so that forces produce realistic accelerations rather than unit-mass impulses, and (c) accurate overlap detection using proper ellipse-boundary-to-boundary distance rather than a centre-in-ellipse proxy. Crucially, the environment does *not* include proximity repulsion forces between agents (the exponential repulsion used in JuPedSim's Social Force Model and Generalized Centrifugal Force Model). Those forces encode a behavioural assumption -- "humans maintain personal space" -- as a deterministic force law. Adding them as world physics would reduce the learned policy to fitting residual weights on top of a known hand-crafted model, undermining the core premise of learning navigation behaviour from scratch. Instead, proximity avoidance is incentivised through a tunable reward signal (see Section 3.4, Tier 1 proximity penalty), giving the policy gradient information to learn spacing behaviour without prescribing the mechanism. This separation keeps the environment physically honest while preserving the policy's freedom to discover its own collision avoidance strategies.

## 3.3 Observation and Action Spaces (Module B)

**Observation space (per agent)**

The observation vector comprises three components: ego state, social sensing, and environment sensing.

**Ego state (8D)**

Relative goal direction (2D, unit vector), own velocity (2D), scalar speed (1D), preferred speed (1D, raw m/s), own torso orientation angle (1D, relative to heading; always 0 in the egocentric frame), and own head orientation angle relative to torso (1D, clamped to ±90°). Preferred speed is exposed as a raw feature so the policy can observe the per-agent speed target it is regularised against — it was previously invisible to the network (added in the agent-dynamics refactor; see the progress log). The torso and head angles are separated because they are independently actuated: the head can rotate up to 90° left or right of the torso’s forward-facing direction before shoulder rotation is required. The raycast FOV follows the head, not the torso — an agent can scan its surroundings without reorienting its body. All values are in the agent’s local egocentric reference frame.

**Social sensing (K × 7D)**

For each of the K nearest neighbours: relative position (2D), relative velocity (2D), body orientation (1D), and approximate body dimensions (2D). With K=8 the social component is 56D.

**Environment sensing via raycasting (N-rays D)**

Without explicit obstacle and wall sensing, agents are effectively blind to the geometry they navigate. This is addressed via a tuneable raycast sensor: N rays are emitted from the agent’s position at evenly spaced angles within its field of vision (FOV), which is anchored to the head orientation, not the torso. Each ray returns the distance to the first intersection with a wall, obstacle, or other agent’s collision boundary, normalised to the maximum sensing range. If a ray hits nothing within range, it returns 1.0 (clear). This yields an N-dimensional vector of distance readings.

The key design parameters are: (a) N, the number of rays, which controls angular granularity (default: 16, spanning a 200° frontal FOV); (b) maximum sensing range (default: 5m); and (c) whether rays cover the full 360° or only the head’s forward-facing field of vision. The FOV-restricted variant is more biologically plausible and creates a perception–action trade-off: the agent can gather information by turning its head (cheap, up to ±90°) without reorienting its torso, but extreme lateral scanning requires shoulder rotation that may hinder forward locomotion. This decoupling means the policy must learn when to look versus when to physically reorient—a decision that real pedestrians make constantly in dense crowds. The 360° (omniscient) variant serves as an ablation control to quantify how much limited, head-anchored perception affects emergent collective behaviour.

Optionally, each ray can return a 2-channel signal rather than a scalar: (distance, hit-type), where hit-type encodes whether the intersection was with a wall (0), an obstacle (0.5), or another agent (1). This allows the policy to distinguish between static geometry and moving neighbours at the cost of doubling the environment-sensing dimensionality.

**Optional temporal and neighbour-memory features.** Beyond the instantaneous ego/social/ray channels, the observation builder can append several history-derived feature blocks, each independently toggleable and therefore an ablation axis in its own right:

- **Navmesh signals (3D):** next-waypoint direction (2D) + path-deviation scalar (1D) from the A\* router (see Module A).
- **Temporal memory (6D):** six scalars summarising the agent’s own trajectory history — displacement from spawn, cumulative path length, path efficiency (displacement ÷ path), elapsed episode fraction, and windowed displacement and goal-progress over the last W steps (default W=50). These let the policy sense whether it is making purposeful progress or is stuck/looping, which an instantaneous observation cannot express.
- **Neighbour velocity history (K × 2D):** for each of the K tracked neighbours, the change in that neighbour’s velocity over the last W_n steps (default 5), rotated into the ego frame — an acceleration proxy signalling whether a neighbour is speeding up or braking. Requires a persistent neighbour-ID tracker so a given neighbour keeps its slot across steps.
- **Neighbour trajectory features (K × 3D):** for each tracked neighbour, three scalars computed on that neighbour’s *own* temporal-memory state (path efficiency, windowed displacement, windowed goal-progress), letting the policy distinguish “we are all stuck together” from “I am the only one stuck.”

**Total observation dimensionality.** The base vector — ego 8D + social (K=8 × 7 = 56D) + N=16 single-channel rays — is 80D; with 2-channel rays it is 96D. The optional blocks add +3 (navmesh), +6 (temporal memory), +16 (neighbour velocity history at K=8), and +24 (neighbour trajectory at K=8), so a fully-instrumented agent reaches 80 + 3 + 6 + 16 + 24 = 129D. The Layer 1 v2 training configuration enables navmesh + temporal memory + neighbour velocity history, giving obs_dim = 105D. All variants are well within the range where MLP policies train reliably.

**The production line is narrower than the maximum, deliberately.** The shipped
policy (`example_model/policy_r0125.onnx`) runs at **89D**: ego + social + 16
single-channel rays + navmesh signals + temporal memory, with
`use_goal_direction = False`. Dropping the goal-direction channel is the
`nogoaldir` ablation promoted to the default: the agent navigates by the routed
next waypoint alone, which is exactly what a deployed model can be handed by
JuPedSim's router. Two further channels were narrowed to match deployment
rather than to save dimensions (`use_jupedsim_style_routing`, see the
2026-07-30 log entry): the waypoint is served router-style at JuPedSim's fixed
0.2 m portal inset instead of the funnel apex, and the `path_deviation` scalar
is pinned to 0.0 in training because deployment cannot compute it. The lesson
generalises: an observation channel that deployment cannot supply faithfully is
better trained *absent or degraded* than trained rich and then approximated.

**Action space**

Continuous, 4-dimensional: desired speed (scalar), desired heading change (scalar), desired torso orientation change (scalar), and desired head orientation change relative to torso (scalar, clamped to ±90°). The head and torso are independently actuated, mirroring human anatomy: the head can freely rotate up to 90° left or right of the torso’s forward-facing direction; beyond that range, shoulder rotation is required. Since the raycast FOV follows the head, this creates two distinct action channels: the head-turn is a low-cost information-gathering action (look around without changing body configuration), while the torso-turn is a higher-cost physical reorientation that alters the collision profile. The policy must learn to use both strategically—for example, scanning an approaching bottleneck with a head turn before committing to a shoulder rotation to slip through.

**Policy architecture**

Shared-parameter PPO with an actor-critic MLP. All agents share one policy network, conditioned on local observations. This is the standard MARL approach for homogeneous agent populations and scales well to hundreds of agents. Agent heterogeneity (body size, desired speed) enters through the observation, not through separate networks.

## 3.4 Reward Function (Module C)

The reward function is the core scientific contribution. It operates in three tiers:

- **Tier 1 — Sparse task rewards.** Goal-reaching bonus (+10), collision penalty (-1 per timestep in contact), proximity penalty (a graded linear ramp on centre-to-centre distance to the worst-offending neighbour: strongest at contact distance `r_i + r_j`, decaying to a near-zero value at an absolute `personal_space_radius` of 1 m, zero beyond; each agent pays the most-negative per-pair penalty across its active neighbours), timeout penalty (-5 if goal not reached within episode). These alone produce functional but potentially alien-looking navigation. The proximity penalty is the reward-side counterpart to the physics-side decision not to include proximity repulsion forces (see Section 3.2, Physics): the environment enforces only physical constraints (contact, inertia), while spacing behaviour is learned through this reward term. The graded-ramp form replaced an earlier binary "flat penalty inside a body-radius multiple" variant because the sharp threshold produced a discontinuous reward surface with zero gradient outside the critical zone.
- **Tier 2 — Smoothness priors.** Acceleration penalty (penalise jerk and angular acceleration), preferred-speed deviation penalty. These regularise the motion toward physically plausible trajectories without using any human data.
- **Tier 3 — Trajectory-matching from real data.** This is where IAS-7’s experimental data becomes an unfair advantage. Using trajectory datasets from PeTrack experiments (bottleneck flow, counterflow, unidirectional flow), compute distributional statistics: velocity autocorrelation functions, neighbour-distance distributions, angular change distributions. Define a style reward that penalises deviations from these distributions at the population level. This is not imitation learning on individual trajectories (which would overfit to specific experiments) but distributional matching—the agent should produce trajectories that are statistically indistinguishable from real pedestrians.

The ablation of these tiers is itself a key scientific output: it reveals which emergent crowd phenomena arise from any rational navigation strategy (Tiers 1–2 only) versus which require specifically human-like motion patterns (Tier 3).

## 3.5 Validation Suite (Module D)

Validation follows a held-out scenario protocol:

- **In-distribution validation.** Evaluate on procedurally generated scenarios similar to training. Measure goal-reaching rate, collision rate, and flow efficiency.
- **Benchmark scenarios.** Reproduce the exact geometries of IAS-7 controlled experiments (bottleneck b=0.8m, 1.0m, 1.2m; unidirectional corridor; bidirectional corridor). Compare simulated fundamental diagrams, flow rates, and trajectory statistics against experimental ground truth.
- **Zero-shot transfer.** Evaluate on scenario classes never seen during training (e.g., T-junction, stairwell, merging flow). The key test: does the policy produce lane formation in counterflow if it was never trained on counterflow? Does it reproduce the faster-is-slower effect at bottlenecks?
- **Head-to-head comparison.** Run the same scenarios in JuPedSim with its current hand-crafted models. Compare trajectory-level metrics (not just macroscopic flow) to quantify what the learned policy gains.

## 3.6 Software Architecture and JuPedSim Integration

The most consequential architectural decision is not which RL algorithm to use—it is where the boundary sits between code that is specific to training and code that must be shared with deployment inside JuPedSim. If the observation construction logic, the raycast engine, or the action interpretation code exists in two separate implementations, they will inevitably drift, and the transfer from training environment to JuPedSim will silently break. The architecture is therefore organised as four logical Python packages with a shared foundation (plus a fifth, crowdrl-torch -- a GPU-batched, training-only reimplementation of the environment step that mirrors crowdrl-core / crowdrl-env for throughput. It is not part of the deployment path and is held numerically in step with core by parity tests; see the progress log).

**Package 1: crowdrl-core (shared foundation)**

This package contains everything that must be identical between training and deployment. It has no dependency on any RL library, no dependency on Gymnasium, and no dependency on JuPedSim. It is a pure geometry/perception/action library. Its submodules are:

- **geometry:** Shapely polygon handling, constrained Delaunay triangulation (via Shewchuk's Triangle library for guaranteed full-coverage CDT), navigation mesh construction, wall-segment extraction from polygon boundaries.
- **navmesh:** A* shortest-path solver on the triangle adjacency graph. Computes next-waypoint direction and path-deviation scalar for any (position, goal) pair. Used both for solvability verification (during environment generation) and as an optional observation signal (during training and deployment).
- **sensing:** Raycast engine (N rays, configurable FOV, head-anchored). K-nearest-neighbour query for social sensing. Both operate on a generic WorldState dataclass containing agent positions, velocities, orientations, and wall segments—agnostic to whether this state comes from the training environment or from JuPedSim.
- **observation:** Assembles the full observation vector from ego state, social sensing, raycasts, and optional navmesh signals. Takes a WorldState and an agent index, returns a numpy array. This is the single function that must be identical between training and deployment—any discrepancy here means the policy sees a different world and produces wrong actions.
- **action:** Maps the 4D policy output (desired speed, heading change, torso orientation change, head orientation change) to kinematic quantities: a desired velocity vector, a new torso angle, and a new head angle. Enforces the ±90° head-to-torso constraint. During training, these feed back into the physics step. During deployment, the desired velocity feeds into JuPedSim’s simulation loop.
- **collision:** Elliptical agent collision detection, contact-force computation, and ray-vs-ellipse intersection for the raycast engine. Used by the training environment and -- because JuPedSim performs *no* collision resolution or overlap handling for the operational layer (its custom-model contract applies the model's returned position verbatim; see Section 3.6) -- also available to the deployment adapter whenever contact-force parity with training is required.

**Package 2: crowdrl-env (training environment)**

Depends on crowdrl-core and Gymnasium. Contains everything specific to the training loop that is not needed at deployment time: the procedural geometry generator (tiers 0–5), the solvability verifier, the crowd composition sampler, the reward modules (tiers 1–3), and the Gymnasium environment wrapper. The wrapper’s step() function calls crowdrl-core’s observation builder and action interpreter, adds physics integration and collision resolution from crowdrl-core’s collision module, and computes rewards. The reset() function calls the procedural generator, runs the solvability verifier, and initialises agent states. External geometry import (IAS-7 test geometries, JuPedSim geometry files) is handled by a loader that produces the same Shapely polygon consumed by the procedural path.

**Package 3: crowdrl-train (training infrastructure)**

Depends on crowdrl-env, PyTorch, and an MARL library (CleanRL or PettingZoo). Contains the MAPPO training loop, the curriculum manager (controls which geometry tiers and agent counts are used at each training stage), hyperparameter configuration, logging (Weights & Biases or TensorBoard), checkpointing, and policy export. The export step converts the trained PyTorch policy to ONNX format, which is the portable artefact consumed by Package 4. This package is never needed at deployment time.

**Package 4: crowdrl-jupedsim (deployment adapter)**

Depends on crowdrl-core, JuPedSim (2.0 line), and ONNX Runtime. This is the integration layer. Its central class is LearnedPolicyModel, a subclass of JuPedSim 2.0's `CustomOperationalModel` -- the pure-Python operational-model layer (see "The operational-model contract" below). JuPedSim invokes the model **per agent** via `compute_next_state(dt, ped, env_query)`, in a compute-then-apply pass over the whole population. For each agent, LearnedPolicyModel: (1) reads the agent's current state -- position, the final goal `ped.final_target` and the routed next waypoint `ped.next_target` that JuPedSim's routing has already set, plus its own per-agent custom state (velocity, torso/head angle, preferred speed, body dimensions, memory); (2) senses the world through the per-step `EnvironmentQuery` -- wall segments via `line_segments_in_range`, neighbours via `other_agents_in_range` (reading each neighbour's custom state for velocity/orientation/body-dims) -- and populates a crowdrl-core WorldState; (3) calls crowdrl-core's observation builder -- the exact same function that runs during training; (4) runs ONNX inference on that observation to produce the 4D action (per-agent by default; optionally a single batched inference over all agents is run once per step and cached, keyed on agent id, since the compute pass visits every agent before any is applied); (5) calls crowdrl-core's action interpreter and integrates the result into a **new position** (velocity filter + semi-implicit Euler, staying inside the walkable area, optionally applying contact forces for training parity), and returns a new immutable custom-state object carrying that position and the updated velocity/orientation/memory. JuPedSim applies the returned position verbatim -- it performs no integration, boundary projection, or collision resolution of its own.

Critically, this package does not depend on PyTorch or on crowdrl-env or crowdrl-train. The only artefact it needs from the training side is the exported .onnx file. A JuPedSim user installs crowdrl-jupedsim and crowdrl-core, loads a policy file, and uses it exactly like any other JuPedSim model. This also contains the benchmark runner: a harness that runs the same scenario with LearnedPolicyModel and with JuPedSim’s existing models (CollisionFreeSpeedModel, SocialForceModel, etc.) and compares trajectory-level and macroscopic metrics.

**The WorldState interface: the contract that holds everything together**

The key abstraction in crowdrl-core is the WorldState dataclass: a flat representation of everything the perception system needs to construct observations. It contains arrays of agent positions, velocities, torso orientations, head orientations, body dimensions, and goal positions, plus the walkable polygon and precomputed wall segments. During training, crowdrl-env populates WorldState from its internal physics state. During deployment, crowdrl-jupedsim populates WorldState from JuPedSim’s agent state API. The observation builder and sensing modules consume only WorldState—they never know which system produced it.

This is the architectural invariant that guarantees transfer: if WorldState is populated correctly from JuPedSim’s agent states, the observation vector will be numerically identical to what the policy saw during training for the same physical configuration. Any drift between the two population paths (training vs. deployment) is a bug that will produce subtle policy failures—so integration tests that compare observations from both paths on identical configurations are a first-class part of the test suite.

**The operational-model contract: what JuPedSim provides vs. what LearnedPolicyModel owns**

JuPedSim 2.0 exposes operational models as pure-Python subclasses of `CustomOperationalModel`, invoked per agent as `compute_next_state(dt, ped, geometry, neighborhood_search)` and required to return a *new* immutable per-agent state object (a frozen dataclass -- returning the same instance raises). The simulation advances in a strict compute-then-apply pass: it computes every agent's next state from the frozen current generation, then swaps the new generation in wholesale. The only field the framework reads back out of the returned state is `position`, which it applies verbatim.

This makes the boundary between JuPedSim and our code sharp -- and it is *not* where earlier drafts of this plan assumed it was. Verified against the 2.0 source (`OperationalDecisionSystem`, `Simulation::Iterate`, `GenericAgent`), the split is:

- **JuPedSim provides:** (a) the walkable-area geometry and wall segments, with containment / intersection queries (`InsideGeometry`, `IntersectsAny`, `get_walls_in_distance_to`); (b) the route waypoint -- the strategical (journey) and tactical (routing-engine) systems run *before* the operational step each iteration and write each agent's next `target`; (c) a neighborhood-search grid (`get_neighboring_agents(pos, radius)`); (d) agent lifecycle, the simulation clock, and trajectory serialisation.
- **JuPedSim does NOT provide** (contrary to earlier drafts): any velocity integration, any boundary clamping, and *any collision resolution or overlap handling whatsoever*. `GenericAgent` carries no velocity and no orientation field at all -- position is owned by the model state; velocity and orientation exist only if a model chooses to store them. Built-in models implement their own avoidance inside `compute_next_state`; there is no separate collision system to inherit. A returned position outside the walkable area is applied as-is and crashes the next iteration.
- **LearnedPolicyModel therefore owns the entire state transition:** read the current state, `ped.final_target` and `ped.next_target`; sense walls and neighbours; build the WorldState and observation (shared crowdrl-core code); run ONNX inference; interpret the 4D action and integrate it into a new position (velocity filter + semi-implicit Euler); keep the agent inside the walkable area; optionally apply contact forces for training parity; and return the new state. Everything an agent can do, how it perceives the world, and how a policy output becomes the next state lives in this one class.

**The orientation gap dissolves.** Earlier drafts weighed two strategies for JuPedSim not tracking torso / head angle -- Strategy A (the adapter keeps a private side-channel dict) and Strategy B (submit a C++ PR extending JuPedSim's agent struct). The custom-model layer supersedes both: per-agent state is an arbitrary immutable Python object, so torso angle, head angle, preferred speed, full shoulder / chest body dimensions, and the temporal / neighbour-memory buffers simply become fields on the frozen state dataclass. A neighbour's state is readable during the callback, so these attributes are available for social sensing too -- the deployment observation can be reconstructed *faithfully*, not approximated. No side-channel bookkeeping (A) and no JuPedSim core change (B) are required: Strategy B is retired, and Strategy A is subsumed by the sanctioned custom state.

**Availability, and how the dependency is actually handled.** `CustomOperationalModel` is a JuPedSim 2.0 feature -- a deliberate breaking change from 1.x, which removes the `ModelType` enum and the `XModelAgentParameters` classes. It lives on upstream `main` and is not in a tagged release; the published PyPI line is still 1.x and lacks the layer entirely. The resolution is *not* a version pin: `crowdrl-jupedsim` declares **no** `jupedsim` dependency at all, deliberately, because declaring `jupedsim>=1.0` made `uv sync` install a 1.x wheel that then silently shadowed a local 2.0 source build. JuPedSim 2.0 is supplied out-of-band from a source build, put on `sys.path` via a `.pth` file in the venv's `site-packages` (build recipe in the 2026-07-30 log entry). Consequences worth stating plainly: site-packages must never contain a competing `jupedsim` install, and every JuPedSim-dependent test is written to `pytest.importorskip` so CI stays green without a build. The trained policy is an external `.onnx` file loaded by path -- but as of 2026-07-30 the repo *does* ship one committed baseline artefact (`example_model/policy_r0125.onnx`, ~1.2 MB, self-describing) so the adapter, its tests and the example notebook are runnable out of the box; the adapter itself still bundles no policy.

**Incremental build path**

The packages are built in dependency order, with each stage producing a usable artefact:

- **Step 1: crowdrl-core.** Build geometry, navmesh, sensing, observation, and action modules. Write unit tests with hand-constructed WorldState instances. This is testable in complete isolation before any RL code exists. Deliverable: a library that, given a polygon and a set of agent states, produces observation vectors and interprets actions.
- **Step 2: crowdrl-env.** Build the Gymnasium wrapper, procedural generator (start with Tiers 0–2), solvability verifier, and Tier 1–2 reward modules. Verify with a random-policy baseline (agents take random actions; confirm observations look correct, rewards are distributed as expected, episodes terminate properly). Deliverable: a Gymnasium environment that produces episodes with procedural or imported geometries.
- **Step 3: crowdrl-train.** Implement the MAPPO training loop, curriculum manager, and policy export. Train initial policies on Tier 0–2 environments. Deliverable: trained .onnx policy files and training logs.
- **Step 4: crowdrl-jupedsim.** Build the LearnedPolicyModel adapter as a `CustomOperationalModel` subclass (JuPedSim 2.0) and the benchmark runner. Write the obs-parity test that asserts the observation built by the adapter is numerically identical to the training-env observation for an identical physical configuration -- this is the transfer guardrail, not an afterthought. Deliverable: a JuPedSim-compatible operational model that loads a .onnx policy by path and runs alongside CollisionFreeSpeedModel, SocialForceModel, etc. **Status: done as of 2026-07-30** for the adapter, the parity guardrail (which found eight real divergence channels -- see `plan/lockstep_parity_analysis.md`) and the e2e scenarios. The cross-model benchmark runner (LearnedPolicyModel vs. CollisionFreeSpeedModel / SocialForceModel on the same scenario) is the one piece of Step 4 still outstanding, and is now the natural next deliverable since the deployment path is trustworthy.

This ordering means that papers can be written after Step 3 (the RL results stand alone), and the JuPedSim integration (Step 4) is a separate deliverable that can proceed in parallel with later training experiments (Tier 3 reward, higher-tier geometries, ablation studies).

# 4. Milestones and Timeline

The project is structured in four phases over approximately 18 months. Each phase produces a usable deliverable, so the project has value even if later phases are delayed.

| Phase | Milestone | Description | Timeline |
|-------|-----------|-------------|----------|
| I | M1: Environment prototype | Modular 2D environment generator with Shapely-based geometry (Tiers 0–2), elliptical agents, Gymnasium API, JuPedSim geometry import, A* solvability verifier (prune/regenerate/strict modes), and navigation mesh router providing optional waypoint signals. Verified with random-policy baseline. | Months 1–3 |
| I | M2: Baseline RL agent | Single-agent PPO navigating to goals in static environments. Sanity check that the training loop works. | Months 2–3 |
| II | M3: MARL training | Multi-agent PPO with parameter sharing, 20–100 agents. Tier 1+2 rewards. Demonstrate collision-free goal-reaching in randomised geometries. | Months 3–6 |
| II | M4: Emergent phenomena | Document emergent behaviours (lane formation, shoulder-turning) from Tier 1+2 rewards alone. First internal report. | Months 5–7 |
| III | M5: Trajectory data integration | Process PeTrack datasets into distributional statistics. Implement Tier 3 style reward. Retrain and compare against Tier 1+2-only policies. | Months 6–9 |
| III | M6: Ablation study | Systematic ablation of reward tiers. Quantify which emergent phenomena require human-data supervision. Core paper contribution. | Months 8–11 |
| IV | M7: Benchmark validation | Reproduce IAS-7 benchmark scenarios. Quantitative comparison against JuPedSim hand-crafted models on trajectory-level and macroscopic metrics. | Months 10–14 |
| IV | M8: Zero-shot transfer | Evaluate on unseen scenario classes. Demonstrate generalisation. Second paper or extended first paper. | Months 12–16 |
| IV | M9: JuPedSim integration | Package learned policy as a JuPedSim locomotion module. Open-source release with documentation and example notebooks. | Months 14–18 — **substantially delivered early** (2026-07-30): adapter, self-describing artefact, e2e scenarios and example notebook 10 are done and validated; the cross-model benchmark runner and the public release remain. |

# 5. Key Design Decisions (Summary)

The following table summarises the critical architectural choices and their rationale. Each of these is a point where your own judgement should override this document—they are recommendations, not prescriptions.

| Decision | Recommended Choice | Rationale |
|----------|-------------------|-----------|
| Training environment | Lightweight 2D physics (not UE5) | UE5 in the training loop is 100–1000× slower. Vector observations don’t benefit from rendering. Architect for a module swap to egocentric vision later. |
| Observation space | Egocentric vector: ego state (8D) + K=8 neighbours (56D) + N=16 head-anchored raycasts (16D) = 80D base; optional navmesh (+3), temporal-memory (+6), neighbour-velocity-history (+16) and neighbour-trajectory (+24) blocks extend it to 105D (current Layer 1 v2) up to 129D | Three-component design: ego state, social sensing (K nearest neighbours), and environment sensing (FOV-restricted raycasts for wall/obstacle distances). Raycasts prevent geometry-blindness. FOV restriction couples perception to torso orientation, making the orientation action meaningful. N and FOV are tuneable for ablation. |
| Action space | 4D continuous: speed, heading, torso angle, head angle (relative to torso, ±90°) | Head and torso are independently actuated, matching human anatomy (head rotates ±90° relative to torso before shoulders must follow). Decoupling separates information-gathering (head turn) from physical reorientation (shoulder turn). Ablation: collapse to 3D (fused head-torso) or 2D (speed + heading only). |
| RL algorithm | PPO with parameter sharing (MAPPO) | Proven stable for cooperative/competitive multi-agent continuous control. Shared parameters scale to large agent counts. |
| Agent count (training) | 20–100 per episode, randomised | Enough for crowd phenomena. >200 agents creates training instability without curriculum learning. |
| Reward architecture | 3-tier: sparse + smoothness + distributional style | Ablation across tiers is the scientific contribution. Tiers are additive and independently testable. |
| Trajectory data source | PeTrack controlled experiments | High precision, known geometry, controlled conditions. Real-event data (EURO 2024) is noisier and harder to control for. |
| Validation protocol | Train on procedural, test on IAS-7 benchmarks + zero-shot | Zero-shot transfer to known experiments is the killer result. Comparison with JuPedSim hand-crafted models grounds the contribution. |

# 6. Risks and Mitigations

- **Risk: MARL training instability at high agent counts.** Mitigation: Start with 20 agents, use curriculum learning to increase. Population-based training (PBT) for hyperparameter search. If scaling beyond ~100 agents fails, the results at moderate scale are still publishable—most real experimental scenarios involve 20–100 participants anyway.
- **Risk: Distributional style reward is noisy or ill-defined.** Mitigation: Start with the simplest distributional metric (velocity autocorrelation) and add complexity only if needed. If Tier 3 fails entirely, the Tier 1+2 ablation is still a valid paper.
- **Risk: Learned policies produce brittle or exploitable behaviour.** Mitigation: Procedural environment randomisation is the main defence. Additionally, evaluate on adversarial perturbations (sudden obstacle insertion, density spikes) as a robustness check.
- **Risk: Computational cost exceeds available resources.** Mitigation: PPO is cheap compared to model-based RL or offline RL. Estimate ~48–96 GPU-hours for full training on a single A100. IAS-7 has HPC access at FZJ. If compute is truly constrained, reduce episode length and agent count—the architecture remains the same.
- **Risk: Reviewers say “this is just robotics MARL applied to pedestrians.” **Mitigation: The contribution is not the algorithm (PPO is standard). The contributions are: (a) the trajectory-distributional reward from real experimental data, (b) the ablation revealing which crowd phenomena are “any rational agent” vs. “specifically human,” and (c) zero-shot transfer validated against decades of controlled crowd experiments. No robotics lab can do (a) or (c).

# 7. Publication and Funding Strategy

## 7.1 Target Publications

- **Paper 1 (Months 10–14):** Core methodological contribution. Target: Transportation Research Part C, or Autonomous Agents and Multi-Agent Systems. Framing: learned pedestrian navigation policies, trained on procedural environments with trajectory-distributional supervision, reproduce known crowd phenomena and generalise zero-shot to unseen geometries.
- **Paper 2 (Months 14–18):** Application/integration paper. Target: Collective Dynamics (IAS-7’s own journal—low barrier, high visibility in the community), or Simulation Modelling Practice and Theory. Framing: JuPedSim with a learned locomotion module, open-source release, benchmark comparisons.
- **Workshop/conference:** Pedestrian and Evacuation Dynamics (PED) conference, or Traffic and Granular Flow (TGF). Preliminary results from Phase II (emergent phenomena from Tier 1+2 rewards) are suitable for a conference contribution by Month 8.

## 7.2 Funding Angles

- **DFG Sachbeihilfe (individual grant):** Natural fit. Framing: fundamental research on emergent collective dynamics from learned individual policies. The ablation study is the kind of clean scientific question DFG likes.
- **BMBF Zivile Sicherheitsforschung:** If framed toward real-time crowd management applications (the learned policy runs faster than real-time, enabling predictive crowd management). Combine with the digital twin angle from Idea 5.
- **Helmholtz AI:** The ML-for-simulation angle fits the Helmholtz AI call format. FZJ is a Helmholtz centre, which gives a structural advantage.

# 8. Open Questions for Discussion

These are genuine unknowns that the plan does not resolve. They require your input and likely some pilot experiments:

- **How much does body orientation matter in practice? **The 4D action space (with independent head and torso control) is motivated by two hypotheses: that shoulder-turning matters at bottlenecks, and that decoupled gaze direction matters for anticipatory navigation. But added action dimensions increase training difficulty. A staged ablation should settle this: 2D (speed + heading only) vs. 3D (fused head-torso orientation) vs. 4D (decoupled head and torso). If the 3D variant already reproduces the key phenomena, head-torso decoupling may not be worth the cost. Conversely, if 4D agents develop qualitatively different scanning-then-committing strategies in dense scenarios, that is itself a novel finding about the role of active perception in crowd dynamics.
- **Which PeTrack datasets are suitable for distributional reward computation? **Not all experimental datasets are equally useful. You need high-density scenarios where interesting dynamics occur, with sufficiently long trajectories to compute meaningful statistics. A data audit is needed before committing to Tier 3.
- **Should the long-term goal be to replace JuPedSim’s locomotion layer or to augment it? **Replacement is cleaner but harder to get adopted. Augmentation (e.g., a learned “correction” applied on top of the existing model) might be more publishable and more immediately useful to the JuPedSim user community.
- **When (if ever) should egocentric vision enter the observation space? **This is the bridge to the PyroCrowd idea (fire + smoke + visibility). If the vector observation space is sufficient for pedestrian dynamics in clear air, visual observations become justified only when visibility conditions vary (smoke, darkness, signage). That’s a follow-up project, not Phase I.

*End of draft. Awaiting your thoughts and corrections.*

---

# Implementation Progress Log

## 2026-03-26 — Step 1 complete, Step 2 substantially complete

### crowdrl-core (Step 1): COMPLETE

All 7 submodules fully implemented and tested (119 unit tests, 100% pass):

| Module | LOC | Status |
|--------|-----|--------|
| `world_state.py` | 117 | WorldState + NavMesh dataclasses |
| `geometry.py` | 258 | Polygon handling, triangulation, navmesh construction |
| `navmesh.py` | 490 | A*, funnel algorithm, `is_passable()` with agent-radius portal-width check |
| `sensing.py` | 215 | Raycast engine (head-anchored, configurable FOV) + KNN social query |
| `observation.py` | 174 | Single observation builder (training + deployment) |
| `action.py` | 171 | 4D action interpreter (speed, heading, torso, head) |
| `collision.py` | 306 | Elliptical collision detection + contact forces |

Key design decisions implemented:
- WorldState is the sole interface between perception and simulation
- `is_passable()` combines A* reachability with per-agent portal-width clearance checks (not just topological `is_reachable()`)
- Observation builder is a single function shared between training and deployment
- Agent clearance radius = max(shoulder_width, chest_depth), consistent across navmesh signals and solvability verification

### crowdrl-env (Step 2): SUBSTANTIALLY COMPLETE

New modules implemented (86 tests, 100% pass):

| Module | LOC | Status |
|--------|-----|--------|
| `geometry_generator.py` | 437 | Tiers 0-2 (open fields, corridors/bottlenecks, L-bends/T-junctions/crossroads) |
| `spawner.py` | ~160 | Crowd composition sampler (anthropometric body dims, speed distributions, separation-enforced placement) |
| `solvability.py` | ~80 | Prune/regenerate/strict modes with clearance-aware passability via `is_passable()` |
| `reward.py` | ~150 | Tier 1 (sparse: goal bonus, collision penalty, timeout, progress shaping) + Tier 2 (smoothness: jerk, angular accel, preferred-speed deviation) |
| `crowd_env.py` | ~280 | Full Gymnasium wrapper: `reset()`, `step()`, `observation_space`, `action_space` |
| `visualiser.py` | 399 | Geometry, navmesh, agent, raycast visualisation |

**CrowdEnv capabilities:**
- Procedural geometry generation with optional multi-tier randomisation per episode
- Agent spawning with heterogeneous body dimensions + preferred speeds
- Solvability verification with agent-radius-aware portal-width checks
- Semi-implicit Euler physics with elliptical collision contact forces
- Tier 1+2 reward computation with mutable temporal state
- Full Gymnasium API (batched obs/actions for MAPPO parameter sharing)

### What remains before training (Step 3):

**crowdrl-env remaining items:**
- [ ] Geometry Tiers 3-5 (rooms, building floors, multi-floor) — NOT blocking training
- [ ] External geometry importer (IAS-7 test geometries) — NOT blocking training
- [ ] Tier 3 reward (distributional style matching from PeTrack data) — Phase III

**crowdrl-train (Step 3) — complete:**
- [x] MAPPO training loop (PPO with parameter sharing)
- [x] Policy network architecture (separate Actor-Critic MLPs)
- [x] Observation + reward normalization (Welford's running stats)
- [x] Rollout buffer with per-agent GAE (variable agent counts)
- [x] Curriculum manager (success-rate-driven phase advancement)
- [x] ONNX export pipeline (actor + frozen normalizer)
- [x] Training logging (TensorBoard)
- [x] Checkpointing (save/load full training state)
- [x] CLI entry point (`crowdrl-train`)

**crowdrl-jupedsim (Step 4) — not started:**
- [ ] LearnedPolicyModel adapter
- [ ] ONNX runtime wrapper
- [ ] Orientation state tracking (Strategy A)

### Test suite: 266 tests total

| Package | Tests | Pass rate |
|---------|-------|-----------|
| crowdrl-core | 119 | 100% |
| crowdrl-env | 86 | 100% |
| crowdrl-train | 61 | 100% |
| **Total** | **266** | **100%** |

### Example notebooks

| # | Title | Status |
|---|-------|--------|
| 01 | Geometry and Navmesh | Complete |
| 02 | Sensing and Observations | Complete |
| 03 | Mini Simulation | Complete |
| 04 | Gymnasium Environment | Complete — demos CrowdEnv reset/step, multi-tier, reward analysis |
| 05 | MAPPO Training | New — networks, buffer, GAE, PPO update, curriculum, mini training run |

## 2026-03-26 — Step 3 complete: crowdrl-train package

Implemented the full MAPPO training pipeline (9 modules, 61 tests, 100% pass rate).

### New modules

| Module | Purpose |
|--------|---------|
| `config.py` | Frozen dataclasses for all hyperparameters (NetworkConfig, PPOConfig, CurriculumConfig, TrainConfig) with JSON serialisation |
| `networks.py` | Separate Actor (diagonal Gaussian) + Critic MLPs with numpy-based orthogonal init (avoids Windows MKL crash) |
| `normalizer.py` | RunningNormalizer (Welford's algorithm) + RewardNormalizer (divide by running std of returns) |
| `buffer.py` | RolloutBuffer storing variable-agent-count timesteps, per-agent GAE with mid-episode termination, FlatBatch for PPO |
| `mappo.py` | MAPPOUpdater: clipped surrogate loss, MSE value loss, separate actor/critic optimizers, KL early stopping, linear LR decay |
| `curriculum.py` | CurriculumManager: rolling goal rate tracking, phase advancement, env config generation |
| `logger.py` | TensorBoard + console logging backends |
| `export.py` | ONNX export (actor + frozen normalizer) with verification against PyTorch |
| `train.py` | Main training loop + checkpointing + CLI entry point |

### Key design decisions (literature-grounded)

- **Full-batch PPO** (n_minibatches=1): Yu et al. (2022) — "Avoid splitting data into mini-batches" for MARL
- **No value loss clipping**: Andrychowicz et al. (2021) — "hurts regardless of threshold"
- **Separate actor/critic**: Andrychowicz et al. (2021) — outperformed shared trunk
- **tanh activation**: Andrychowicz et al. (2021) — beat ReLU on 4/5 continuous control envs
- **Gradient clip 10.0**: Yu et al. (2022) — more permissive for multi-agent
- **Welford's running normalisation**: Andrychowicz et al. (2021) — "Always use observation normalisation"

### Windows compatibility

- `torch.nn.init.orthogonal_` crashes via LAPACK/MKL access violation → replaced with numpy QR-based init
- PyTorch backward pass crashes inside pytest process → PPO update tests run as subprocesses

### What remains before deployment (Step 4):

- crowdrl-jupedsim package (ONNX runtime adapter for JuPedSim)
- Large-scale training runs (10M+ timesteps)
- Tier 3 reward (distributional style matching)
- Geometry Tiers 3-5

## 2026-03-27 — Vectorized environments and training scaling

### Parallelisation: SubprocVecEnv + RolloutCollector

Added subprocess-parallel environment execution to `crowdrl-train` for higher training throughput.

| Module | Purpose |
|--------|---------|
| `vec_env.py` | `SubprocVecEnv` — N CrowdEnv instances in separate processes (`spawn` context for Windows), communicating via `multiprocessing.Pipe`. Main process sends commands (reset/step/reconfigure/close), workers execute env logic in parallel. |
| `rollout_collector.py` | `RolloutCollector` — collects transitions from all workers with central GPU inference (one batched forward pass per step). Uses **per-env buffers** to handle variable `n_agents` across envs. Computes GAE per buffer, then merges into a single `FlatBatch`. |
| `config.py` | Added `VecEnvConfig(n_envs, n_steps_per_collect)` to `TrainConfig` |

### Key architectural decisions

- **Per-env buffers**: Each env gets its own `RolloutBuffer` because different envs have different `n_agents`. Interleaving into a shared buffer causes shape mismatches in GAE computation.
- **Per-episode bootstrap for GAE**: `buffer.compute_gae()` extended to accept `list[NDArray]` — incomplete episodes get V(s_last) from the critic, completed episodes get zeros.
- **Central GPU inference**: All observations concatenated into one forward pass, then split back per env. GPU cost is roughly constant regardless of N_ENVS.
- **`train.py` dispatching**: `train()` routes to `_train_single()` (N=1) or `_train_vec()` (N>1).

### Geometry sizing lesson

Initial training runs with default geometry sizes (fields up to 25m, corridors up to 30m, max_steps=1000) produced only 3 completed episodes in 21M agent-steps — agents moved brownianly in vast empty spaces. Compact geometries (8–15m fields, 8–18m corridors, max_steps=200) matching notebook 05's successful runs are essential. Crowding comes from agent-to-area ratio, not absolute agent count.

### Reward extension

Added `inverse_distance_weight` to `RewardConfig` — continuous proximity-to-goal signal (`weight / (distance + 1.0)`). Disabled by default (weight=0.0), backward compatible.

### Updated test suite: 276 tests total

| Package | Tests | Pass rate |
|---------|-------|-----------|
| crowdrl-core | 119 | 100% |
| crowdrl-env | 86 | 100% |
| crowdrl-train | 71 | 100% |
| **Total** | **276** | **100%** |

### Example notebooks

| # | Title | Status |
|---|-------|--------|
| 06 | Full Training (Vectorized) | Rewritten — 32 workers, 5000 rollouts, compact geometries, curriculum, live progress output |

### What remains before deployment (Step 4):

- crowdrl-jupedsim package (ONNX runtime adapter for JuPedSim)
- Large-scale training runs with vectorized envs
- Tier 3 reward (distributional style matching)
- Geometry Tiers 3-5

## 2026-03-28 — crowdrl-torch: GPU-vectorised environments

### New package: crowdrl-torch

A full GPU-vectorised re-implementation of the environment step, replacing `SubprocVecEnv` with batched tensor operations on a single GPU. All N_ENVS environments are processed in one call with shapes `(E, N, ...)`. No subprocess pipes, no IPC overhead.

| Module | Purpose |
|--------|---------|
| `types.py` | `EnvConfig` (frozen dataclass from `CrowdEnvConfig`) + `TorchWorldState` (NamedTuple of all state tensors) |
| `action.py` | Vectorised action interpretation (speed, heading, torso, head) |
| `collision.py` | Pairwise elliptical collision detection + Hertzian contact forces |
| `walls.py` | Wall distance computation + boundary enforcement |
| `sensing.py` | Batched raycasting (head-anchored FOV) + KNN social query |
| `observation.py` | Full observation builder (ego + social + rays), mirrors `crowdrl-core` |
| `reward.py` | Vectorised reward computation (all Tier 1+2 terms) |
| `step.py` | `batched_step()` — the complete step function, `torch.compile`-compatible |
| `batched_env.py` | `BatchedTorchEnv` — manages N_ENVS on GPU with async CPU reset thread pool |
| `episode_factory.py` | CPU-side episode generation (geometry, spawning, solvability, navmesh) |
| `geometry_repr.py` | NumPy padding for CPU→GPU transfer |
| `normalizer.py` | Welford running stats for obs/reward normalisation (GPU tensors) |
| `torch_collector.py` | Rollout collection + GAE computation on GPU |

**Key capabilities:**
- `torch.compile(mode="reduce-overhead")` for kernel fusion + CUDA graph capture
- Async CPU episode generation via `ThreadPoolExecutor` (no step-blocking resets)
- >100k steps/sec on single laptop GPU (target met)
- Windows support via `triton-windows` package with MAX_PATH workaround
- `test_equivalence.py` validates numerical parity with CPU `crowdrl-core`

**Deviations from original plan:**
- The plan described a 5th package (`crowdrl-torch`) not in the original 4-package architecture. It sits alongside `crowdrl-train` rather than replacing it — `crowdrl-train` handles the PPO/curriculum logic, `crowdrl-torch` handles the GPU environment. The `torch_collector.py` bridges the two.
- `SubprocVecEnv` (added 2026-03-27) was superseded within days by the GPU-vectorised approach. It remains in `crowdrl-train` as a CPU fallback but is no longer the primary training path.

### Reward extensions

Added to `crowdrl-env` and ported to `crowdrl-torch`:
- `wall_proximity_penalty` — smooth gradient penalty before hard wall contact (configurable threshold as multiple of agent body radius)
- `action_rate_penalty` — penalises frame-to-frame action change, targeting policy output directly (more direct than jerk/angular-acceleration smoothness terms)
- `inverse_distance_weight` — continuous proximity-to-goal signal

### CI fixes

- Pinned `torch==2.6.0+cu126` with GPU index for training, CPU-only override in CI
- `triton-windows` restricted to `sys_platform == "win32"` to avoid breaking Linux CI
- CI uses `--no-sync` for pytest to preserve CPU torch override

### Updated test suite: 288 tests total

| Package | Tests | Pass rate |
|---------|-------|-----------|
| crowdrl-core | 119 | 100% |
| crowdrl-env | 86 | 100% |
| crowdrl-train | 71 | 100% |
| crowdrl-torch | 12 | 100% |
| **Total** | **288** | **100%** |

### Example notebooks

| # | Title | Status |
|---|-------|--------|
| 06 | Full Training | Rewritten for GPU-vectorised `crowdrl-torch`, async resets, ONNX export, MP4 video rendering |

## 2026-03-30 — GPU-native navmesh waypoint signals

### Problem

Agents in Tier 1-2 geometries (corridors, T-junctions, bottlenecks) only see a straight-line goal direction that points through walls. They need shortest-path guidance (navmesh waypoints) in their observation vector, but CPU navmesh code cannot be called per-step without destroying GPU throughput.

### Solution: pre-compute + pure tensor lookup

Sparse waypoints (typically 1-8 turning points) are pre-computed once at episode reset via A* + funnel on CPU, then stored as padded GPU tensors. Each step uses a pure-tensor lookup with zero CPU involvement:

1. **Pre-compute at reset (CPU, amortised):** `shortest_path()` per agent → waypoint sequence + cumulative remaining path lengths. ~1ms added to episode reset.
2. **Per-step GPU lookup:** `torch.gather` for current + next waypoints, distance-weighted blending (closer waypoint = less influence for smooth gradient), ego-frame rotation, path deviation from pre-computed cumulative lengths. Monotonic cursor advancement via `torch.where`.
3. **Observation signal:** 3D — `[waypoint_dir_ego_x, waypoint_dir_ego_y, path_deviation]`, concatenated to produce 82D obs (up from 79D).

All operations are pure tensor ops, fully `torch.compile`-compatible. Computational cost: ~10 element-wise ops on (E, N) tensors — comparable to one layer of contact force computation.

### Files changed

| File | Change |
|------|--------|
| `crowdrl-core/observation.py` | Added `navmesh_max_waypoints` to `ObsConfig` |
| `crowdrl-torch/types.py` | Added waypoint fields to `EnvConfig` + `TorchWorldState` |
| `crowdrl-torch/episode_factory.py` | Pre-compute waypoints per agent at reset |
| `crowdrl-torch/geometry_repr.py` | Pad waypoint arrays to fixed shape |
| `crowdrl-torch/batched_env.py` | Thread waypoints through full reset/step pipeline |
| `crowdrl-torch/observation.py` | `compute_navmesh_signals()` — pure tensor ops |
| `crowdrl-torch/step.py` | Waypoint cursor advancement + wiring |
| `examples/06_full_training.ipynb` | Enable `use_navmesh=True`, infer `obs_dim`, document all reward terms |

### Deviation from plan

The original plan (Section 3.2) described navmesh waypoint signals as computed per-step from the A* router. The implementation pre-computes sparse waypoints at episode reset and uses GPU tensor lookups per step. This was necessary to maintain >100k steps/sec throughput. The observation signal content (next-waypoint direction + path deviation) matches the plan, but the computation path is fundamentally different. See `plan/gpu_navmesh_waypoints.md` for the full implementation design.

The plan's "path deviation scalar" was described as "perpendicular distance from the agent's current position to the planned path, normalised by corridor width." The implementation uses `(remaining_path_length / euclidean_distance_to_goal) - 1` instead — a ratio that captures how much longer the actual path is vs. a straight line. This is more informative (tells the agent how "windy" its remaining path is) and doesn't require computing point-to-polyline distance, which would be expensive on GPU.

### Current status summary

**Milestone M1 (Environment prototype): COMPLETE**
- All geometry tiers 0-2 implemented and tested
- Solvability verification with 3 modes
- Navmesh router providing waypoint signals
- GPU-vectorised environment with >100k steps/sec

**Milestone M2 (Baseline RL agent): COMPLETE**
- Single-agent PPO verified during M3 development

**Milestone M3 (MARL training): SUBSTANTIALLY COMPLETE**
- MAPPO with parameter sharing, 20-100 agents
- Tier 1+2 rewards including wall proximity, action rate, inverse distance
- GPU-vectorised training with `torch.compile` + CUDA graphs
- Curriculum manager with success-rate-driven phase advancement
- ONNX export pipeline verified
- **Remaining:** Large-scale training runs (10M+ timesteps) to validate convergence and emergent behaviour

**Milestones M4-M9: NOT STARTED**

### What remains

**Immediate (training validation):**
- [ ] Large-scale training runs with GPU env + navmesh waypoints
- [ ] Verify agents learn to follow waypoints in Tier 1-2 geometries
- [ ] Document emergent behaviours (M4)

**Medium-term:**
- [ ] Geometry Tiers 3-5
- [ ] External geometry importer (IAS-7 test geometries)
- [ ] Tier 3 reward (distributional style matching from PeTrack data)

**Deployment:**
- [ ] crowdrl-jupedsim package (ONNX runtime adapter)
- [ ] Integration tests (obs parity between training and deployment)

## 2026-03-30 — Tier 3 geometry + existence penalty + curriculum expansion

### Tier 3 procedural geometry (crowdrl-env)

Implemented two new geometry tiers completing the "rooms with obstacles" layer from the project plan (Section 3.2, Tier 3).

**Tier 3a — Rooms with obstacles:**
- Starts from a random Tier 0–2 base room
- Places random obstacles via rejection sampling: rectangular furniture blocks (random rotation) and circular columns, buffered 0.3m from walls
- Obstacle coverage configurable (default 5–20% of floor area)
- Cuts 1–2 door openings in bounding-box walls (configurable width 0.8–2.0m)
- Optional shared-goal mode (configurable probability, default 40%): all agents target one evacuation exit
- Metadata tracks: base tier/shape, obstacle count, door count, shared-goal flag

**Tier 3b — Composed multi-room layouts:**
- Generates 2–3 rooms from Tier 0–2 primitives (configurable range)
- Arranges rooms side-by-side with connecting corridor links (1.5–4.0m gap, 1.5–3.0m wide)
- Places obstacles inside merged walkable area
- Cuts 1–2 exterior evacuation doors
- Spawn regions: translated room interiors; goal regions: evacuation doors + connector corridors
- Metadata tracks: room count/shapes, connector count, evacuation door count

**Statistics (50 samples each):**
- Tier 3a: 76.6 ± 67.8 m² area, 14.4 ± 9.5 holes, 248.9 ± 159.6 navmesh triangles
- Tier 3b: 64.0 ± 35.6 m² area, 9.2 ± 6.1 holes, 166.5 ± 101.6 navmesh triangles
- Solvability (30 geometries × 20 pairs): 3a mean 81.7%, 3b mean 77.5% — validates need for prune/regenerate modes

**New config fields on `GeometryConfig`:**
- `obstacle_coverage_range`, `obstacle_min_size`, `obstacle_max_size`, `column_radius_range`
- `door_width_range`, `shared_goal_probability`, `n_rooms_range`

### Existence penalty (crowdrl-env + crowdrl-torch)

Added `existence_penalty` to `RewardConfig` (default -0.01): a small negative reward every step an agent is active. Pressures agents to reach goals quickly rather than drifting. Applied to both CPU (`crowdrl-env/reward.py`) and GPU (`crowdrl-torch/reward.py`) reward paths, threaded through `EnvConfig`.

### Curriculum expansion (crowdrl-train)

Extended `DEFAULT_CURRICULUM_PHASES` from 4 to 6 phases:

| Phase | Name | Tiers | Agents | Threshold |
|-------|------|-------|--------|-----------|
| 1 | easy | 0 | 5–10 | 0.6 |
| 2 | medium | 0–1 | 8–20 | 0.5 |
| 3 | hard | 1–2 | 20–50 | 0.5 |
| 4 | rooms | 2, 3a | 15–40 | 0.5 |
| 5 | complex | 3a, 3b | 20–60 | 0.4 |
| 6 | full | 0–3b | 20–100 | 0.0 |

The "rooms" phase introduces obstacles in familiar corridor shapes; "complex" adds multi-room composition. The terminal "full" phase now covers all implemented tiers.

### Example notebook 07 — Complex Geometry

New notebook `examples/07_complex_geometry.ipynb` demonstrating:
- Tier 3a generation with obstacle and door visualisation
- Tier 3a forced shared-goal (evacuation) mode
- Tier 3b multi-room composition (2 and 3 rooms)
- Navmesh construction on cluttered polygons with holes
- A* + funnel shortest paths through obstacles and corridor links
- Solvability statistics across 30 geometries per tier
- Geometry summary statistics (area, holes, wall segments, triangles)

### Files changed

| File | Change |
|------|--------|
| `crowdrl-env/geometry_generator.py` | +395 LOC: Tier 3a/3b generators, obstacle placement, door cutting |
| `crowdrl-env/tests/test_geometry_generator.py` | +117 LOC: TestTier3a (5 tests) + TestTier3b (7 tests) |
| `crowdrl-env/reward.py` | Added `existence_penalty` field + computation |
| `crowdrl-torch/reward.py` | Ported existence penalty to GPU |
| `crowdrl-torch/types.py` | Added `existence_penalty` to EnvConfig |
| `crowdrl-train/config.py` | 2 new curriculum phases, expanded "full" phase |
| `crowdrl-train/tests/test_config.py` | Updated phase count + relaxed monotonicity assertion |
| `examples/07_complex_geometry.ipynb` | New notebook for Tier 3 demonstration |
| `examples/06_full_training.ipynb` | Minor cleanup (cell IDs, output clearing) |

### Updated "what remains"

**Immediate (training validation):**
- [x] Large-scale training runs with GPU env + navmesh waypoints
- [x] Verify agents learn to follow waypoints in Tier 1-2 geometries
- [x] Verify curriculum progresses through Tier 3a/3b phases
- [ ] Document emergent behaviours (M4)

**Medium-term:**
- [ ] Geometry Tiers 4-5 (building floors, multi-floor evacuation)
- [ ] External geometry importer (IAS-7 test geometries)
- [ ] Tier 3 reward (distributional style matching from PeTrack data)

**Deployment:**
- [ ] crowdrl-jupedsim package (ONNX runtime adapter)
- [ ] Integration tests (obs parity between training and deployment)

## 2026-03-31 -- Solvability overhaul + successful 200-rollout training

### Problem

Agents in complex geometries (Tier 3a/3b) were getting stuck in narrow gaps between close obstacles, even when a passable route existed. The solvability checker falsely approved paths where agents could not physically fit.

Three root causes:

1. **Portal-width checks used diagonal edges.** Delaunay triangulation creates portal edges (shared triangle edges) that can span diagonally through narrow gaps. A 1m gap between two obstacles can produce a ~1.41m diagonal portal, falsely passing the width check.
2. **No rotation safety margin.** `is_passable` checked `portal_width >= 2 * agent_radius` exactly, with no margin for the agent's widest orientation (shoulder width).
3. **Geometry generator created impassable openings.** Obstacle placement used 0.3m wall margin regardless of agent size. Tier 3b connector corridors used bounding-box overlap to size openings, but convex room walls can be angled, making the actual opening narrower than intended.

### Solution: 3-stage solvability check + geometry enforcement

**`is_passable()` now has 3 stages** (in `crowdrl-core/navmesh.py`):

1. **A* reachability** -- topological path exists on the triangle graph.
2. **Portal-width filter** -- fast rejection: every portal >= `2 * effective_radius`.
3. **Minkowski erosion** -- the walkable polygon is eroded inward by `effective_radius` using Shapely `buffer(-r)`. Start and goal must remain connected in the eroded polygon. This is geometrically exact: if a disc of radius r can traverse between two points in a polygon, they are connected in the polygon eroded by r.

`effective_radius = agent_radius * clearance_factor` (default 1.2 = 20% margin).

An earlier approach (buffering the funnel path and checking if it stayed inside the polygon) was too sensitive to Shapely floating-point slivers at obstacle corners, producing false negatives that dropped Tier 3A survival from 88% to 6%. The Minkowski erosion approach is both geometrically exact and robust.

**Geometry generator changes** (in `crowdrl-env/geometry_generator.py`):

- New `GeometryConfig.min_passage_width` (default 0.7m) enforced on bottleneck apertures, door openings (Tier 3a and 3b), and corridor connector widths.
- `_place_obstacles()` uses `min_passage_width` as both wall margin and inter-obstacle gap, preventing adjacent obstacles from creating impassable corridors.
- Tier 3b connector placement validates effective opening width against actual room geometry (not just bounding box) at each junction. Retries up to 10 positions, with a fallback that extends deeply into both rooms.

**NavMesh stores source polygon** (`NavMesh.polygon` field) for the erosion check.

### Per-tier agent survival rates (30 agents, 10 episodes, clearance_factor=1.2)

| Tier | Before | After |
|------|--------|-------|
| Tier 0 | ~83% | 100% |
| Tier 1 | ~93% | 100% |
| Tier 2 | 100% | 100% |
| Tier 3A | ~6% | 88% |
| Tier 3B | ~19% | 78% |

### Training validation

Full 200-rollout training run (`examples/06_full_training.ipynb`) with all tiers, navmesh waypoints, and the new solvability checks completed successfully with high goal rates and good movement patterns.

### Files changed

| File | Change |
|------|--------|
| `crowdrl-core/world_state.py` | Added `NavMesh.polygon` field |
| `crowdrl-core/geometry.py` | Store polygon in `build_navmesh()` |
| `crowdrl-core/navmesh.py` | 3-stage `is_passable()` with `clearance_factor` + `_validate_path_clearance()` |
| `crowdrl-core/tests/test_navmesh.py` | +4 test classes (polygon storage, clearance validation, geometric passability) |
| `crowdrl-env/solvability.py` | `clearance_factor` parameter on `verify_solvability()` |
| `crowdrl-env/crowd_env.py` | `solvability_clearance_factor` config + propagate `min_passage_width` |
| `crowdrl-env/geometry_generator.py` | `min_passage_width`, obstacle gap enforcement, Tier 3b connector validation |
| `crowdrl-env/tests/test_solvability.py` | +3 test classes (clearance factor, geometric clearance, close obstacles) |
| `crowdrl-torch/episode_factory.py` | Propagate `clearance_factor` + `min_passage_width` |
| `docs/agent_pipeline.md` | Updated reward table, action limits, obs dim, section 8 |
| `docs/environment_mechanics.md` | Updated solvability description with 3-stage check |

### Updated test suite: 315 tests total

| Package | Tests | Pass rate |
|---------|-------|-----------|
| crowdrl-core | 130 | 100% |
| crowdrl-env | 102 | 100% |
| crowdrl-train | 71 | 100% |
| crowdrl-torch | 12 | 100% |
| **Total** | **315** | **100%** |

### Updated milestone status

**Milestone M3 (MARL training): COMPLETE**
- MAPPO with parameter sharing, 20-100 agents
- Tier 1+2 rewards including wall proximity, action rate, existence penalty
- GPU-vectorised training with navmesh waypoint signals
- Curriculum manager progressing through all tiers (0 through 3b)
- 200-rollout training run validated with high goal rates
- Solvability checker ensures agents are never given impossible paths

**Milestone M4 (Emergent phenomena): IN PROGRESS**
- High goal rates and good movement patterns observed
- Remaining: systematic documentation of emergent behaviours

### Updated "what remains"

**Immediate:**
- [ ] Document emergent behaviours from 200-rollout run (M4)
- [ ] Run longer training (500+ rollouts) to push curriculum to later phases
- [ ] Quantify emergent phenomena: lane formation, shoulder turning, gap exploitation

**Medium-term:**
- [ ] Geometry Tiers 4-5 (building floors, multi-floor evacuation)
- [ ] External geometry importer (IAS-7 test geometries)
- [ ] Tier 3 reward (distributional style matching from PeTrack data)

**Deployment:**
- [ ] crowdrl-jupedsim package (ONNX runtime adapter)
- [ ] Integration tests (obs parity between training and deployment)

## 2026-04-11 -- Graded proximity ramp, single-node DDP, rollout carry-over

### Graded agent-proximity ramp (crowdrl-env + crowdrl-torch)

Replaced the previous binary agent-proximity penalty (flat `-0.005` inside a
`2.0 * agent_radius` threshold) with a **graded linear ramp on the
centre-to-centre distance to the worst-offending neighbour**. The new form
has three config fields on `RewardConfig` / `EnvConfig`:

| Field | Default | Role |
|-------|---------|------|
| `agent_proximity_penalty_near` | -0.005 | Per-pair penalty at contact (`r_i + r_j`) |
| `agent_proximity_penalty_far`  | -0.0001 | Per-pair penalty right at `personal_space_radius` |
| `personal_space_radius`        | 1.0 m   | Absolute centre-to-centre cutoff (not body-relative) |

Each ordered pair `(i, j)` contributes
`pair_penalty = (1 - t) * near + t * far` with
`t = clip((d_ij - contact) / (personal_space_radius - contact), 0, 1)`.
Each agent pays `min over j of pair_penalty_ij` -- the **worst** per-neighbour
penalty in its personal-space zone. This tracks the closest encroachment
rather than summing over density, and provides a continuous gradient all
the way from 1 m out to contact (the old binary form was flat inside and
had no gradient outside, so the policy only learned the threshold, not the
distance).

The per-pair distance computation now lives inside `compute_rewards` itself
(both the NumPy path in `crowdrl-env/reward.py` and the vectorised torch
path in `crowdrl-torch/reward.py`). The `agent_distances` parameter was
removed from both signatures; callers no longer compute a scalar
"min distance per agent" outside the reward function.

### Reward weight re-tuning

While touching the reward module, the other weights were aligned with the
current training runs rather than the stale values in the docs:

| Field | Old default | New default |
|-------|-------------|-------------|
| `wall_proximity_penalty` | -0.3 | -0.1 |
| `progress_weight` | 0.1 | 1.0 |
| `jerk_penalty_weight` | -0.01 | -1e-6 |
| `angular_accel_penalty_weight` | -0.005 | -1e-4 |
| `speed_deviation_weight` | -0.03 | -1e-3 |

The smoothness weights are deliberately tiny so they regularise without
overriding collision/progress in congested scenarios. `progress_weight` at
1.0 (rather than 0.1) makes the potential-based shaping term comparable in
magnitude to the goal bonus over an episode, which visibly accelerates
learning in the early curriculum phases.

`docs/environment_mechanics.md` and `docs/agent_pipeline.md` reward tables,
budget breakdown, and Section B.1 / 8.3 have been rewritten to match.

### Single-node multi-GPU training via DD-PPO

Added a DD-PPO-style distributed training path (Wijmans et al. 2019)
without moving to the standard PyTorch `DistributedDataParallel` wrapper.
Rationale: the actor is invoked via `evaluate_actions()` (re-scoring the
rollout actions for the PPO log-ratio), not through `forward()`, so DDP's
autograd hooks would not fire on the rollout minibatches. A manual
gradient all-reduce is both explicit and matches the pattern CleanRL uses
for PPO.

**New module:** `packages/crowdrl-torch/src/crowdrl_torch/distributed.py`

| Helper | Purpose |
|--------|---------|
| `init_distributed(backend="nccl")` / `cleanup_distributed()` | Process group lifecycle driven by `torchrun` env vars |
| `is_distributed` / `is_main_rank` / `get_rank` / `get_world_size` | Rank queries with single-process fall-backs |
| `allreduce_gradients(model)` | Flatten all `.grad` tensors into one contiguous buffer, single `all_reduce(SUM)`, divide by world size, unflatten |
| `TorchRunningNormalizer.sync_across_ranks()` | Parallel Welford merge of obs-normaliser mean / var / count |
| `sync_reward_normalizer(rn, device)` | Same parallel Welford merge for the reward normalizer's return-variance tracker, plus averaged running return |
| `gather_episode_stats(local)` | `all_gather_object` of per-rank episode dicts to rank 0 |
| `broadcast_curriculum_state(mgr)` | Broadcast curriculum phase after rank 0 advancement decisions |
| `distributed_seed(base)` / `seed_everything(seed)` | Per-rank seed helpers (env diversity, deterministic model init) |

**Integration into `crowdrl-train/mappo.py`:**

- `MAPPOUpdater` learned a `distributed: bool | None` flag (auto-detected
  via `torch.distributed.is_initialized()` by default).
- Every `actor_loss.backward()` and `critic_loss.backward()` is followed by
  an inlined flat all-reduce before the optimiser step. Effective batch
  becomes `local_batch * world_size`; no learning-rate scaling (matching
  CleanRL convention).
- **KL early-stopping fix:** under DDP each rank's minibatch produces a
  different local approximate KL. If ranks early-stop independently, one
  rank exits the epoch loop while another is still issuing gradient
  all-reduces, deadlocking NCCL on mismatched collectives. The updater
  now `all_reduce`s the KL tensor inside the loop and uses the **global**
  KL for the early-stop decision, so all ranks agree. Two regression
  tests live in `packages/crowdrl-train/tests/test_mappo.py` (subprocess
  gloo groups with a spy on `dist.all_reduce`).

**New frozen dataclass:** `DDPConfig(backend="nccl")` in
`crowdrl_train.config`, exported from `crowdrl_train/__init__.py`.

**Launch pattern:**
`torchrun --standalone --nproc_per_node=N train_mappo.py`.

Full design rationale, synchronisation table, and a usage snippet for the
training loop are in `plan/ddp_single_node.md`.

### Rollout collector: cross-collect episode carry-over

Both `RolloutCollector` (CPU subproc path, `crowdrl-train`) and
`TorchRolloutCollector` (GPU path, `crowdrl-torch`) previously called
`env.reset_all()` at the start of **every** `collect()`. Any in-flight
episode was discarded, wasting agent-steps and biasing recorded episode
statistics toward short episodes that happen to fit inside one rollout.

Both collectors now persist episode tracking state across `collect()`
calls:

- Initial reset is **lazy** (first `collect()` only); subsequent calls
  reuse the existing env and episode trackers.
- An episode that straddles a collect boundary counts the full episode
  reward across both rollouts. Episode statistics now reflect real
  episode lengths, not rollout-length caps.
- A new "trailing incomplete segment" is bootstrapped from the critic at
  the segment's **post-step** observation (`s_T`), not the already-normalised
  `s_{T-1}` that used to sit in the buffer. `s_T` is normalised exactly once
  at the end of `collect()` via a new `_final_obs_norm` field.
- The torch collector slices each segment over the full `max_agents` axis
  and relies on the per-step `active_mask` to select real agents, rather
  than inferring `n_agents` from the first step (which broke for
  carry-over segments where terminated agents can be scattered across the
  row).
- `BatchedTorchEnv` grew a per-env `env_tiers: list[str]` field that
  records each env's current geometry tier (e.g. `"TIER_3B"`) on every
  reset, and the collector attaches it as `ep_dict["geometry_tier"]` so
  per-tier episode statistics are available without a new collective.
- `BatchedTorchEnv._async_step()` rebuilds observations for envs whose
  async reset completed mid-collect, so callers see the new episode's
  initial obs instead of stale zeros.

### Export wrapper device isolation fix

`PolicyForExport` in `crowdrl_train/export.py` used to hold references to
the actor's `feature_net` and `action_mean` modules. A downstream
`wrapper.cpu()` (used in the ONNX export pipeline) would silently move the
original actor's parameters to CPU, breaking any subsequent GPU operation
on the training model. `PolicyForExport.__init__` now **deep-copies**
both submodules so the wrapper and the live training actor are fully
independent. Regression tests in
`packages/crowdrl-train/tests/test_export.py` verify that `export_onnx`
leaves the source actor on its original device with and without a
normalizer.

### New tests and CI

- `packages/crowdrl-torch/tests/test_distributed.py` -- single-process no-op
  checks for all distributed helpers (safe without `torchrun`).
- `packages/crowdrl-train/tests/test_export.py` -- export wrapper device
  isolation + numerical parity with the source actor.
- `packages/crowdrl-train/tests/test_mappo.py` -- two new subprocess-based
  DDP regression tests that spin up a single-rank `gloo` group and verify
  the KL collective fires and the global KL is the one used for the
  early-stop check.
- `pyproject.toml`: `testpaths` extended to `["packages", "tests"]` so a
  repo-root `tests/` directory can host cross-package integration tests.

### Files changed

| File | Change |
|------|--------|
| `crowdrl-env/reward.py` | Graded proximity ramp (three new config fields), removed `agent_distances` arg |
| `crowdrl-env/crowd_env.py` | Drop `compute_min_agent_distances` call (pair distances now live in `compute_rewards`) |
| `crowdrl-torch/reward.py` | Vectorised graded proximity ramp; drop `agent_distances` arg |
| `crowdrl-torch/step.py` | Drop pairwise distance computation; `compute_rewards` now handles it |
| `crowdrl-torch/types.py` | `EnvConfig`: new `agent_proximity_penalty_near/_far/personal_space_radius` fields |
| `crowdrl-torch/distributed.py` | **New** -- DDP lifecycle, grad sync, normalizer sync, curriculum broadcast |
| `crowdrl-torch/normalizer.py` | `TorchRunningNormalizer.sync_across_ranks()` -- parallel Welford merge |
| `crowdrl-torch/batched_env.py` | Per-env `env_tiers`, post-reset obs rebuild, tier name threaded through async resets |
| `crowdrl-torch/torch_collector.py` | Cross-collect episode carry-over, full-N slicing, post-step bootstrap, per-tier stats |
| `crowdrl-torch/__init__.py` | Re-export DDP helpers |
| `crowdrl-train/config.py` | **New** `DDPConfig` frozen dataclass |
| `crowdrl-train/__init__.py` | Re-export `DDPConfig` |
| `crowdrl-train/mappo.py` | `distributed` flag, global KL early stop, inline grad all-reduce |
| `crowdrl-train/rollout_collector.py` | Cross-collect episode carry-over, post-step bootstrap |
| `crowdrl-train/export.py` | Deep-copy actor modules inside `PolicyForExport` |
| `crowdrl-train/tests/test_mappo.py` | +2 DDP regression tests (subprocess gloo) |
| `crowdrl-train/tests/test_export.py` | **New** -- export wrapper device isolation |
| `crowdrl-torch/tests/test_distributed.py` | **New** -- DDP no-op tests |
| `pyproject.toml` | `testpaths = ["packages", "tests"]` |
| `plan/ddp_single_node.md` | **New** -- DDP design doc |
| `docs/environment_mechanics.md` | Reward values, B.1 / B.6 / B.7 rewrite, appendix table |
| `docs/agent_pipeline.md` | Reward tables, Section 8 expansion (proximity ramp, carry-over, DDP, export fix) |

### What remains

Most of the list from 2026-03-31 still stands. The current run in
`examples/06_full_training.ipynb` needs to be re-executed against the new
reward surface before its results become the canonical baseline for M4
emergent-phenomena analysis.

**Immediate:**
- [ ] Re-run `examples/06_full_training.ipynb` with the graded proximity
      ramp and re-tuned reward weights
- [ ] Document emergent behaviours once the new baseline is in
- [ ] Shake-down a real 2+ GPU DDP run on the HPC cluster (single-node)

**Medium-term:**
- [ ] Geometry Tiers 4-5 (building floors, multi-floor evacuation)
- [ ] External geometry importer (IAS-7 test geometries)
- [ ] Tier 3 reward (distributional style matching from PeTrack data)

**Deployment:**
- [ ] crowdrl-jupedsim package (ONNX runtime adapter)
- [ ] Integration tests (obs parity between training and deployment)

## 2026-05-26 -- Agent-dynamics refactor: Layer 1 biomechanical recalibration + Layer 1 v2

Work on branch `agent_dynamics_refactor` (design of record:
`plan/agent_dynamics_refactor.md`). This entry consolidates ~6 weeks of
training iteration into the plan and motivates the version bump to v7 -- the
observation space in Section 3.3 grew (see "Observation-space expansion"
below). A standalone end-of-day snapshot lives in
`docs/2026-05-26_layer1_v2_handover.md`.

### Diagnosis: kinematic targets were unconstrained ("ice-skating")

The pre-refactor action interpreter treated the policy output as a near-instant
kinematic target. Per-step rate caps were pi/12 for heading and torso
(1500 deg/s) and pi/3 for head (6000 deg/s), and the velocity filter
`desired_velocity_weight` was effectively off (0.8 = tau ~12 ms at dt=0.01s).
Agents could pivot and reverse velocity almost instantaneously -- the
"ice-skating" regime: gliding, omnidirectional, visibly non-human, with no
incentive to brake before obstacles. `examples/09_reward_landscape.ipynb`
confirmed the pathology on paper: under the old weights a "plow through"
trajectory scored *higher* than "brake before the wall".

### Layer 1 -- biomechanical recalibration (now the dataclass defaults)

The action space is unchanged in form (4D), but its *limits* were re-grounded
in the human walking envelope and are now the `ActionConfig` / `CrowdEnvConfig`
defaults:

| Quantity | Was | Now (default) | Basis |
|----------|-----|---------------|-------|
| Heading rate cap | pi/12 (1500 deg/s) | 0.020 rad/step = 115 deg/s | Hicheur 2007 walking yaw |
| Torso rate cap | pi/12 (1500 deg/s) | 0.010 rad/step = 57 deg/s | hip rotation limit |
| Head rate cap | pi/3 (6000 deg/s) | 0.030 rad/step = 172 deg/s | head scans fastest |
| Desired speed | 0..1.5 m/s symmetric | [-0.5, +2.0] m/s asymmetric | forward >> backward gait |
| Velocity filter `desired_velocity_weight` | 0.8 (no filter) | 0.05 (tau ~200 ms) | first-order inertia |
| `preferred_speed` in ego obs | absent | exposed (raw m/s) | policy can see its speed target |

Smoothness reward weights were rebalanced so they regularise without dominating
task reward; the current defaults are `jerk` -1e-5, `angular_accel` -1e-2,
`speed_deviation` -0.005, and `action_rate` -0.01 (now enabled), with
`progress_weight` raised to 1.0 so potential-based shaping is comparable to the
goal bonus over an episode.

### Observation-space expansion (the v7 obs delta)

Three optional, independently-toggleable feature blocks were added to the
single shared observation builder (design: `plan/neighbor_memory_extension.md`,
`plan/agent_memory_research.md`):

- **Temporal memory (+6D):** the agent's own trajectory summary (displacement
  from spawn, cumulative path, path efficiency, elapsed fraction, windowed
  displacement and goal-progress). Lets the policy sense "am I progressing or
  stuck/looping" -- which an instantaneous obs cannot express.
- **Neighbour velocity history (+K*2 = 16D):** per tracked neighbour, its
  velocity change over the last W_n steps (acceleration proxy). Needs a
  persistent neighbour-ID tracker so a neighbour keeps its slot across steps.
- **Neighbour trajectory features (+K*3 = 24D):** per tracked neighbour, its
  own path-efficiency / windowed-progress scalars -- "is everyone stuck, or
  just me." Implemented but not enabled in the current run.

Base obs is now 80D (ego 8 + social 56 + rays 16). The Layer 1 v2 run enables
navmesh + temporal memory + neighbour velocity history = **105D**; full
instrumentation is 129D. See Section 3.3.

### Layer 1 v2 -- empirical escape from ice-skating (and an open assumption)

The recalibrated caps alone were too tight for PPO's early-curriculum
exploration budget: the policy could not discover avoidance fast enough and the
curriculum stalled. Three changes together broke the equilibrium (commits
`63db6e7`, `843c993`, `367b85f`):

1. Reward retune validated against notebook 09 -- the brake trajectory now
   scores +12 over plow-through (was -1.3): `speed_deviation` -0.1 -> -0.005,
   `jerk` -1e-4 -> -1e-5.
2. `preferred_speed` exposed in the ego observation (obs 104 -> 105).
3. **Action caps loosened back above the biomechanical envelope** -- ~4/2/4
   deg/step in `exp_layer1_seed43.yaml` and 10/10/10 in
   `exp_layer1_seed43_retune.yaml`, versus the 1.15/0.57/1.72 deg/step
   defaults.

With (3) the curriculum cleared all six phases by ~rollout 40 and stabilised at
0.80-0.86 goal rate in the terminal `full` phase. **Caveat / open assumption:**
the loosened caps sit *above* the comfortable-walking band. This is an
empirical fix awaiting a literature justification, and is exactly the
assumption the next iteration aims to remove (see "Direction" below).

### Remaining failure mode + collision-suppression retune

Training videos at ~rollout 150 show **collision-dominated success**: with
`goal_bonus +50` against `collision_penalty -1/step`, plowing through a
neighbour (a ~10-step, -10 contact) is cheaper than yielding. The retune config
`exp_layer1_seed43_retune.yaml` targets this -- `collision_penalty` -1 -> -5,
`agent_proximity_penalty_near` -0.01 -> -0.05, `desired_velocity_weight`
0.05 -> 0.2 (so the agent can brake within one avoidance window). It was
warm-started (`--init_from`) from the v2 run's rollout 180 and reached rollout
100 of a planned 500 before pausing; **it has not yet been evaluated.**

### Tooling added on the branch

- Per-rollout checkpointing (`checkpoint_interval`) and `--init_from`
  warm-start (loads weights + normalizers, resets curriculum/optimizer/history),
  recorded under `_launch.init_from` in each run's `config_resolved.yaml`.
- Polygon-free GPU-batched eval rendering
  (`BatchedTorchEnv(disable_auto_reset=True)` + `render_episode_video`).

### Milestone status (updated)

- **M3 (MARL training):** pipeline COMPLETE and clears the full curriculum, but
  the M3 *quality bar* -- collision-free goal-reaching -- is NOT yet met
  (collision-dominated success). Infrastructurally done, behaviourally in
  progress.
- **M4 (Emergent phenomena):** BLOCKED on the above. Lane formation /
  shoulder-turning cannot be cleanly documented while agents interpenetrate.
  The agent-dynamics refactor is the campaign to unblock M4.
- **Layer 2** (second-order action semantics -- accelerations + explicit
  yaw-rate state; `plan/agent_dynamics_refactor.md` Section 4): DESIGNED, NOT
  STARTED. Deferrable if the retune yields clean counterflow; otherwise it is
  the path to physically-grounded turning.
- **M5-M9** (Tier 3 reward, ablation, benchmark, zero-shot, JuPedSim): NOT
  STARTED.

### Direction for the next iteration

Priority: **return the action model to its biomechanical envelope and reduce
the number of standing assumptions**, rather than buying goal-rate with
ever-looser caps.

- [ ] Re-establish biomechanically-grounded action caps (115/57/172 deg/s) as
      the trained envelope; if PPO cannot discover avoidance inside it, fix
      that through curriculum / exploration / reward shaping -- not by raising
      the caps above human capability.
- [ ] Justify or retire each added assumption: the loosened caps, and the
      temporal / neighbour-memory observation blocks (ablate -- does each block
      earn its dimensionality?).
- [ ] Evaluate the retune rollout-100 checkpoint before launching further runs.
- [ ] Decide Layer 2 (second-order action semantics) vs. continued Layer 1
      tuning based on whether head-on counterflow resolves cleanly.

**Medium-term / Deployment:** unchanged from the 2026-04-11 entry (Tiers 4-5,
IAS-7 importer, Tier 3 reward; crowdrl-jupedsim adapter + obs-parity tests).

## 2026-07-20 -- Progress reconciliation + JuPedSim 2.0 integration surface

This entry reconciles the plan with ~8 weeks of undocumented work (the log had
stalled at 2026-05-26 while development continued to the 2026-06-19 master
merges) and records a source-level study of JuPedSim 2.0's new pure-Python
operational-model layer, which reshapes the Section 3.6 integration design
(Section 3.6 has been rewritten accordingly). Work begins on branch
`feat/jupedsim-integration`.

### Undocumented work since 2026-05-26 (consolidated from git history)

| Area | What landed | Location |
|------|-------------|----------|
| Trajectory export | Kinora / pedpy-compatible HDF5 trajectory exporter (PR #5) -- first bridge into the JuPedSim analysis ecosystem and a down-payment on Module D | `crowdrl-env/kinora_export.py`, `test_kinora_export.py` |
| Validation | Fixed-scenario behavioural scorecard (goal / collision / wall / freeze-deadlock / stuck / speed / path-efficiency) + periodic in-training scorecard | `crowdrl-train/scorecard.py`, `scripts/eval_scorecard.py`, `scripts/diagnose_stuck_agents.py`, `scripts/analyze_run.py` |
| Reward | Impact-speed (velocity-weighted) collision and proximity penalties; `collision_penalty_cap` | `crowdrl-env/reward.py`, `crowdrl-torch/reward.py` |
| Stability | Large NaN-hardening campaign from big runs: running-count caps on obs / reward normalisers, `nan_to_num` on the obs builder, `log_std` clamp, NaN-grad skip, gated tripwires | normalizer / mappo / obs paths |
| Action model | Speed-turn coupling wired through core / env / torch action interpreters | action modules |
| Training | tanh policy + truncation-aware GAE stabilisation; minimum spawn-goal distance in the spawner | `crowdrl-train`, `crowdrl-env/spawner.py` |
| Ablation | `nogoaldir` line -- single next-waypoint nav signal + path-aware reward / stuck, agents navigating on waypoints alone; a stable big-rooms / density run produced 36 ONNX checkpoints | `configs/`, `results_exp_nogoaldir_*` |
| HPC | JURECA setup (`sc_uv_crowdrl/`, sbatch scripts, `fix_cuda_libs.py`) | repo root |

Net effect: Module D (validation) and the trajectory-data pipeline are further
along than the pre-2026-05-26 plan implies, but spread across crowdrl-train /
crowdrl-env / scripts rather than framed as a validation package.

### Repository reality vs. the four-package architecture

- **Five packages, not four.** `crowdrl-torch` (GPU-batched env) is the real
  training path and drives the root `train_mappo.py` (a ~72 KB experiment / CLI
  driver -- the actual entry point, not `crowdrl-train/train.py`). Design
  principle #1 ("one observation builder used everywhere") is upheld by
  `crowdrl-torch/tests/test_equivalence.py` parity, not by a literally single
  implementation.
- **crowdrl-jupedsim is still a bare stub** -- one docstring, no
  `LearnedPolicyModel`, no ONNX runtime loop, no tests. This is the work this
  branch begins.
- Trained `.onnx` policies and `results_*` runs exist locally but are export
  artefacts, not committed baselines; the intended baseline weights live
  off-repo.

### JuPedSim 2.0 operational-model contract (source study)

Integration targets `jupedsim.models.custom_model.CustomOperationalModel`
(upstream `main`; JuPedSim 2.0, a breaking change from 1.x, not yet tagged).
Verified against the 2.0 C++ / bindings source:

- The model is a Python subclass invoked **per agent** as
  `compute_next_state(dt, ped, geometry, neighborhood_search)`, returning a
  *new* frozen per-agent state. Stepping is compute-then-apply; the framework
  reads only `position` from the returned state and applies it verbatim
  (`OperationalDecisionSystem::Run` -> `agents.swap`).
- **JuPedSim provides:** walkable-area geometry + wall queries via the per-step
  `EnvironmentQuery` (`inside_geometry`, `intersects_any`,
  `line_segments_in_range`, `no_wall_between`); the agent's final goal
  `ped.final_target` AND its routed next waypoint `ped.next_target` (both set
  by the strategical + tactical systems before the operational step; renamed
  and exposed to Python by upstream PR #1626 -- note upstream flags
  `next_target`-as-location as temporary, likely becoming an orientation);
  neighbour queries (`other_agents_in_range`, self-excluding); lifecycle /
  clock / serialisation.
- **JuPedSim does NOT provide** velocity integration, boundary clamping, or *any*
  collision / overlap resolution. `GenericAgent` has no velocity or orientation
  field; position is owned by the model state.
- **Consequence:** `LearnedPolicyModel` owns the whole state transition --
  sensing, WorldState, observation (shared crowdrl-core), ONNX inference, action
  interpretation, integration to a new position, staying in-bounds, and any
  contact forces. Per-agent custom state (a frozen dataclass) carries velocity,
  torso / head angle, preferred speed, body dims, and memory, so the old
  Strategy A / B "orientation gap" is moot (both retired; see Section 3.6).

This corrects two errors in earlier drafts: JuPedSim does *not* "handle its own
collision resolution during deployment", and it does *not* integrate the model's
output -- it applies the returned position directly.

### Decisions taken (this session)

- Target **JuPedSim 2.0 `main`** (accept the moving pre-release branch to get the
  custom-model layer).
- **Reconcile the plan first** (this entry + the Section 3.6 rewrite) before
  writing adapter code.
- The local `C:\Users\Fabian\dev\jupedsim` `origin` is a personal fork; upstream
  `main` is the source of the 2.0 layer.

### Milestone status (updated)

- **M1-M3 (env / baseline / MARL pipeline):** COMPLETE as infrastructure; the M3
  quality bar (collision-free goal-reaching) remains the open behavioural target
  from the agent-dynamics campaign.
- **M4 (emergent phenomena):** IN PROGRESS / blocked on collision-clean
  behaviour, as before.
- **Module D (validation):** partially materialised early (scorecard + Kinora
  HDF5 export) but not yet framed or packaged as validation.
- **M9 (JuPedSim integration):** STARTED (this branch). De-risked by the
  custom-model layer and the existing exporter; a walking-skeleton adapter is now
  cheap and no longer needs to wait until Months 14-18.

### What remains / next steps

*(Checkbox states below are as of 2026-07-20 and are kept for the record. All
but the benchmark runner are done -- see the 2026-07-30 entry; note the
dependency item resolved the opposite way to what was expected, by declaring no
`jupedsim` dependency at all.)*

- [ ] Update `crowdrl-jupedsim/pyproject.toml` dependency from `jupedsim>=1.0` to
      the 2.0 line.
- [x] Walking-skeleton `LearnedPolicyModel(CustomOperationalModel)`: single
      agent, disc body, `ped.final_target` as goal, per-agent ONNX, no
      raycasts -- prove the loop runs.
- [ ] Obs-parity harness: identical physical config through the training obs
      builder and the adapter WorldState, asserted numerically identical
      (promote `crowdrl-env/tests/test_integration.py` to a real cross-engine
      test).
- [ ] Full obs (raycasts from `geometry`, faithful neighbour body / orientation),
      then batched-inference cache, then the benchmark runner (LearnedPolicyModel
      vs. CollisionFreeSpeedModel / SocialForceModel, trajectory-level +
      macroscopic metrics via the Kinora / pedpy export).
- Medium-term / deployment items from prior entries unchanged (Tiers 4-5, IAS-7
  importer, Tier 3 reward).

## 2026-07-30 -- Deployment path closed: trained *for* JuPedSim, validated at millimetre scale

The three sessions since 2026-07-20 built the adapter, hardened it, and found
eight train/deploy divergence channels (inventory and measured before/after in
`plan/lockstep_parity_analysis.md`; the two HIGH findings and the retraction of
the "policy absorbing state" reading are in `plan/handover_2026-07-30.md`). This
entry records what closed the loop: a policy **trained under the deployment
routing contract**, which turns the remaining train/deploy gap from a caveat
into a measurement.

### Headline

Fine-tuning the previous best checkpoint with `use_jupedsim_style_routing: true`
collapsed the corner-scenario trajectory gap between the training engine and the
deployed adapter from **~39 mm to 2.23 mm** (1% of a body radius, over a ~10 s
route), with the per-agent exit lag reduced to exactly JuPedSim's 2 bookkeeping
iterations. In both cases the comparison is the artefact against *its own*
training semantics -- i.e. the honest question "does deployment reproduce what
this policy was trained to do", not a flattering re-baselining.

Design principle #1 ("one observation builder, used everywhere") is what made
this measurable at all. The principle that earned its keep this session is
newer, and belongs in the plan explicitly: **do not train on a signal
deployment cannot supply.** Applied twice now -- `use_goal_direction=False`
(navigate by the routed waypoint alone) and now the routing contract -- it has
outperformed every attempt to reconstruct the richer signal at deployment time.

### The run: `exp_jps_routing_ft_r0400`

A short fine-tune of `exp_nogoaldir_stable_bigrooms_density_v4` r0400 (via
`--init_from`, fresh optimiser and curriculum) with exactly three deltas from
that recipe: the routing contract on, lr 5e-4 -> 2e-4 cosine, and a curriculum
starting at the composed phase rather than re-earning Tier 0. 2x RTX 4090,
DDP, 1.28 M agent-steps per rollout per rank (2.56 M effective per update).

Planned 600 rollouts; **stopped at ~575 and the artefact taken from rollout
125**, because the fixed eval suite regressed after ~rollout 150 and never fully
recovered:

| checkpoint | goal_rate | collisions | stuck | freeze | path_eff |
|---|---|---|---|---|---|
| r0400 baseline (previous best) | 0.954 | 0.116 | 0.159 | 0.099 | 0.917 |
| **r0125 (shipped)** | **0.975** | **0.093** | **0.032** | **0.084** | **0.920** |
| r0350 (trough) | 0.782 | 0.055 | 0.442 | 0.207 | 0.818 |
| r0500 (partial rebound) | 0.849 | 0.071 | 0.203 | 0.162 | 0.845 |

The shape of the regression is the interesting part: **collision rate kept
improving while freeze and stuck fractions climbed.** The policy was not
degrading randomly -- it was buying collision avoidance with goal completion,
i.e. becoming over-conservative in exactly the dense regimes we care about.
In-training GoalRate stayed at 0.82-0.90 throughout, so the training signal did
not show it; only the fixed-scenario scorecard did. Two process lessons:

- **The periodic in-training scorecard earned its cost.** Because every
  checkpoint had a scorecard written beside it during the run, selecting the
  best checkpoint was a lookup, not a re-evaluation campaign, and stopping early
  cost nothing.
- **In-training goal rate is not a stopping criterion.** It is measured on the
  training distribution, which the curriculum is simultaneously changing. The
  fixed suite is the one that saw this.

r0125's weakness is sharply localised: **14 of 15 scorecard scenarios score
goal_rate 1.000**, and the whole deficit sits in `composed_hi` at 100 agents
(goal 0.629, freeze 0.367, speed/preferred 0.41). That is the target for the
next round, and it is a *behavioural* target -- yielding, dense packing, wall
recovery -- not an interface one.

### Deployment validation

Against a local JuPedSim 2.0 source build at `49e3ddebd`:

| check | result |
|---|---|
| Full test suite | **660 passed, 2 skipped**, ruff clean (skips are the opt-in legacy-path scenario) |
| e2e corner, 4 agents | 4/4 exit, steps 729/808/896/973 (9.7 s) |
| e2e bottleneck, 12 agents, 1.4 m | 12/12 exit (7.1 s), min centre distance 0.416 m |
| Notebook 10, native vs adapter | worst-case deviation **2.23 mm**; exit lag exactly 2 steps for all 4 agents |
| Lockstep byte-parity | still byte-identical (`np.array_equal`, no tolerance) |

What remains irreducible is now cleanly separated from what was fixable:
JuPedSim's operational-model contract is a **per-agent callback**, so neighbours
are one step stale when the ego's contact forces are computed (channel 6) and
the exit stage lags removal by two iterations (channel 7); underneath sits a
~3.7e-15 float-reassociation floor in the observation builder (~1 ulp). None of
these are adapter defects, and no retraining removes them.

### `example_model` is now a shipped baseline, not a placeholder

`example_model/policy_r0125.onnx` (schema-v2 self-describing: obs/action config,
trained dynamics `w=0.8` / clamp 3.0 / contact 30000-500, and provenance
including per-field `dynamics_provenance`) is committed alongside its
`config_resolved.yaml` and `scorecard_r0125.json`, with `scorecard_r0400.json`
kept as the comparison baseline. `policy_r0400.onnx` is dropped. Torch
checkpoints and renders next to it are gitignored -- the `.pt` is reproducible
from the run directory, the render is regenerable. Consequence: the adapter
tests and notebook 10 are runnable from a fresh clone plus a JuPedSim build,
with no off-repo weights.

### Notebook 10 simplified to the deployment story

`examples/10_jupedsim_learned_model.ipynb` no longer demonstrates
`LockstepPolicyModel`. With the interactive adapter canonically correct, the
comparison that matters is native-vs-`LearnedPolicyModel`; Lockstep remains in
the package as a **validation instrument** (it bypasses the router, journeys and
stage transitions by design, and needs full geometry plus every exit polygon up
front), pinned by `tests/test_lockstep_byte_parity.py` and described rather than
demonstrated. Removing it also removed the last stale prose: both scenario
sections had claimed an agent is lost to a wall-facing absorbing state, which
was retracted on 2026-07-30 and is measurably false now.

Three bugs surfaced by re-running it end to end, all now fixed. Exit times were
being read from sqlite frame numbers (the recorder samples every 4th frame, so
with all agents exiting the old filter reported 1 of 4 exits -- `run_scenario`
now returns per-agent exit steps from the roster). The parity prose still
described the waypoint source and `path_deviation` as open gaps after the
fine-tune had closed them. And the third is an upstream trap worth recording on
its own:

**`SqliteTrajectoryWriter` must be closed explicitly, or the tail of every
recording is silently lost.** The writer commits every 100th write
(`commit_every_nth_write=100`) and `Simulation` never closes it -- it calls only
`begin_writing` and `write_iteration_state`, so nothing flushes the buffer at
the end of a run. The corner recording held 200 frames instead of 244 (7.96 s of
a 9.74 s run) and the bottleneck 100 instead of 178 (**44% of the run missing**),
with no error and no warning: the sqlite file is valid, just short. Every
JuPedSim example notebook works around this by reaching into the private
attribute (`simulation._writer.close()`), which is itself a smell -- there is no
public close, no context manager, and no destructor path. Two consequences for
us: the notebook now binds the writer and closes it, and the transfer-fidelity
number was re-measured over the full route (worst-case deviation is 2.23 mm
either way -- the truncation had been hiding the last 18% of the trajectory, not
inflating the agreement). Added to the upstream co-draft list below.

### Environment: reproducing the toolchain on a second machine

The 2026-07-30 handover's build recipe was machine-specific. Reproducing it on a
second Windows box surfaced three things worth recording, because each cost real
time:

- **JuPedSim 2.0 builds fine with CMake + Ninja + MSVC** (no `make` needed):
  configure with `-G Ninja -DPython_EXECUTABLE=<venv python>` from a
  `vcvars64.bat` shell, output `lib/py_jupedsim.cp312-win_amd64.pyd`, then put
  `<build>/lib` and `<src>/python_modules/jupedsim` on `sys.path` via a `.pth`
  in the venv's site-packages. Clone with `--recurse-submodules` (CGAL, fmt,
  glm, googletest, pybind11).
- **A too-new system VC++ runtime breaks extension modules.** System32
  `msvcp140`/`vcruntime140` at 14.50.35719 made DllMain fail for both
  onnxruntime's `onnxruntime_pybind11_state.pyd` and triton-windows'
  `libtriton.pyd` ("DLL initialization routine failed"). Torch itself was
  unaffected, so the only visible symptom was `torch.compile` silently falling
  back to eager -- a **2.9x throughput loss** (57 k -> 164 k agent-steps/s
  global) that a training log reports as one warning line. Fix: preload the VS
  2022 14.44 redist CRT from a venv-local directory via `sitecustomize.py`. The
  general lesson: "DLL initialization routine failed" across *several unrelated*
  pybind11 packages is a system-CRT problem, and CRTs can be too new, not only
  too old.
- **Windows multi-GPU DDP cannot use `torchrun` with these wheels.** They are
  built without libuv, and the elastic rendezvous requests the libuv TCPStore
  regardless of `USE_LIBUV=0`, so `train_mappo.py`'s auto-`torchrun` path dies
  before training starts. Launch the ranks directly with
  `RANK`/`LOCAL_RANK`/`WORLD_SIZE`/`MASTER_ADDR` and `USE_LIBUV=0`, and set
  `ddp_backend: gloo` (NCCL does not exist on win32). Worth fixing in
  `train_mappo.py` rather than rediscovering.

### Repository state (audited 2026-07-30)

The five-package structure from the 2026-07-20 entry holds; what changed is that
`crowdrl-jupedsim` is no longer a stub. **50 source modules, 45 test files, 662
collected tests.**

| Package | Modules | LOC | Tests | Role |
|---|---|---|---|---|
| `crowdrl-core` | 9 | ~4.0 k | 208 | The shared contract: `world_state`, `observation` (the single builder, 923 LOC), `sensing`, `navmesh` (A\* + funnel), `geometry`, `collision`, `action`, `config_io` |
| `crowdrl-env` | 9 | ~4.5 k | 157 | Gymnasium `CrowdEnv`, procedural generator (Tiers 0-3b), reward, spawner, solvability, visualiser, Kinora HDF5 export |
| `crowdrl-torch` | 15 | ~4.5 k | 66 | The GPU-batched training path (batched env, step, obs/sensing/reward ports, DD-PPO) |
| `crowdrl-train` | 13 | ~3.8 k | 94 | MAPPO loop, config, buffer/GAE, networks, **ONNX export**, scorecard, curriculum |
| `crowdrl-jupedsim` | 4 | ~1.4 k | 70 | `model.py` `LearnedPolicyModel` (the deployment path), `policy.py` (`OnnxPolicy` + metadata resolution), `lockstep.py` (the validation instrument) |

Four honest observations from the audit, each an action item rather than a
complaint:

- **The largest implementation file is not in a package.** Root `train_mappo.py`
  is ~1,900 LOC / 78 KB and is the real entry point (the `crowdrl-train`
  `train.py` loop is the library version). It has accumulated the experiment
  driver, the eval plots, the render/scorecard subprocess spawning and the
  torchrun relaunch. It works, but it is the one place where "packages hold the
  implementation" is not true.
- **The config that produced the shipped model is untracked.**
  `configs/*.yaml` is gitignored except three whitelisted files, by deliberate
  policy (experiment configs stay local). The reproducibility substitute is
  real -- `example_model/config_resolved.yaml` is committed and is the resolved
  config of the shipping run -- but the recipe file itself
  (`exp_jps_routing_ft_r0400.yaml`) exists only on the training machine. Either
  whitelist the configs that produce shipped artefacts, or state that
  `config_resolved.yaml` is the contract. The latter is cheaper and already true.
- **There are two parallel doc trees.** `plan/` (22 files) and `docs/` (5 files:
  `agent_pipeline.md`, `environment_mechanics.md`, and three dated summaries)
  overlap in purpose with no stated precedence. `plan/` is the canonical one --
  it holds this document and the progress log; `docs/` holds the two long
  mechanism references that are genuinely useful and three session summaries
  that duplicate the `plan/handover_*` role.
- **No `TODO`/`FIXME` markers anywhere** in `packages/`, `scripts/` or `tests/`.
  Open work is recorded in prose (docstrings and handovers) instead. That is a
  defensible convention, but it means the deliberate gaps below are invisible to
  a code reader: the YAML lossy-gap guard in `config_io.cfg_dict_from_env_config`
  (docstring-only), Lockstep's fixed-roster/no-pass-detection limitation,
  `crowd_env.py:243` `preferred_speeds` for temporal-off configs, and the absent
  full-step numpy-vs-torch trajectory equivalence test.

### Milestone status (updated)

- **M1-M3:** unchanged as infrastructure. The M3 quality bar is now *measured
  per scenario* rather than asserted: collision-free goal-reaching holds
  everywhere except the 100-agent composed case.
- **M4 (emergent phenomena):** still the open behavioural target; the
  over-conservatism finding above is the concrete blocker to attack first.
- **M9 (JuPedSim integration):** adapter, self-describing artefact, e2e
  scenarios, byte-exact validation instrument and example notebook all
  delivered, years ahead of the Months 14-18 slot. Outstanding: the cross-model
  benchmark runner and the public release.
- **Module D (validation):** the scorecard is now demonstrably load-bearing (it
  is what caught the regression and selected the artefact). Framing it as a
  validation package is overdue.

### What remains / next steps

Ordered by what unblocks the most:

- [ ] **Attack the 100-agent composed regime** -- the only failing scenario, and
      the one M4 depends on. The over-conservatism signature (collisions down,
      freeze up) suggests the proximity/collision penalties now dominate the
      progress term at density; a density-aware weighting or a wall-facing
      zero-progress penalty are the cheap experiments, and the trained-but-unused
      240 deg/s low-speed pivot is available for wall recovery.
- [ ] **Cross-model benchmark runner** (LearnedPolicyModel vs.
      CollisionFreeSpeedModel / SocialForceModel on one scenario, trajectory-level
      + macroscopic metrics through the Kinora/pedpy export). This is the last
      piece of Step 4 and the natural bridge to M7.
- [ ] **`no_p_dev` ablation** -- the routing contract pins `path_deviation` to
      0.0, which is not the same as training without the channel. Removing it
      outright is the cleaner statement of the same principle.
- [ ] Make Windows multi-GPU launch work out of the box in `train_mappo.py`
      (direct-rank fallback when the libuv rendezvous is unavailable).
- [ ] **YAML lossy-gap guard** in `cfg_dict_from_env_config` (carried from the
      2026-07-29 review, still docstring-only): raise when a non-default
      `RaycastConfig` / `k_neighbours` / `head_limit` would be silently dropped.
- [ ] **Upstream JuPedSim items to co-draft, never post directly** (carried,
      unchanged): `Agent.next_target` calling `.next_destination`;
      `environment_query.py`'s two unbound methods; radius-aware/configurable
      router inset (verified to be a feature request, not a missed knob -- the
      A* assumes point-size agents); whether the 2-iteration exit lag is
      intended; the dt convention.
- [ ] **Pass detection via `Simulation.iteration_count()`** (carried) -- would
      eliminate the wholesale-roster-replacement failure class in Lockstep, but
      needs the Simulation handle after construction. Documented limitation for
      now.
- [ ] Medium-term items unchanged: Tiers 4-5, IAS-7 geometry importer, Tier 3
      distributional reward, `plan/CrowdRL_Project_Plan_v5.docx` staleness.


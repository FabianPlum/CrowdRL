**CrowdRL**

Learning Crowd Navigation Policies via

Multi-Agent Reinforcement Learning on Synthetic Arenas

Project Plan — Draft for Internal Discussion

Dr. Fabian Plum

IAS-7 — Zivile Sicherheitsforschung

Forschungszentrum Jülich

March 2026

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

JuPedSim computes shortest paths by triangulating the walkable area and measuring distances between triangle centroids. Our environment generator adopts the same approach: upon geometry creation, the walkable polygon is triangulated (constrained Delaunay triangulation via Triangle or Shapely), producing a navigation mesh. An A* search on the triangle adjacency graph then verifies, for every (spawn, goal) pair, whether a path exists. The verification step operates in three modes:

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

## 3.3 Observation and Action Spaces (Module B)

**Observation space (per agent)**

The observation vector comprises three components: ego state, social sensing, and environment sensing.

**Ego state (7D)**

Relative goal position (2D), own velocity (2D), own torso orientation angle relative to heading (1D), own head orientation angle relative to torso (1D, clamped to ±90°), and current heading direction (1D). The torso and head angles are separated because they are independently actuated: the head can rotate up to 90° left or right of the torso’s forward-facing direction before shoulder rotation is required. This distinction matters because the raycast FOV follows the head, not the torso—an agent can scan its surroundings without reorienting its body. All values are in the agent’s local egocentric reference frame.

**Social sensing (K × 7D)**

For each of the K nearest neighbours: relative position (2D), relative velocity (2D), body orientation (1D), and approximate body dimensions (2D). With K=8 the social component is 56D.

**Environment sensing via raycasting (N-rays D)**

Without explicit obstacle and wall sensing, agents are effectively blind to the geometry they navigate. This is addressed via a tuneable raycast sensor: N rays are emitted from the agent’s position at evenly spaced angles within its field of vision (FOV), which is anchored to the head orientation, not the torso. Each ray returns the distance to the first intersection with a wall, obstacle, or other agent’s collision boundary, normalised to the maximum sensing range. If a ray hits nothing within range, it returns 1.0 (clear). This yields an N-dimensional vector of distance readings.

The key design parameters are: (a) N, the number of rays, which controls angular granularity (default: 16, spanning a 200° frontal FOV); (b) maximum sensing range (default: 5m); and (c) whether rays cover the full 360° or only the head’s forward-facing field of vision. The FOV-restricted variant is more biologically plausible and creates a perception–action trade-off: the agent can gather information by turning its head (cheap, up to ±90°) without reorienting its torso, but extreme lateral scanning requires shoulder rotation that may hinder forward locomotion. This decoupling means the policy must learn when to look versus when to physically reorient—a decision that real pedestrians make constantly in dense crowds. The 360° (omniscient) variant serves as an ablation control to quantify how much limited, head-anchored perception affects emergent collective behaviour.

Optionally, each ray can return a 2-channel signal rather than a scalar: (distance, hit-type), where hit-type encodes whether the intersection was with a wall (0), an obstacle (0.5), or another agent (1). This allows the policy to distinguish between static geometry and moving neighbours at the cost of doubling the environment-sensing dimensionality.

**Total observation dimensionality.** With K=8 neighbours and N=16 single-channel rays: 7 (ego) + 56 (social) + 16 (raycast) = 79D. With 2-channel rays: 7 + 56 + 32 = 95D. Optionally, the navigation mesh router (see Module A) provides a next-waypoint direction (2D) and a path deviation scalar (1D), bringing the “signposted” agent total to 82D or 98D respectively. All variants are well within the range where MLP policies train reliably.

**Action space**

Continuous, 4-dimensional: desired speed (scalar), desired heading change (scalar), desired torso orientation change (scalar), and desired head orientation change relative to torso (scalar, clamped to ±90°). The head and torso are independently actuated, mirroring human anatomy: the head can freely rotate up to 90° left or right of the torso’s forward-facing direction; beyond that range, shoulder rotation is required. Since the raycast FOV follows the head, this creates two distinct action channels: the head-turn is a low-cost information-gathering action (look around without changing body configuration), while the torso-turn is a higher-cost physical reorientation that alters the collision profile. The policy must learn to use both strategically—for example, scanning an approaching bottleneck with a head turn before committing to a shoulder rotation to slip through.

**Policy architecture**

Shared-parameter PPO with an actor-critic MLP. All agents share one policy network, conditioned on local observations. This is the standard MARL approach for homogeneous agent populations and scales well to hundreds of agents. Agent heterogeneity (body size, desired speed) enters through the observation, not through separate networks.

## 3.4 Reward Function (Module C)

The reward function is the core scientific contribution. It operates in three tiers:

- **Tier 1 — Sparse task rewards.** Goal-reaching bonus (+10), collision penalty (−1 per timestep in contact), timeout penalty (−5 if goal not reached within episode). These alone produce functional but potentially alien-looking navigation.
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

The most consequential architectural decision is not which RL algorithm to use—it is where the boundary sits between code that is specific to training and code that must be shared with deployment inside JuPedSim. If the observation construction logic, the raycast engine, or the action interpretation code exists in two separate implementations, they will inevitably drift, and the transfer from training environment to JuPedSim will silently break. The architecture is therefore organised as four Python packages with a shared foundation.

**Package 1: crowdrl-core (shared foundation)**

This package contains everything that must be identical between training and deployment. It has no dependency on any RL library, no dependency on Gymnasium, and no dependency on JuPedSim. It is a pure geometry/perception/action library. Its submodules are:

- **geometry:** Shapely polygon handling, constrained Delaunay triangulation, navigation mesh construction, wall-segment extraction from polygon boundaries.
- **navmesh:** A* shortest-path solver on the triangle adjacency graph. Computes next-waypoint direction and path-deviation scalar for any (position, goal) pair. Used both for solvability verification (during environment generation) and as an optional observation signal (during training and deployment).
- **sensing:** Raycast engine (N rays, configurable FOV, head-anchored). K-nearest-neighbour query for social sensing. Both operate on a generic WorldState dataclass containing agent positions, velocities, orientations, and wall segments—agnostic to whether this state comes from the training environment or from JuPedSim.
- **observation:** Assembles the full observation vector from ego state, social sensing, raycasts, and optional navmesh signals. Takes a WorldState and an agent index, returns a numpy array. This is the single function that must be identical between training and deployment—any discrepancy here means the policy sees a different world and produces wrong actions.
- **action:** Maps the 4D policy output (desired speed, heading change, torso orientation change, head orientation change) to kinematic quantities: a desired velocity vector, a new torso angle, and a new head angle. Enforces the ±90° head-to-torso constraint. During training, these feed back into the physics step. During deployment, the desired velocity feeds into JuPedSim’s simulation loop.
- **collision:** Elliptical agent collision detection and contact force computation. Used only by the training environment (JuPedSim handles its own collision resolution during deployment). Included in core because the raycast engine needs to intersect rays with agent collision boundaries.

**Package 2: crowdrl-env (training environment)**

Depends on crowdrl-core and Gymnasium. Contains everything specific to the training loop that is not needed at deployment time: the procedural geometry generator (tiers 0–5), the solvability verifier, the crowd composition sampler, the reward modules (tiers 1–3), and the Gymnasium environment wrapper. The wrapper’s step() function calls crowdrl-core’s observation builder and action interpreter, adds physics integration and collision resolution from crowdrl-core’s collision module, and computes rewards. The reset() function calls the procedural generator, runs the solvability verifier, and initialises agent states. External geometry import (IAS-7 test geometries, JuPedSim geometry files) is handled by a loader that produces the same Shapely polygon consumed by the procedural path.

**Package 3: crowdrl-train (training infrastructure)**

Depends on crowdrl-env, PyTorch, and an MARL library (CleanRL or PettingZoo). Contains the MAPPO training loop, the curriculum manager (controls which geometry tiers and agent counts are used at each training stage), hyperparameter configuration, logging (Weights & Biases or TensorBoard), checkpointing, and policy export. The export step converts the trained PyTorch policy to ONNX format, which is the portable artefact consumed by Package 4. This package is never needed at deployment time.

**Package 4: crowdrl-jupedsim (deployment adapter)**

Depends on crowdrl-core, JuPedSim, and ONNX Runtime. This is the integration layer. Its central class is LearnedPolicyModel, which implements JuPedSim’s operational model interface. At each simulation timestep, LearnedPolicyModel performs four steps: (1) reads JuPedSim agent states (positions, velocities, goals) and constructs a WorldState dataclass; (2) calls crowdrl-core’s observation builder to construct per-agent observation vectors—this is the exact same function call that runs during training; (3) runs batch ONNX inference on all observation vectors to produce 4D action outputs; (4) calls crowdrl-core’s action interpreter to map actions to desired velocities and orientations, which JuPedSim’s simulation loop then integrates.

Critically, this package does not depend on PyTorch or on crowdrl-env or crowdrl-train. The only artefact it needs from the training side is the exported .onnx file. A JuPedSim user installs crowdrl-jupedsim and crowdrl-core, loads a policy file, and uses it exactly like any other JuPedSim model. This also contains the benchmark runner: a harness that runs the same scenario with LearnedPolicyModel and with JuPedSim’s existing models (CollisionFreeSpeedModel, SocialForceModel, etc.) and compares trajectory-level and macroscopic metrics.

**The WorldState interface: the contract that holds everything together**

The key abstraction in crowdrl-core is the WorldState dataclass: a flat representation of everything the perception system needs to construct observations. It contains arrays of agent positions, velocities, torso orientations, head orientations, body dimensions, and goal positions, plus the walkable polygon and precomputed wall segments. During training, crowdrl-env populates WorldState from its internal physics state. During deployment, crowdrl-jupedsim populates WorldState from JuPedSim’s agent state API. The observation builder and sensing modules consume only WorldState—they never know which system produced it.

This is the architectural invariant that guarantees transfer: if WorldState is populated correctly from JuPedSim’s agent states, the observation vector will be numerically identical to what the policy saw during training for the same physical configuration. Any drift between the two population paths (training vs. deployment) is a bug that will produce subtle policy failures—so integration tests that compare observations from both paths on identical configurations are a first-class part of the test suite.

**Handling the representation gap: torso and head orientation in JuPedSim**

JuPedSim’s current agent model does not track torso or head orientation—agents have a position, a velocity, and a desired direction, but no body angle. The learned policy requires these. There are two strategies to resolve this:

- **Strategy A: Adapter-side state tracking.** LearnedPolicyModel maintains its own torso and head orientation state per agent as side-channel data, updated each timestep by the policy’s action output. JuPedSim does not need to know about these angles—it receives a desired velocity vector and handles movement as usual. The orientation state is private to the adapter and only affects observation construction and raycast direction. This is the zero-modification path: JuPedSim’s codebase is untouched.
- **Strategy B: JuPedSim agent state extension.** Extend JuPedSim’s agent model to include torso and head orientation as first-class state variables. This is the cleaner long-term solution—it means the orientation state participates in JuPedSim’s collision resolution and trajectory output—but it requires a pull request to JuPedSim and agreement from the modelling team.

Recommendation: start with Strategy A for the initial integration (Milestone M9). It gets a working system without blocking on JuPedSim core changes. Once the learned policy demonstrates value, propose Strategy B as a JuPedSim feature request backed by concrete results. The transition from A to B is straightforward: move the orientation state from the adapter’s side channel into JuPedSim’s agent struct, and update the WorldState population code in crowdrl-jupedsim to read from there instead of from its own bookkeeping.

**Incremental build path**

The packages are built in dependency order, with each stage producing a usable artefact:

- **Step 1: crowdrl-core.** Build geometry, navmesh, sensing, observation, and action modules. Write unit tests with hand-constructed WorldState instances. This is testable in complete isolation before any RL code exists. Deliverable: a library that, given a polygon and a set of agent states, produces observation vectors and interprets actions.
- **Step 2: crowdrl-env.** Build the Gymnasium wrapper, procedural generator (start with Tiers 0–2), solvability verifier, and Tier 1–2 reward modules. Verify with a random-policy baseline (agents take random actions; confirm observations look correct, rewards are distributed as expected, episodes terminate properly). Deliverable: a Gymnasium environment that produces episodes with procedural or imported geometries.
- **Step 3: crowdrl-train.** Implement the MAPPO training loop, curriculum manager, and policy export. Train initial policies on Tier 0–2 environments. Deliverable: trained .onnx policy files and training logs.
- **Step 4: crowdrl-jupedsim.** Build the LearnedPolicyModel adapter and the benchmark runner. Write integration tests that verify observation consistency between the training environment and the JuPedSim adapter on identical configurations. Deliverable: a JuPedSim-compatible model that loads a .onnx file and runs as a drop-in replacement for CollisionFreeSpeedModel.

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
| IV | M9: JuPedSim integration | Package learned policy as a JuPedSim locomotion module. Open-source release with documentation and example notebooks. | Months 14–18 |

# 5. Key Design Decisions (Summary)

The following table summarises the critical architectural choices and their rationale. Each of these is a point where your own judgement should override this document—they are recommendations, not prescriptions.

| Decision | Recommended Choice | Rationale |
|----------|-------------------|-----------|
| Training environment | Lightweight 2D physics (not UE5) | UE5 in the training loop is 100–1000× slower. Vector observations don’t benefit from rendering. Architect for a module swap to egocentric vision later. |
| Observation space | Egocentric vector: ego state (7D) + K=8 neighbours (56D) + N=16 head-anchored raycasts (16D) = 79D | Three-component design: ego state, social sensing (K nearest neighbours), and environment sensing (FOV-restricted raycasts for wall/obstacle distances). Raycasts prevent geometry-blindness. FOV restriction couples perception to torso orientation, making the orientation action meaningful. N and FOV are tuneable for ablation. |
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
- [ ] Large-scale training runs with GPU env + navmesh waypoints
- [ ] Verify agents learn to follow waypoints in Tier 1-2 geometries
- [ ] Verify curriculum progresses through Tier 3a/3b phases
- [ ] Document emergent behaviours (M4)

**Medium-term:**
- [ ] Geometry Tiers 4-5 (building floors, multi-floor evacuation)
- [ ] External geometry importer (IAS-7 test geometries)
- [ ] Tier 3 reward (distributional style matching from PeTrack data)

**Deployment:**
- [ ] crowdrl-jupedsim package (ONNX runtime adapter)
- [ ] Integration tests (obs parity between training and deployment)


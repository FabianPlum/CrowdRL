# CrowdRL Validation Benchmark Plan

## 1. Purpose

This document details the validation and benchmarking strategy for CrowdRL learned
pedestrian navigation policies. The goal is to test whether MARL-trained agents
reproduce known emergent phenomena from pedestrian dynamics literature -- phenomena
that were never explicitly trained for -- and to quantify how they compare against
established hand-crafted models (JuPedSim's CollisionFreeSpeedModel,
GeneralizedCentrifugalForceModel, SocialForceModel, AnticipationVelocityModel).

This plan builds on the validation suite described in Section 3.5 of the
CrowdRL Project Plan v5 (Module D) and operationalises Milestones M4 (emergent
phenomena), M7 (benchmark validation), and M8 (zero-shot transfer).

---

## 2. Validation Scenarios

Each scenario targets one or more known collective phenomena. Scenarios are ordered
by increasing complexity. All geometries use Shapely Polygons with holes, matching
both CrowdRL and JuPedSim conventions.

### 2.1 Lane Formation in Bidirectional Flow

**Phenomenon:** Spontaneous self-organisation of counterflowing pedestrians into
unidirectional lanes, reducing collisions and increasing throughput.

**Reference:**
- Feliciani & Nishinari, Phys. Rev. E 94, 032304 (2016)
- JuPedSim lane-formation notebook

**Geometry:**
- Corridor: 3.0m wide x 38.0m long (matching JuPedSim notebook)
- 12m spawn zones at each end, 10m central measurement area (x in [14, 24])
- No obstacles -- pure bidirectional counterflow setup
- Shapely: `box(0, 0, 38, 3)`

**Agent Setup:**
- Total: 54 agents (matching Feliciani experiment)
- Four flow ratio configurations:
  - 6/0 (unidirectional, r=0.000): 54 left-to-right, 0 right-to-left
  - 5/1 (r=0.167): 45 left-to-right, 9 right-to-left
  - 4/2 (r=0.333): 36 left-to-right, 18 right-to-left
  - 3/3 (balanced, r=0.500): 27 left-to-right, 27 right-to-left
- Preferred speed: 1.34 m/s +/- 0.1 m/s (Weidmann distribution)
- Spawned in grid formation in waiting areas at corridor ends

**Key Metrics (see Section 3):**
- Order parameter (Phi)
- Rotation range
- Fundamental diagram (density vs flow)
- x-velocity and y-velocity distributions per phase
- Crossing time statistics
- Band index (lateral position clustering)

**What Success Looks Like:**
- Order parameter Phi > 0.8 for balanced flow (3/3), indicating clear lane structure
- Lanes emerge without explicit lane-seeking reward
- Crossing times comparable to empirical values (~7.3s average)

---

### 2.2 Bottleneck Flow and Faster-is-Slower Effect

**Phenomenon:** At narrow bottlenecks, increasing desired speed beyond a threshold
*decreases* throughput due to clogging. Known as the "faster-is-slower" effect
(Helbing, Farkas, Vicsek, Nature 2000).

**Reference:**
- Seyfried et al. (2009), Enhanced empirical data for the fundamental diagram
  and the flow through bottlenecks (Springer)
- Seyfried et al. (2005), J. Stat. Mech. P10002 (single-file fundamental diagram)
- JuPedSim double-bottleneck notebook
- Rimea Test 11 (bottleneck flow)

**Geometry:**
- Room: 10m x 10m with single exit bottleneck
- Bottleneck widths: 0.8m, 1.0m, 1.2m, 1.5m, 2.0m, 3.0m
- Approach corridor: 2m before bottleneck
- Exit corridor: 5m after bottleneck
- Shapely: rectangular polygon with two rectangular hole "walls" forming the
  bottleneck channel

**Agent Setup:**
- 20, 40, 60, 80, 100 agents (density sweep)
- All agents move from room toward exit through bottleneck
- Preferred speed: 1.0 to 2.0 m/s in 0.2 m/s increments (speed sweep)
- Spawned uniformly in the room area

**Key Metrics:**
- Flow rate J(t) at bottleneck cross-section (persons/m/s)
- Specific flow J_s = J / bottleneck_width
- Time-to-clear (all agents through bottleneck)
- Density profile upstream of bottleneck (should show compression wave)
- Clogging frequency: number of flow interruptions > 1s
- Faster-is-slower curve: flow rate vs desired speed at each bottleneck width

**What Success Looks Like:**
- Specific flow matches Seyfried et al. empirical values (1.6-2.0 pers/m/s for
  b >= 1.0m)
- Faster-is-slower effect emerges at narrow bottlenecks (b <= 1.0m)
- Arch-like formations visible upstream of bottleneck

---

### 2.3 Single-File Movement

**Phenomenon:** In narrow corridors where overtaking is impossible, pedestrians
form single-file chains. Speed adapts to the slowest member. Step-and-stop
waves propagate backward through the chain.

**Reference:**
- Seyfried et al., Phys. Rev. E (2006)
- JuPedSim single-file notebook

**Geometry:**
- Corridor: 0.8m wide x 20m long (prevents overtaking)
- Circular variant: oval track with 0.8m width (for steady-state measurement)

**Agent Setup:**
- 5, 10, 15, 20, 25 agents (density sweep in narrow corridor)
- All unidirectional
- Heterogeneous preferred speeds (mean 1.34 m/s, std 0.26 m/s)

**Key Metrics:**
- Headway distribution (distance to agent ahead)
- Speed-density fundamental diagram (single-file regime)
- Speed adaptation time (how quickly agents match leader speed)
- Stop-and-go wave propagation speed
- Time-space diagram (position vs time for each agent)

**What Success Looks Like:**
- Linear speed-density relationship in single-file regime
- Agents maintain minimum headway without excessive stopping
- Speed adapts smoothly to slower leaders (no oscillations)

---

### 2.4 Movement Around Corners

**Phenomenon:** Pedestrians slow down at corners and follow the inner wall (shortest
path), creating asymmetric density distributions. Known as the "corner effect."

**Reference:**
- RiMEA Test 6
- JuPedSim corner notebook

**Geometry:**
- L-shaped corridor: 3m wide, 10m per arm, 90-degree bend
- Variations: sharp corner vs rounded corner (radius 0, 1m, 2m)

**Agent Setup:**
- 20-40 agents, unidirectional flow around the corner
- Continuous injection at entrance (one agent per 1-2 seconds)

**Key Metrics:**
- Speed profile around corner (deceleration/acceleration)
- Trajectory curvature (inner-wall bias)
- Density distribution in corner region
- Flow rate vs corner angle

**What Success Looks Like:**
- Agents slow down approaching the corner
- Trajectories hug the inner wall
- Speed recovery within 2-3m after the corner

---

### 2.5 Room Evacuation

**Phenomenon:** Evacuation from a room with one or two exits exhibits competitive
dynamics, arch formation at exits, and the faster-is-slower effect under panic
conditions.

**Reference:**
- Helbing, Farkas, Vicsek, Nature 407 (2000)
- RiMEA Test 11/12

**Geometry:**
- Room: 15m x 15m
- Single exit: 1.0m wide, centred on one wall
- Two-exit variant: 1.0m exits on opposite walls (route choice test)

**Agent Setup:**
- 50, 100, 150 agents
- All agents target the exit(s)
- Speed levels: "calm" (1.0 m/s), "hurried" (1.5 m/s), "panicked" (2.0 m/s)

**Key Metrics:**
- Evacuation time (last agent exits)
- Flow rate at exit over time
- Pressure (density) near exit
- Arch formation detection (semi-circular density front upstream of exit)
- Exit utilisation ratio (two-exit variant)

**What Success Looks Like:**
- Evacuation time scales sub-linearly with agent count
- Faster-is-slower effect visible at "panicked" speed
- Arch formation at exit visible in density maps

---

### 2.6 T-Junction and Merging Flow

**Phenomenon:** At T-junctions, two merging streams alternate access (zipper
merging). Dominant stream suppresses minority stream at high density ratios.

**Reference:**
- Zhang et al., J. Stat. Mech. (2012)
- CrowdRL Tier 2 geometry generator (T-junctions already supported)

**Geometry:**
- Main corridor: 3m x 15m
- Branch corridor: 3m x 8m, joining at midpoint
- T-junction angle: 90 degrees

**Agent Setup:**
- Main flow: 30 agents (left to right)
- Branch flow: 10, 20, 30 agents (branch to main corridor)

**Key Metrics:**
- Merging efficiency (fraction of agents that merge without stopping)
- Flow rate at junction cross-section
- Zipper ratio (alternation frequency between streams)
- Conflict count (simultaneous agents in junction zone)

**What Success Looks Like:**
- Zipper-like alternation at balanced flow ratios
- Dominant stream takes priority at imbalanced ratios
- No deadlock at junction

---

### 2.7 Waiting and Queuing

**Phenomenon:** Pedestrians form queues at service points or narrow passages. Queue
discipline, approach behaviour, and personal space maintenance are culturally and
density-dependent.

**Reference:**
- JuPedSim queues_waiting notebook

**Geometry:**
- Corridor: 5m wide x 20m long with a 1m gate at the midpoint
- Gate opens/closes periodically (every 10s open, 5s closed)

**Agent Setup:**
- 30 agents approaching from one side
- Gate blocks passage when closed, opens periodically

**Key Metrics:**
- Queue formation time
- Queue length over time
- Spacing in queue (personal space)
- Flow rate after gate opens

---

## 3. Metrics Catalogue

### 3.1 Macroscopic Metrics

| Metric | Formula | Unit | Tool |
|--------|---------|------|------|
| Density (rho) | N_area / A | persons/m^2 | Voronoi cells (PedPy) |
| Flow (J) | rho * v_x | persons/(m*s) | Measurement line |
| Specific flow (J_s) | J / width | persons/(m*s) | Measurement line |
| Speed (v) | mean(\|v_i\|) over measurement area | m/s | Per-frame average |
| Fundamental diagram | rho vs J scatter | - | All of above |

### 3.2 Lane Formation Metrics

| Metric | Formula | Source |
|--------|---------|-------|
| **Order parameter (Phi)** | Phi = (1/N) sum_j ((n_j^L - n_j^R)/(n_j^L + n_j^R))^2 | Feliciani & Nishinari (2016), Eq. 6-7 |
| **Rotation range** | max(r_z) - min(r_z), where R = curl(F) on velocity grid | Feliciani & Nishinari (2016), Eq. 8-9 |
| **Band index** | Fraction of agents whose y-position lateral neighbours share their direction | Adapted from Rex & Lowen (2007) |
| **Number of lanes** | Peaks in y-position histogram of x-velocity sign clusters | Visual + automatic |

**Order parameter details:**
- Divide measurement area into a grid of 0.2m x 0.2m cells
- For each row j, count agents moving left (n_j^L) and right (n_j^R) based on x-velocity sign
- phi_j = ((n_j^L - n_j^R) / (n_j^L + n_j^R))^2
- Phi = mean(phi_j) over all rows
- Phi = 1.0: perfect lanes (each row has only one direction); Phi -> 0: fully mixed
- Note: only meaningful during full bidirectional flow (Phase 3 of Feliciani's 5-phase model)

**Rotation range details:**
- Same 0.2m grid as order parameter
- Compute mean x-velocity and mean y-velocity in each cell -> vector field F(x,y)
- Compute curl: r_z = dF_y/dx - dF_x/dy (finite differences on grid)
- Rotation range = max(r_z) - min(r_z) across all cells
- Low rotation range -> laminar flow (well-formed lanes)
- High rotation range -> turbulent flow (lane dissolution, mixing)

### 3.3 Trajectory-Level Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| **Crossing time** | Time for agent to traverse measurement area | s |
| **Path efficiency** | Euclidean distance / actual path length | ratio (0-1] |
| **Speed autocorrelation** | Temporal correlation of agent speed | dimensionless |
| **Angular change distribution** | Distribution of heading changes per step | rad |
| **Neighbour distance** | Distribution of distances to nearest neighbour | m |
| **Collision rate** | Fraction of timesteps with body overlap | ratio |

### 3.4 Body-Level Metrics (CrowdRL-specific)

These metrics leverage CrowdRL's richer agent model (independent head/torso):

| Metric | Description |
|--------|-------------|
| **Head scanning frequency** | How often agents turn head > 30 deg from torso |
| **Shoulder turning at bottlenecks** | Torso angle change rate in constrictions |
| **Anticipatory gaze** | Head turns toward approaching agents before evasion |
| **Gaze-action lag** | Time between head turn and subsequent body turn |

---

## 4. Comparison Protocol

### 4.1 Models to Compare

| Model | Source | Type |
|-------|--------|------|
| **CrowdRL (ours)** | ONNX policy from crowdrl-train | Learned (MARL) |
| **CollisionFreeSpeedModel** | JuPedSim built-in | Velocity-based |
| **GeneralizedCentrifugalForceModel** | JuPedSim built-in | Force-based |
| **SocialForceModel** | JuPedSim built-in | Force-based |
| **AnticipationVelocityModel** | JuPedSim built-in | Velocity-based |

### 4.2 Head-to-Head Comparison Method

For each scenario:

1. **Same geometry.** Identical Shapely polygon loaded into both CrowdRL env and JuPedSim.
2. **Same agent count and spawn positions.** Deterministic seeding.
3. **Same preferred speeds.** Drawn from same distribution with same seed.
4. **Same simulation time.** Fixed maximum timesteps.
5. **Trajectory collection.** Both systems output (t, agent_id, x, y, vx, vy) at 10 Hz.
6. **Metric computation.** Identical analysis code applied to both trajectory sets.
7. **Statistical comparison.** Each scenario run N=10 times with different seeds.
   Report mean +/- std for all metrics. Two-sample t-test or Mann-Whitney U for
   significance.

### 4.3 Trajectory Output Format

Standard format for all models (compatible with PedPy):

```
frame, id, x, y, vx, vy, torso_angle, head_angle
```

Frame rate: 100 Hz (CrowdRL dt=0.01s), downsampled to 10 Hz for analysis.

---

## 5. Implementation Roadmap

### Phase 1: Lane Formation Benchmark (Current)
- [x] Define corridor geometry as Shapely polygon
- [x] Implement ONNX model loading and inference loop
- [x] Run bidirectional flow with 4 flow ratios
- [x] Compute order parameter, rotation range
- [x] Compute fundamental diagram, velocity distributions
- [x] Visualise trajectories colour-coded by direction and speed
- [ ] Compare against JuPedSim CollisionFreeSpeedModel (requires crowdrl-jupedsim)
- Notebook: `examples/08_lane_formation_test.ipynb`

### Phase 2: Bottleneck and Single-File
- [ ] Bottleneck geometry generator (parameterised width)
- [ ] Faster-is-slower speed sweep
- [ ] Single-file corridor with density sweep
- [ ] Fundamental diagram comparison
- Notebooks: `examples/09_bottleneck_test.ipynb`, `examples/10_single_file_test.ipynb`

### Phase 3: Complex Scenarios
- [ ] Corner movement (L-bend from Tier 2 generator)
- [ ] Room evacuation
- [ ] T-junction merging flow
- Notebooks: `examples/11_corner_test.ipynb`, `examples/12_evacuation_test.ipynb`

### Phase 4: Full Comparison Suite
- [ ] Implement crowdrl-jupedsim adapter (LearnedPolicyModel)
- [ ] Run all scenarios with all JuPedSim models
- [ ] Statistical comparison tables
- [ ] Publication-quality figures
- Notebook: `examples/13_model_comparison.ipynb`

---

## 6. Analysis Dependencies

| Package | Purpose |
|---------|---------|
| `crowdrl-core` | WorldState, observation builder, action interpreter |
| `crowdrl-env` | CrowdEnv, geometry generator, visualiser |
| `crowdrl-train` | Export/load utilities, config |
| `onnxruntime` | ONNX model inference |
| `matplotlib` | Visualisation |
| `numpy` | Numerical computation |
| `scipy` | Voronoi tessellation, statistical tests |
| `shapely` | Geometry definition |

PedPy integration is deferred to Phase 4 (requires trajectory export to PedPy format).
For Phases 1-3, all metrics are computed directly from CrowdRL trajectory arrays.

---

## 7. Key References

1. Feliciani, C. & Nishinari, K. (2016). Empirical analysis of the lane formation
   process in bidirectional pedestrian flow. *Phys. Rev. E*, 94, 032304.
   -- Order parameter, rotation range, 5-phase model, velocity grid analysis

2. Seyfried, A., Steffen, B., Klingsch, W. & Boltes, M. (2005). The fundamental
   diagram of pedestrian movement revisited. *J. Stat. Mech.*, P10002.
   -- Single-file fundamental diagram, density-velocity relationship

2b. Seyfried, A., Boltes, M., Kähler, J., Klingsch, W., Portz, A., Rupprecht, T.,
   Schadschneider, A., Steffen, B. & Winkens, A. (2009). Enhanced empirical data
   for the fundamental diagram and the flow through bottlenecks. In *Pedestrian
   and Evacuation Dynamics 2008*, pp. 145-156. Springer.
   -- Bottleneck fundamental diagrams, specific flow measurements

3. Helbing, D., Farkas, I. & Vicsek, T. (2000). Simulating dynamical features of
   escape panic. *Nature*, 407, 487-490.
   -- Faster-is-slower effect, arch formation, competitive evacuation

4. Rex, M. & Löwen, H. (2007). Lane formation in oppositely charged colloids
   driven by an electric field: Chaining and two-dimensional crystallization.
   *Phys. Rev. E*, 75, 051402.
   -- Order parameter for lane detection in bidirectional particle flow

5. Nowak, S. & Schadschneider, A. (2012). Quantitative analysis of pedestrian
   counterflow in a cellular automaton model. *Phys. Rev. E*, 85, 066128.
   -- Order parameter adapted for pedestrian dynamics

6. Zhang, J., Klingsch, W., Schadschneider, A. & Seyfried, A. (2012). Ordering in
   bidirectional pedestrian flows and its influence on the fundamental diagram.
   *J. Stat. Mech.*, P02002.
   -- Bidirectional flow, density effects, fundamental diagrams

7. JuPedSim documentation: https://www.jupedsim.org/stable/
   -- CollisionFreeSpeedModel, GCFM, SocialForceModel, example notebooks

8. PedPy (PedestrianDynamics/PedPy): https://github.com/PedestrianDynamics/PedPy
   -- Voronoi density, measurement lines, trajectory analysis toolkit

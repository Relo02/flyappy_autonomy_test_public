# MPC Path Planner and Probabilistic Grid Map — Implementation Report

This document summarizes the design and implementation details of the sampling-based Model Predictive Controller (MPC) used for the Flyappy agent, the Probabilistic Grid Map used for obstacle memory and inflation, mathematical formulation, observed results, plotting/diagnostics, and concluding remarks about trade-offs between precision and speed.

**Repository file:** [flyappy_ros.py](flyappy_autonomy_code_py/src/flyappy_autonomy_code/flyappy_ros.py)

---

## 1. High-level overview

- Goal: let the bird traverse an asteroid field as fast as possible in a fixed time window (one minute), passing through vertical gaps while avoiding collisions.
- Approach: sampling-based (random shooting) MPC that evaluates many candidate constant-acceleration rollouts over a finite horizon using a simple 2D point-mass dynamics model integrated with RK4. Obstacle memory and inflation are provided by a Probabilistic Grid Map that accumulates lidar scans over time.

Key components:
- Sampling & rollout generator (ax, ay samples per candidate)
- RK4 integrator for predicted states
- Per-rollout cost combining collision, boundary, effort, jerk, gap-attraction, obstacle-clearance and forward progress terms
- Probabilistic grid map maintained from lidar scans and used to extract inflated obstacles for collision checks and clearance costs
- Heuristics: gap queue, forward-block detection, perturbation (anti-stuck) nudges

---

## 2. Dynamics & numerical integration

We model the bird as a 2D point with state (x, y, vx, vy) and control as constant accelerations (ax, ay) and the continuous-time dynamics:

$$ \dot x = v_x, \quad \dot y = v_y $$
$$ \dot v_x = a_x, \quad \dot v_y = a_y $$

For a fixed acceleration control over a step of duration $\Delta t$, the numerical integrator used is RK4 applied to the position / velocity vectors. For each candidate path the integrator advances for $H$ steps (horizon length). The RK4 update used in code (vectorized) computes new position and velocity at each sub-step; the update is a 4th-order accurate single-step integrator.

---

## 3. Sampling & candidate rollouts

- Number of samples: `N` (e.g., 200).
- Horizon: `H` (e.g., 30 steps).
- Each candidate is a pair $(a_x, a_y)$ sampled from a biased Gaussian / uniform distribution. The biasing logic includes:
  - Ax sampled around a mean that moves the bird toward a `TARGET_VX` (favors forward motion).
  - Ay sampled around a mean computed from the vertical error to a targeted gap center when a gap is known.

Thus the sample distribution is flexible (allows exploration and exploitation). Sampling is clamped to safe accelerations.

---

## 4. Cost function

For each rollout we compute a scalar cost composed of multiple additive terms. Let $p_x^{(t)}, p_y^{(t)}, v_x^{(t)}, v_y^{(t)}$ be the predicted states at horizon step $t$ for a candidate control. Define $t=1..H$.

Main cost terms used (notations match code variable names):

- Collision penalty (hard death): large constant if any rollout point collides with an inflated obstacle.
- Boundary proximity penalty: penalize closeness to ground/ceiling when within safety margin.
- Backward penalty: penalize negative/very small forward position.
- Jerk penalty: penalize squared difference between candidate acceleration and previously commanded acceleration (to encourage smooth changes):

$$ J_{jerk} = w_{jerk} \cdot \|a - a_{prev}\|^2 $$

- Effort penalty: squared norm of accelerations to discourage wasteful control:

$$ J_{effort} = w_{effort} \cdot \|a\|^2 $$

- Speed tracking penalty for x: encourage $v_x$ near `TARGET_VX`:

$$ J_{vx} = w_{vx} (v_x - v_{target})^2 $$

- Gap attraction (vertical error): quadratic penalty to draw $p_y$ toward gap center $y_g$, optionally scaled by urgency based on forward distance to gap:

$$ J_{yerr} = w_{yerr} \cdot (p_y - y_g)^2 \cdot U(d_x) $$

where $U(d_x)$ is an urgency factor (e.g., $U(d_x)=\exp(-d_x/\tau)$ with $d_x$ forward distance to gap).

- Horizontal approach cost: quadratic penalty on $(p_x - x_g)^2$ scaled by urgency.

- Obstacle clearance soft cost: computed from the per-rollout minimum distance $\rho_{min}$ to any obstacle along the trajectory and penalized exponentially when close:

$$ J_{obs} = w_{obs} \cdot \exp\left(-\frac{\rho_{min}}{\sigma}\right) $$

This term encourages trajectories that keep larger clearance over the entire horizon; it is scaled more conservatively when no good gap is detected and relaxed when a suitable gap exists (so the controller can commit through the opening).

- Forward progress reward: negative cost encouraging positive advancement along X:

$$ J_{fwd} = -w_{fwd} \cdot p_x \cdot F(U(d_x)) $$

The total cost evaluated per candidate is the sum over relevant horizon quantities (with many terms evaluated at the final horizon state or aggregated across the horizon as implemented).

Candidate selection picks the control with minimum total cost.

---

## 5. Collision checking and probabilistic grid map

Given that the provided laser scans data are sparse, the planner uses a Probabilistic Grid Map (PGM) to maintain a memory of obstacles over time and provide inflated obstacle positions for collision checking. If we directly plug in the row laser scans, the bird may collide with obstacles since the scans are instantaneous and sparse, thus they don't capture enough enviroment morphology information. 
The Probabilistic Grid Map (PGM) stores occupancy belief on a local 2D grid and is updated from latest lidar scans, with scan points transformed in local coordinates using the current bird pose / velocity estimate. The PGM responsibilities:

- Fuse multiple scans over time to maintain memory of obstacles that temporarily leave the instantaneous scan (useful when passing behind obstacles or occlusions).
- Provide a set of occupied cells (positions) to the planner. These cells are inflated in the planner by an inflation radius (grid expansion) to account for the bird radius and motion uncertainty. In addition, the inflation helps to create more continuity in the obstacle representation, reducing the chance of skim collisions.

Collision checks for rollouts: compute vectorized distance from predicted positions to all inflated obstacles, take minimum distance per rollout point, then across the horizon compute $\rho_{min}$. Hard-collision occurs when $\rho_{min}$ falls below a collision radius (immediate large cost).

The planner also uses the grid map to find vertical gaps using a sweep over Y of occupied cells (`find_largest_gap`) and maintain a short queue of candidate gaps (`gap_queue`) with temporal TTL so the planner can commit to a gap for a good amount of time, without being influenced by further possible detected gaps.

---

## 6. Heuristics: forward-block detection & perturbation

To avoid getting stuck when forward lanes appear blocked by obstacles within the lidar FOV, the planner discretizes multiple forward X-bins and checks multiple vertical offsets (sub-bands) per bin. If a high fraction of bins are fully occupied, the agent considers the forward corridor blocked and activates a perturbation: a temporary Y acceleration nudge that explores upward or downward to find a passable channel.

This behavior is implemented as a diagnostic check that records a forward-block fraction and triggers the perturbation if it exceeds a threshold.

---

## 7. Implementation notes and parameter knobs

- The main implementation is in `flyappy_ros.py` and uses vectorized numpy operations for rollouts and RK4 integration.
- Important tunable parameters (examples used in experiments):
  - `HORIZON` (30), `NUM_PATHS` (200)
  - `SAFE_ACC_Y` / `MAX_ACC_Y` (2.5 m/s^2)
  - `JERK_WEIGHT` (20.0), `EFFORT_WEIGHT` (30.0), `Y_ERROR_WEIGHT` (60.0)
  - `OBSTACLE_COST_WEIGHT` (10.0), `COLLISION_RADIUS` (0.12)
  - `TARGET_VX` (tunable, example 0.3–1.0 m/s depending on aggressiveness)
  - Forward-block bins & offsets: `FWD_BIN_START`, `FWD_BIN_END`, `FWD_BIN_COUNT`, `FWD_Y_OFFSETS`

---

## 8. Results (observed behavior)

Short summary of experiments and observed outcomes:

- Baseline (conservative weights, high jerk penalty): the bird avoided rapid vertical changes.
- Reduced jerk penalty + small decay on `last_acc`, plus `TARGET_VX` bias: allowed the planner to "dump" vertical acceleration (accelerations decayed) and made the bird more responsive to gap attraction.
- Adding per-rollout obstacle clearance (`J_{obs}`) prevented risky skimming trajectories that previously led to collisions inside gaps.
- Forward-block detection + perturbation helped escape local minima where every immediate forward bin had obstacles; the bird began to explore in Y to find passable corridors outside current lidar FOV.

### Video proof:

```
![MPC Planner Demo](https://raw.githubusercontent.com/Relo02/flyappy_autonomy_test_public/master/flyappy_autonomy_code_py/gif/mpc_planner.gif)
```

## 9. Plotting & diagnostics

The node already produces several on-node plot elements (Matplotlib interactive):

- `occupied_plot`: raw grid-occupied cells
- `inflated_plot`: inflated obstacle positions used for collision checks
- `free_plot`: free-space samples
- `best_path_plot`: planner's selected trajectory over the horizon
- `bird_plot`: current bird pose (Y)
- `gap_plot`: selected gap center marker

To save the current figure from the running node, modify `flyappy_ros.py` to call `self.fig.savefig('results/current_viz.png')` at the desired point in the control loop, or use the interactive window menu to save.

Recommended diagnostic logs to enable while tuning (set rclpy logger level to DEBUG):

- `BestPathDebug` lines (min-distance along best path, obstacle cost, forward-block fraction, occupied bins)
- `GAP_PASSED` events to confirm gap traversal

Example: run the node and enable debug logging. Then after a short run generate plots of:

- Histogram of `min_dist_over_horizon` across chosen best paths
- Time-series of forward speed `v_x` and commanded `a_x` / `a_y`
- Scatter of inflated obstacles & best path overlay (saved figure)

---

## 10. Conclusions & trade-offs (precision vs speed)

For the project goal — maximizing forward speed while surviving for one minute and passing through asteroids — the planner must balance two competing objectives:

1. Precision / Safety: higher obstacle penalties, larger horizons, and conservative sampling favor safe trajectories with large clearances but often result in slower average forward speed because the bird avoids risky but shorter paths.
By choosing larger horizons with this MPC approach, the future propagated predictions would allow to explore in advance the possible paths and in principle select the cheeper one. However, witha a sample based MPC, increasing too much the hodizon will result in a sparser exploration of the action space, which can lead to less optimal paths without prioritizing clearance for guaranteed safety.

2. Speed / Aggressiveness: biasing `a_x` toward a larger `TARGET_VX`, lowering jerk/effort penalties, and reducing obstacle-cost scale let the bird commit to narrower gaps and travel faster, but increase collision risk especially if the grid map is noisy or the lidar FOV misses obstacles during the pass.

Summary: pushing for maximum speed requires careful risk-aware relaxation of obstacle penalties combined with robust gap detection and per-rollout clearance evaluation. The approach in `flyappy_ros.py` provides several levers (sampling bias, clearance cost, perturbation) to manage this trade-off.
The main drowback of this sampling-based MPC is that it can be sensible to local minima when the gap is not partially visible in the lidar FOV.
I've choosen sampling-based MPC for its good robustness against modeling errors, non-linearities and non-convex obstacle shapes, at the cost of computational efficiency and optimality guarantees. It is also well suited for real-time operation on limited hardware since it can be parallelized and vectorized easily.

---

## 11. Next steps & improvements

A good approach ti improve the planning performances, we can adopt RL techniques to learn a better planning policy that can guide the sampling distribution toward better candidate rollouts.

---

Authored:  Lorenzo Ortolani.

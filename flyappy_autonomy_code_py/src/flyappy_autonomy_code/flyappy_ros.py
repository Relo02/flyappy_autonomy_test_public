import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import LaserScan
from .probabilistic_grid_map import ProbabilisticGridMap

class FlyappyRos():
    """
    ROS 2 Node for Flyappy Bird autonomous control.
    
    Implementation Details:
    - Uses Model Predictive Control (MPC) via random shooting (sampling).
    - Integrates a Probabilistic Grid Map for temporal obstacle memory.
    - Features boundary-aware sampling and aggressive safety costs.
    """

    def __init__(self, node):
        self.node = node
        self.logger = self.node.get_logger()

        # --- Visualization Data ---
        # Store lidar points from the last 3 scans for visualization context
        self.lidar_history = []  

        # ==========================================
        # 1. MPC & Simulation Parameters
        # ==========================================
        self.HORIZON = 30           # Lookahead steps (longer helps reach gaps)
        self.DT = 0.033             # Time step (30Hz)
        self.NUM_PATHS = 200        # Number of candidate trajectories sampled
        
        # ==========================================
        # 2. Physical Constraints & Safety
        # ==========================================
        self.BIRD_RADIUS = 0.05         # 5cm physical radius
        self.OBSTACLE_INFLATION = 0.1   # Buffer zone around obstacles
        self.MAX_VX = 1.5               # Velocity cap for stability
        self.MIN_VX = -1.5
        
        # Sensor Thresholds
        self.MAX_OBSTACLE_RANGE = 2.0   # Ignore far obstacles to focus on immediate threat
        self.MIN_OBSTACLES_COUNT = 2.0

        # Boundary Constraints (Global Y coordinates)
        self.Y_GROUND = -1.30
        self.Y_CEILING = 2.40
        self.Y_SAFE_MARGIN = 0.5       # Buffer from ceiling/ground

        # Acceleration Limits (Conservative for survival)
        self.MAX_ACC_X = 1.5
        self.MAX_ACC_Y = 2.5            # Increase Y capability to reach gaps
        self.SAFE_ACC_X = self.MAX_ACC_X
        self.SAFE_ACC_Y = self.MAX_ACC_Y

        # ==========================================
        # 3. Cost Function Weights & Penalties
        # ==========================================
        self.COLLISION_COST = 100000.0      # Instant death penalty
        self.BOUNDARY_WEIGHT = 50000.0      # Stay away from floor/ceiling
        self.STOP_LIMIT = 0.1              # X threshold for backward penalty
        self.SAFETY_DIST = 1.0              # Hard safety halo, before 0.5
        
        # Tuning Weights
        # Jerk weight: lower so the controller can change acceleration (allows dumping)
        self.JERK_WEIGHT = 20.0
        # Effort: moderate penalty
        self.EFFORT_WEIGHT = 30.0
        # Vertical velocity penalty: keep moderate so reaching gaps is possible
        self.SPEED_Y_WEIGHT = 10.0
        self.SPEED_X_WEIGHT = 5.0           # Maintain forward momentum
        # Increase gap attraction to prioritize moving toward gap
        self.Y_ERROR_WEIGHT = 60.0
        # Horizontal approach weight for steering toward gap X position
        self.X_ERROR_WEIGHT = 20.0
        # Trajectory obstacle penalty (encourage larger clearance along rollouts)
        self.OBSTACLE_COST_WEIGHT = 10.0
        # Collision radius used for hard collision checks during rollouts
        self.COLLISION_RADIUS = 0.12
        # Reward for forward progress (reduces cost for trajectories that move forward)
        self.FORWARD_REWARD_WEIGHT = 2.0 # (before 5.0)
        # Forward-block detection parameters (discretize FOV in X and Y offsets)
        self.FWD_BIN_START = 0.2
        self.FWD_BIN_END = 1.4
        self.FWD_BIN_COUNT = 6
        self.FWD_BIN_HALF_WIDTH = 0.12
        self.FWD_Y_OFFSETS = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
        self.FWD_Y_WINDOW = 0.20
        self.FWD_BLOCKED_THRESHOLD = 0.75
        # Diagnostic state for forward-block checks
        self.last_forward_block_frac = 0.0
        self.last_occupied_bins = 0
        self.last_forward_bin_count = self.FWD_BIN_COUNT
        self.VX_MAX_WEIGHT = 1.0            # Speed limit penalty
        self.BACKWARD_PENALTY_WEIGHT = 2.0  
        self.BASE_REWARD = 1.0              
        self.TARGET_VX = 0.3                # Desired cruising speed (bias forward to reach gaps)

        # ==========================================
        # 4. State Management
        # ==========================================
        self.current_vel = np.array([0.0, 0.0])
        self.last_acc = np.array([0.0, 0.0])
        self.pos_y = 0.0
        self.pos_x = 0.0
        self.obstacles = np.empty((0, 2))
        self.scan_received = False
        self.last_scan_time = None

        # Navigation Heuristics (Gap & Stuck Detection)
        self.largest_gap_center = None  
        self.largest_gap_size = 0.0     
        self.gap_history = []           
        self.farthest_point_target = None
        self.LOW_SPEED_THRESHOLD = 0.70 

        # Gap queue: maintain ordered planned gaps. MPC targets queue head until reached.
        self.gap_queue = []  # list of dicts: {'center': np.array, 'size': float, 't': timestamp}
        self.GAP_QUEUE_MAX = 4
        self.GAP_TTL = 1.5          # seconds before a queued gap expires
        self.GAP_REACHED_Y = 0.05   # vertical threshold (m) to consider a gap reached
        self.GAP_REACHED_X = 0.10   # forward distance (m) threshold to consider gap reached
        self.GAP_SIMILAR_DIST = 0.25 # distance to consider two gaps the same
        self.GAP_MIN_ACCEPT_SIZE = 0.4  # if all gaps <= this, consider stuck and perturb
        self.GAP_WAIT_AFTER_REACHED = 3.0  # seconds to wait after a gap is first reached
        # Gap pass logging / state
        self.LOG_GAP_PASS = True
        self.gap_passed = False
        self.last_gap_pass_t = None
        self.GAP_PASS_TTL = 4.0     # seconds to keep `gap_passed` True after pass

        # Perturbation Logic (Anti-Stuck)
        self.min_obstacle_dist = float('inf')
        self.stuck_counter = 0
        self.STUCK_DIST_THRESHOLD = 0.0
        self.STUCK_VEL_THRESHOLD = 0.0
        self.STUCK_FRAMES_TRIGGER = 4
        self.perturbation_direction = 1
        self.perturbation_active = False
        self.perturbation_frames = 0
        self.PERTURBATION_DURATION = 12

        # ==========================================
        # 5. Mapping & Visualization
        # ==========================================
        self.grid_map = ProbabilisticGridMap(logger=self.logger)
        
        # Plotting Setup
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Graphical Elements
        self.occupied_plot, = self.ax.plot([], [], 'rs', markersize=4, alpha=0.5, label='Raw Obstacles')
        self.inflated_plot, = self.ax.plot([], [], 'mo', markersize=3, alpha=0.6, label='Inflated (collision)')
        self.free_plot, = self.ax.plot([], [], 'g.', markersize=2, alpha=0.3, label='Free Space')
        self.best_path_plot, = self.ax.plot([], [], 'b-', linewidth=3, label='Best Path', zorder=10)
        self.bird_plot, = self.ax.plot([0], [0], 'ko', markersize=10, label='Bird', zorder=11)
        self.gap_plot, = self.ax.plot([], [], 'y*', markersize=20, markeredgewidth=2, 
                                        markeredgecolor='orange', label='Target Gap', zorder=12)
        
        # Boundary Lines
        self.ground_line = self.ax.axhline(y=self.Y_GROUND, color='r', linestyle='--', linewidth=2, label='Ground')
        self.ceiling_line = self.ax.axhline(y=self.Y_CEILING, color='r', linestyle='--', linewidth=2, label='Ceiling')

        # Axis Config
        self.ax.set_xlim(-1, 5)
        self.ax.set_ylim(-2.5, 3.0)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right', fontsize=8)
        self.ax.set_title("Flyappy MPC with Probabilistic Grid Mapping")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        
        # ROS Interfaces
        self.pub_acc = self.node.create_publisher(Vector3, '/flyappy_acc', 10)
        self.node.create_subscription(Vector3, '/flyappy_vel', self.vel_callback, 10)
        self.node.create_subscription(LaserScan, '/flyappy_laser_scan', self.scan_callback, 10)
            
        self.timer = self.node.create_timer(self.DT, self.control_loop)
        self.logger.info("Flyappy MPC Initialized with Real-Time Plotter")

    def vel_callback(self, msg):
        """Updates bird state (velocity and approximate Y position integration)."""
        self.current_vel = np.array([msg.x, msg.y])
        self.pos_y += msg.y * self.DT
        self.pos_x += msg.x * self.DT

    def find_largest_gap(self, scan_points):
        """
        Heuristic to find the safest vertical opening.
        Uses a sweep-line approach sorted by Y coordinate.
        """
        if len(scan_points) == 0:
            # Default to center if clear
            return np.array([2.0, (self.Y_GROUND + self.Y_CEILING) / 2.0]), (self.Y_CEILING - self.Y_GROUND)

        # Filter bounds
        in_bounds = (scan_points[:, 1] >= self.Y_GROUND) & (scan_points[:, 1] <= self.Y_CEILING)
        points = scan_points[in_bounds]

        target_x = np.median(scan_points[:, 0]) if len(scan_points) > 0 else 2.0

        # Add virtual boundaries (min/max visible points)
        min_idx = np.argmin(scan_points[:, 1])
        max_idx = np.argmax(scan_points[:, 1])
        virtual_bounds = np.array([scan_points[min_idx], scan_points[max_idx]])

        if len(points) > 0:
            combined = np.vstack([points, virtual_bounds])
        else:
            combined = virtual_bounds

        # Sort by Y to find vertical gaps
        sorted_y = combined[np.argsort(combined[:, 1])]

        max_gap = 0.0
        center = None
        for i in range(len(sorted_y) - 1):
            g = sorted_y[i+1, 1] - sorted_y[i, 1]
            if g > max_gap:
                max_gap = g
                center = np.array([target_x + 0.2, (sorted_y[i+1, 1] + sorted_y[i, 1]) / 2.0])

        return center, max_gap

    def is_gap_similar(self, a, b):
        """Return True if two gap centers are within a small distance."""
        if a is None or b is None:
            return False
        return np.linalg.norm(np.array(a) - np.array(b)) < self.GAP_SIMILAR_DIST

    def is_gap_reached(self, gap_center):
        """Return True if the bird's Y is close enough to the gap center Y."""
        if gap_center is None:
            return False
        # Require both vertical proximity and that the gap is close enough in X (in front of the bird)
        y_ok = abs(self.pos_y - gap_center[1]) < self.GAP_REACHED_Y
        x_ok = (self.pos_x - gap_center[0] < self.GAP_REACHED_X)
        return y_ok and x_ok

    def update_gap_queue(self, center, size, timestamp):
        """Maintain an ordered queue of gaps (FIFO). Merge similar gaps and drop expired ones."""
        # Drop invalid center
        if center is None:
            return

        # Purge expired entries first
        self.gap_queue = [g for g in self.gap_queue if (timestamp - g['t']) < self.GAP_TTL]

        # If similar to an existing queued gap, refresh/merge it
        for g in self.gap_queue:
            if self.is_gap_similar(g['center'], center):
                g['t'] = timestamp
                g['size'] = max(g['size'], float(size))
                return

        # Otherwise append new gap to the queue
        self.gap_queue.append({'center': np.array(center), 'size': float(size), 't': timestamp})

        # Keep queue bounded
        if len(self.gap_queue) > self.GAP_QUEUE_MAX:
            self.gap_queue.pop(0)

    def scan_callback(self, msg):
        """
        Processes LiDAR data:
        1. Filters invalid/max-range points.
        2. Updates Probabilistic Grid Map.
        3. Inflates obstacles for collision checking.
        """
        full_ranges = np.array(msg.ranges)
        full_angles = np.linspace(msg.angle_min, msg.angle_max, len(full_ranges))
        current_time = self.node.get_clock().now().nanoseconds / 1e9

        # DT Calculation
        if self.last_scan_time is not None:
            dt = current_time - self.last_scan_time
        else:
            dt = self.DT 
        self.last_scan_time = current_time

        # Filter: Only keep points within valid range and outside bird radius
        valid_mask = (np.isfinite(full_ranges) &
                      (full_ranges > self.BIRD_RADIUS) &
                      (full_ranges < self.MAX_OBSTACLE_RANGE))

        if np.any(valid_mask):
            valid_ranges = full_ranges[valid_mask]
            valid_angles = full_angles[valid_mask]

            # Filter out empty space (near max range)
            obstacle_mask = valid_ranges < (self.MAX_OBSTACLE_RANGE * 0.9)

            if np.any(obstacle_mask):
                raw_scan_points = np.column_stack((
                    valid_ranges[obstacle_mask] * np.cos(valid_angles[obstacle_mask]),
                    valid_ranges[obstacle_mask] * np.sin(valid_angles[obstacle_mask])
                ))
            else:
                raw_scan_points = np.empty((0, 2))
        else:
            raw_scan_points = np.empty((0, 2))
        
        # Maintain visualization history
        self.lidar_history.append(raw_scan_points.copy())
        if len(self.lidar_history) > 3:
            self.lidar_history.pop(0)

        # Update Grid Map (Global Coordinates)
        self.grid_map.update_with_scan(raw_scan_points, self.current_vel, self.pos_y, current_time, dt)

        # Update Minimum Obstacle Distance
        if len(raw_scan_points) > 0:
            obstacle_dists = np.linalg.norm(raw_scan_points, axis=1)
            self.min_obstacle_dist = np.min(obstacle_dists)
        else:
            self.min_obstacle_dist = float('inf')

        # Retrieve & Inflate Obstacles from Grid Map
        grid_obstacles = self.grid_map.get_occupied_cells()
        
        if len(grid_obstacles) > 0:
            inflated = [grid_obstacles]
            # Expansion in 4 directions
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                offset = grid_obstacles.copy()
                offset[:, 0] += dx * self.OBSTACLE_INFLATION
                offset[:, 1] += dy * self.OBSTACLE_INFLATION
                inflated.append(offset)
            self.obstacles = np.vstack(inflated)
        else:
            self.obstacles = grid_obstacles.copy()

        # If forward path is blocked across several forward bins, trigger
        # perturbation so the bird searches in Y beyond the LIDAR FOV.
        try:
            if self.check_forward_blocked():
                if not self.perturbation_active:
                    # pick a direction away from the obstacle cluster
                    rel = self.obstacles.copy()
                    rel[:, 0] -= self.pos_x
                    front = rel[rel[:, 0] > 0]
                    mean_y = np.mean(front[:, 1]) if len(front) > 0 else (self.Y_GROUND + self.Y_CEILING) / 2.0
                    self.perturbation_direction = 1 if self.pos_y < mean_y else -1
                    self.perturbation_active = True
                    self.perturbation_frames = 0
                    self.logger.info(f"FORWARD BLOCK DETECTED: enabling perturbation dir={self.perturbation_direction}")
        except Exception:
            pass

        # Update Gap: compute current gap, push into queue, and maintain history
        current_gap_center, current_gap_size = self.find_largest_gap(grid_obstacles)

        # Update persistent queue of planned gaps
        self.update_gap_queue(current_gap_center, current_gap_size, current_time)

        # Maintain short-term history for fallback
        self.gap_history.append((current_gap_center, current_gap_size, grid_obstacles.copy()))
        if len(self.gap_history) > 4:
            self.gap_history.pop(0)

        # Purge expired queue entries (safety)
        self.gap_queue = [g for g in self.gap_queue if (current_time - g['t']) < self.GAP_TTL]

        # If head of queue is reached, mark reached time and only drop it after a wait interval
        while len(self.gap_queue) > 0:
            head = self.gap_queue[0]
            if self.is_gap_reached(head['center']):
                # set first-seen reached timestamp
                if 'reached_t' not in head:
                    head['reached_t'] = current_time
                    break
                # pop only if waited long enough
                if (current_time - head['reached_t']) >= self.GAP_WAIT_AFTER_REACHED:
                    # Mark gap as passed (we waited the required time while at the gap)
                    try:
                        gap_center = head.get('center', None)
                    except Exception:
                        gap_center = None
                    self.gap_passed = True
                    self.last_gap_pass_t = current_time
                    if self.LOG_GAP_PASS:
                        self.logger.info(f"GAP PASSED: center={gap_center} size={head.get('size',0):.2f} t={current_time:.2f}s")
                    self.gap_queue.pop(0)
                    continue
                else:
                    break
            else:
                # If gap is no longer reached, clear timestamp and stop
                if 'reached_t' in head:
                    del head['reached_t']
                break

        # Clear gap_passed after TTL so it only indicates recent pass
        if self.last_gap_pass_t is not None and (current_time - self.last_gap_pass_t) > self.GAP_PASS_TTL:
            self.gap_passed = False
            self.last_gap_pass_t = None

        # Prefer queue head as active target, fallback to history or current scan
        if len(self.gap_queue) > 0:
            head = self.gap_queue[0]
            self.largest_gap_center, self.largest_gap_size = head['center'], head['size']
        else:
            if len(self.gap_history) > 0:
                max_gap = max(self.gap_history, key=lambda g: g[1] if g[0] is not None else -np.inf)
                self.largest_gap_center, self.largest_gap_size = max_gap[0], max_gap[1]
            else:
                self.largest_gap_center, self.largest_gap_size = current_gap_center, current_gap_size

        # Stuck detection: if all known gaps (queue + history + current) are <= threshold
        sizes = []
        if current_gap_size is not None:
            sizes.append(float(current_gap_size))
        sizes.extend([g['size'] for g in self.gap_queue])
        sizes.extend([h[1] for h in self.gap_history if h[1] is not None])
        max_detected_gap = max(sizes) if len(sizes) > 0 else 0.0

        if max_detected_gap <= self.GAP_MIN_ACCEPT_SIZE:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            # clear perturbation when a valid gap appears
            self.perturbation_active = False
            self.perturbation_frames = 0

        # If persistently stuck, enable perturbation to explore in Y
        if self.stuck_counter > self.STUCK_FRAMES_TRIGGER:
            if not self.perturbation_active:
                # choose direction towards center to maximize chances
                mid_y = (self.Y_GROUND + self.Y_CEILING) / 2.0
                self.perturbation_direction = 1 if self.pos_y < mid_y else -1
                self.perturbation_active = True
                self.perturbation_frames = 0

        self.scan_received = True

        # Debug Logging
        if len(raw_scan_points) > 0 or len(self.obstacles) > 0:
            y_mask = (raw_scan_points[:, 1] >= self.Y_GROUND) & (raw_scan_points[:, 1] <= self.Y_CEILING)
            raw_scan_points = raw_scan_points[y_mask]
            stats = self.grid_map.get_map_statistics()
            
            scan_range = f"[{np.min(raw_scan_points[:, 1]):.2f}, {np.max(raw_scan_points[:, 1]):.2f}]" if len(raw_scan_points) > 0 else "[]"
            map_range = f"[{np.min(self.obstacles[:, 1]):.2f}, {np.max(self.obstacles[:, 1]):.2f}]" if len(self.obstacles) > 0 else "[]"

            self.logger.debug(
                f"Scan: {len(raw_scan_points)} pts Y:{scan_range} | "
                f"Grid: {stats['occupied']} occ, {stats['free']} free | Map Y:{map_range}"
            )

    def rk4_step(self, p, v, a, dt):
        """Runge-Kutta 4th Order Integration for System Dynamics."""
        k1_p = v
        k1_v = a
        
        k2_p = v + 0.5 * dt * k1_v
        k2_v = a
        
        k3_p = v + 0.5 * dt * k2_v
        k3_v = a
        
        k4_p = v + dt * k3_v
        k4_v = a
        
        new_p = p + (dt / 6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
        new_v = v + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        return new_p, new_v

    def check_forward_blocked(self):
        """Return True if forward discretized bins are mostly occupied.

        The method inspects `self.obstacles` relative to the bird position
        and checks a set of forward x-bins. If a high fraction of bins
        contain obstacles within a small lateral band around the bird's
        current y, we treat the forward corridor as blocked and return True.
        """
        if len(self.obstacles) == 0:
            return False

        # Relative obstacle coordinates (bird is at pos_x)
        rel = self.obstacles.copy()
        rel[:, 0] -= self.pos_x

        # Consider only obstacles in front
        front = rel[rel[:, 0] > 0]
        if len(front) == 0:
            return False

        # Discretize forward distances (meters from bird)
        bins = np.linspace(self.FWD_BIN_START, self.FWD_BIN_END, self.FWD_BIN_COUNT)

        occupied_bins = 0
        # For each forward bin, check multiple vertical offsets. A bin is
        # considered blocked only if ALL vertical offsets detect an obstacle
        # within the small Y window (i.e., no free sub-band in that bin).
        for bx in bins:
            offset_blocked = True
            for y_off in self.FWD_Y_OFFSETS:
                target_y = self.pos_y + y_off
                mask = (
                    (front[:, 0] >= (bx - self.FWD_BIN_HALF_WIDTH)) &
                    (front[:, 0] <= (bx + self.FWD_BIN_HALF_WIDTH)) &
                    (np.abs(front[:, 1] - target_y) <= self.FWD_Y_WINDOW)
                )
                if not np.any(mask):
                    # this vertical offset has free space in this bin
                    offset_blocked = False
                    break
            if offset_blocked:
                occupied_bins += 1

        # If >= threshold of bins are blocked, consider forward corridor blocked
        frac = occupied_bins / float(len(bins))
        # Save diagnostics for logging
        self.last_forward_block_frac = frac
        self.last_occupied_bins = int(occupied_bins)
        self.last_forward_bin_count = int(len(bins))
        return frac >= self.FWD_BLOCKED_THRESHOLD

    def control_loop(self):
        """Main MPC Control Loop."""
        if not self.scan_received:
            return

        # Lightly decay previous acceleration to allow commanded accelerations to ``dump'' over time
        self.last_acc *= 0.7

        # ==========================================
        # 1. Action Sampling (Random Shooting)
        # ==========================================
        # Biased sampling based on gap target when available, otherwise fallback to positional biases
        # Bias forward acceleration sampling toward reaching the desired `TARGET_VX`.
        ax_mean = np.clip((self.TARGET_VX - self.current_vel[0]) * 1.5, -self.SAFE_ACC_X, self.SAFE_ACC_X)
        ax_samples = np.clip(
            np.random.normal(ax_mean, self.SAFE_ACC_X * 0.6, self.NUM_PATHS),
            -self.SAFE_ACC_X, self.SAFE_ACC_X
        )

        if self.largest_gap_center is not None:
            # Bias vertical acceleration mean toward the gap center (proportional to vertical error)
            gap_y = float(self.largest_gap_center[1])
            gap_diff = gap_y - self.pos_y
            mean_bias = np.clip(gap_diff * 1.2, -self.SAFE_ACC_Y, self.SAFE_ACC_Y)
            ay_samples = np.clip(
                np.random.normal(mean_bias, self.SAFE_ACC_Y * 0.45, self.NUM_PATHS),
                -self.SAFE_ACC_Y, self.SAFE_ACC_Y
            )
        elif self.pos_y < (self.Y_GROUND + 0.2):
            # Biased Upward near ground
            ay_samples = np.random.uniform(0.0, self.SAFE_ACC_Y, self.NUM_PATHS)
        elif self.pos_y > (self.Y_CEILING - 0.2):
            # Biased Downward near ceiling
            ay_samples = np.random.uniform(-self.SAFE_ACC_Y, 0.0, self.NUM_PATHS)
        else:
            # Gaussian centered at 0 (Smooth Flight)
            ay_samples = np.clip(
                np.random.normal(0.0, self.SAFE_ACC_Y * 0.4, self.NUM_PATHS),
                -self.SAFE_ACC_Y, self.SAFE_ACC_Y
            )

        controls = np.column_stack((ax_samples, ay_samples))

        # ==========================================
        # 2. Trajectory Prediction
        # ==========================================
        all_paths_x = np.zeros((self.NUM_PATHS, self.HORIZON))
        all_paths_y = np.zeros((self.NUM_PATHS, self.HORIZON))

        p_x, p_y = np.zeros(self.NUM_PATHS), np.full(self.NUM_PATHS, self.pos_y)
        v_x, v_y = np.full(self.NUM_PATHS, self.current_vel[0]), np.full(self.NUM_PATHS, self.current_vel[1])
        costs = np.zeros(self.NUM_PATHS)
        collision_mask = np.zeros(self.NUM_PATHS, dtype=bool)

        # Track the minimum obstacle distance encountered along each rollout
        min_dist_over_horizon = np.full(self.NUM_PATHS, np.inf)

        for t in range(self.HORIZON):
            # Vectorized Integration
            p_pts_current = np.column_stack((p_x, p_y))
            v_pts_current = np.column_stack((v_x, v_y))
            
            new_p, new_v = self.rk4_step(p_pts_current, v_pts_current, controls, self.DT)
            
            p_x, p_y = new_p[:, 0], new_p[:, 1]
            v_x, v_y = new_v[:, 0], new_v[:, 1]
            
            all_paths_x[:, t] = p_x
            all_paths_y[:, t] = p_y

            # Boundary Checks
            p_pts = np.column_stack((p_x, p_y))
            collision_mask |= (p_y > self.Y_CEILING) | (p_y < self.Y_GROUND)

            # Obstacle Checks
            if len(self.obstacles) > 0:
                # Compensate for relative velocity (shift obstacles)
                obs_shifted = self.obstacles.copy()
                obs_shifted[:, 0] -= (self.current_vel[0] * self.DT * t)

                # Vectorized distance check
                diff = p_pts[:, np.newaxis, :] - obs_shifted[np.newaxis, :, :]
                dist_sq = np.sum(diff**2, axis=2)
                min_dist_sq = np.min(dist_sq, axis=1)
                min_dist = np.sqrt(min_dist_sq)

                # update per-path minimum distance seen over the horizon
                min_dist_over_horizon = np.minimum(min_dist_over_horizon, min_dist)

                # Hard collision checks using class-level collision radius
                collision_mask |= (min_dist < self.COLLISION_RADIUS)

                hard_collision = min_dist < self.SAFETY_DIST
                collision_mask |= hard_collision

        # ==========================================
        # 3. Cost Evaluation
        # ==========================================
        costs[collision_mask] += self.COLLISION_COST

        # Backward motion penalty
        backward_mask = p_x <= self.STOP_LIMIT
        costs[backward_mask] += np.abs(p_x[backward_mask]) * self.BACKWARD_PENALTY_WEIGHT

        # Perturbation Reward (when active)
        if self.perturbation_active:
            y_movement = p_y - self.pos_y
            desired_movement = y_movement * self.perturbation_direction
            perturbation_reward = desired_movement * 60.0
            costs -= perturbation_reward

        # Gap Attraction (Commented out in original, kept for structure)
        # default urgency (0..1). When a gap is present it's set below.
        urgency_multiplier = 0.0
        if self.largest_gap_center is not None:
            gap_x, gap_y = self.largest_gap_center[0], self.largest_gap_center[1]
            dist_x = max(0.1, gap_x)
            urgency_multiplier = np.exp(-dist_x / 1.5)
            y_error_sq = (p_y - gap_y)**2
            # Apply urgency multiplier based on approximate forward distance to gap
            gap_dist_sq = self.Y_ERROR_WEIGHT * y_error_sq * urgency_multiplier
            costs += gap_dist_sq
            # Encourage forward progress toward the gap X coordinate (at horizon)
            x_error_sq = (p_x - gap_x)**2
            costs += urgency_multiplier * self.X_ERROR_WEIGHT * x_error_sq

        # Target Speed Penalty
        costs += (v_x - self.TARGET_VX)**2 * self.SPEED_X_WEIGHT

        # Boundary Proximity Penalties (Soft Constraints)
        ground_dist = p_y - self.Y_GROUND
        ceiling_dist = self.Y_CEILING - p_y
        ground_violation = np.maximum(0, self.Y_SAFE_MARGIN - ground_dist)
        ceiling_violation = np.maximum(0, self.Y_SAFE_MARGIN - ceiling_dist)

        costs += (ground_violation**2) * self.BOUNDARY_WEIGHT
        costs += (ceiling_violation**2) * self.BOUNDARY_WEIGHT

        # Velocity & Control Effort Penalties
        overspeed_mask = v_x > self.MAX_VX
        costs[overspeed_mask] += (v_x[overspeed_mask] - self.MAX_VX)**2 * self.VX_MAX_WEIGHT
        costs += (v_y**2) * self.SPEED_Y_WEIGHT
        
        jerk = np.sum((controls - self.last_acc)**2, axis=1)
        costs += jerk * self.JERK_WEIGHT
        
        effort = np.sum(controls**2, axis=1)
        costs += effort * self.EFFORT_WEIGHT

        # Obstacle clearance cost: penalize rollouts that come close to obstacles
        # Smaller min distances -> exponentially larger penalty. Scale penalty when
        # committing to a gap (urgency_multiplier near 1) so the planner is more
        # conservative during gap passes.
        obstacle_cost = self.OBSTACLE_COST_WEIGHT * np.exp(-min_dist_over_horizon / 0.15)
        # If the detected gap is large enough, reduce obstacle penalty scale so
        # the planner can commit through the free opening. Otherwise remain
        # conservative and penalize close passes heavily.
        if self.largest_gap_size is not None and self.largest_gap_size > self.GAP_MIN_ACCEPT_SIZE:
            obstacle_scale = 1.0 + 0.4 * urgency_multiplier
        else:
            obstacle_scale = 1.0 + 1.5 * urgency_multiplier
        costs += obstacle_cost * obstacle_scale

        # Reward forward progress: lower cost for trajectories that advance in X.
        # Scale reward by urgency so we encourage forward commitment when approaching a gap.
        forward_scale = 0.5 + 0.5 * urgency_multiplier
        forward_reward = - self.FORWARD_REWARD_WEIGHT * (p_x * forward_scale)
        costs += forward_reward

        # Select Best Trajectory
        best_idx = np.argmin(costs)
        
        # ==========================================
        # 4. Visualization
        # ==========================================
        if self.node.get_clock().now().nanoseconds % 5 == 0: 
            viz_data = self.grid_map.get_visualization_data()

            if len(viz_data['occupied']) > 0:
                self.occupied_plot.set_data(viz_data['occupied'][:, 0], viz_data['occupied'][:, 1])
            else:
                self.occupied_plot.set_data([], [])

            if len(self.obstacles) > 0:
                self.inflated_plot.set_data(self.obstacles[:, 0], self.obstacles[:, 1])
            else:
                self.inflated_plot.set_data([], [])

            if len(viz_data['free']) > 0:
                self.free_plot.set_data(viz_data['free'][:, 0], viz_data['free'][:, 1])
            else:
                self.free_plot.set_data([], [])

            self.best_path_plot.set_data(all_paths_x[best_idx], all_paths_y[best_idx])
            self.bird_plot.set_data([0], [self.pos_y])

            if self.largest_gap_center is not None:
                self.gap_plot.set_data([self.largest_gap_center[0]], [self.largest_gap_center[1]])
            else:
                self.gap_plot.set_data([], [])

            # Target Info Logging
            if self.perturbation_active:
                dir_str = "UP" if self.perturbation_direction > 0 else "DOWN"
                target_str = f"PERTURB_{dir_str}({self.perturbation_frames}/{self.PERTURBATION_DURATION})"
            elif self.farthest_point_target is not None and self.current_vel[0] < self.LOW_SPEED_THRESHOLD:
                target_str = f"FarAttr@Y={self.farthest_point_target[1]:.2f}"
            elif self.largest_gap_center is not None:
                target_str = f"VGap@Y={self.largest_gap_center[1]:.2f}(h={self.largest_gap_size:.2f}m)"
            else:
                target_str = "Center"

            stuck_str = f" Stuck:{self.stuck_counter}" if self.stuck_counter > 3 else ""
            self.logger.info(
                f"Y: {self.pos_y:.2f} Vx: {self.current_vel[0]:.2f} MinObs:{self.min_obstacle_dist:.2f}{stuck_str} | {target_str} | "
                f"Acc: [{controls[best_idx][0]:.2f}, {controls[best_idx][1]:.2f}] | Cost: {costs[best_idx]:.1f}"
            )

            # Detailed debug diagnostics for selected trajectory and forward-block status
            try:
                self.logger.debug(
                    f"BestPathDebug: minDist={min_dist_over_horizon[best_idx]:.3f} "
                    f"obsCost={obstacle_cost[best_idx]:.3f} "
                    f"fwdBlockedFrac={self.last_forward_block_frac:.2f} "
                    f"occBins={self.last_occupied_bins}/{self.last_forward_bin_count}"
                )
            except Exception:
                pass

            if self.gap_passed:
                try:
                    self.logger.debug(f"GAP_PASSED active t={self.last_gap_pass_t:.2f}")
                except Exception:
                    self.logger.debug("GAP_PASSED active")

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        # ==========================================
        # 5. Command Smoothing & Execution
        # ==========================================
        best_acc = controls[best_idx]

        # Emergency Mode: Increase responsiveness near boundaries
        CRITICAL_DISTANCE = 0.20
        ground_dist = self.pos_y - self.Y_GROUND
        ceiling_dist = self.Y_CEILING - self.pos_y

        if ground_dist < CRITICAL_DISTANCE or ceiling_dist < CRITICAL_DISTANCE:
            alpha = 0.6  # Emergency: faster response
            self.logger.warn(f"EMERGENCY MODE: Ground={ground_dist:.3f}m, Ceiling={ceiling_dist:.3f}m")
        else:
            alpha = 0.5  # Normal: smoother response (increased to allow faster dumping)

        # Low-pass filter
        final_acc = (alpha * best_acc) + ((1.0 - alpha) * self.last_acc)

        # If perturbation is active, nudge the Y acceleration to explore for gaps
        if self.perturbation_active:
            PERTURB_ACC = 0.25
            final_acc[1] += PERTURB_ACC * self.perturbation_direction
            # Progress frame counter and flip direction periodically to avoid bias
            self.perturbation_frames += 1
            if self.perturbation_frames > self.PERTURBATION_DURATION:
                self.perturbation_direction *= -1
                self.perturbation_frames = 0

        # Safety Clamping
        final_acc[0] = np.clip(final_acc[0], -self.SAFE_ACC_X, self.SAFE_ACC_X)
        final_acc[1] = np.clip(final_acc[1], -self.SAFE_ACC_Y, self.SAFE_ACC_Y)

        if not np.allclose(final_acc, (alpha * best_acc) + ((1.0 - alpha) * self.last_acc), atol=1e-6):
            unclamped = (alpha * best_acc) + ((1.0 - alpha) * self.last_acc)
            self.logger.warn(f"CLAMPED: {unclamped} -> {final_acc}")

        self.last_acc = final_acc

        cmd = Vector3()
        cmd.x, cmd.y = float(final_acc[0]), float(final_acc[1])
        self.pub_acc.publish(cmd)
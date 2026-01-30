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
        self.HORIZON = 25           # Lookahead steps (shorter = more reactive)
        self.DT = 0.033             # Time step (30Hz)
        self.NUM_PATHS = 200        # Number of candidate trajectories sampled
        
        # ==========================================
        # 2. Physical Constraints & Safety
        # ==========================================
        self.BIRD_RADIUS = 0.05         # 5cm physical radius
        self.OBSTACLE_INFLATION = 0.1   # Buffer zone around obstacles
        self.MAX_VX = 1.5               # Velocity cap for stability
        self.MIN_VX = 0.0
        
        # Sensor Thresholds
        self.MAX_OBSTACLE_RANGE = 5.0   # Ignore far obstacles to focus on immediate threat
        self.MIN_OBSTACLES_COUNT = 2.0

        # Boundary Constraints (Global Y coordinates)
        self.Y_GROUND = -1.30
        self.Y_CEILING = 2.40
        self.Y_SAFE_MARGIN = 0.15       # Buffer from ceiling/ground

        # Acceleration Limits (Conservative for survival)
        self.MAX_ACC_X = 0.5
        self.MAX_ACC_Y = 0.5            # Low Y acc prevents oscillation
        self.SAFE_ACC_X = self.MAX_ACC_X
        self.SAFE_ACC_Y = self.MAX_ACC_Y

        # ==========================================
        # 3. Cost Function Weights & Penalties
        # ==========================================
        self.COLLISION_COST = 100000.0      # Instant death penalty
        self.BOUNDARY_WEIGHT = 50000.0      # Stay away from floor/ceiling
        self.STOP_LIMIT = 0.05              # X threshold for backward penalty
        self.SAFETY_DIST = 0.5              # Hard safety halo
        
        # Tuning Weights
        self.JERK_WEIGHT = 10.0             # Smoothness is priority
        self.EFFORT_WEIGHT = 10.0           # Minimize energy usage
        self.SPEED_Y_WEIGHT = 2.0           # Penalize vertical velocity (damping)
        self.SPEED_X_WEIGHT = 5.0           # Maintain forward momentum
        self.Y_ERROR_WEIGHT = 10.0          # Gap attraction strength
        self.VX_MAX_WEIGHT = 1.0            # Speed limit penalty
        self.BACKWARD_PENALTY_WEIGHT = 2.0  
        self.BASE_REWARD = 1.0              
        self.TARGET_VX = 0.3                # Desired cruising speed

        # ==========================================
        # 4. State Management
        # ==========================================
        self.current_vel = np.array([0.0, 0.0])
        self.last_acc = np.array([0.0, 0.0])
        self.pos_y = 0.0
        self.obstacles = np.empty((0, 2))
        self.scan_received = False
        self.last_scan_time = None

        # Navigation Heuristics (Gap & Stuck Detection)
        self.largest_gap_center = None  
        self.largest_gap_size = 0.0     
        self.gap_history = []           
        self.farthest_point_target = None
        self.LOW_SPEED_THRESHOLD = 0.70 

        # Perturbation Logic (Anti-Stuck)
        self.min_obstacle_dist = float('inf')
        self.stuck_counter = 0
        self.STUCK_DIST_THRESHOLD = 0.0
        self.STUCK_VEL_THRESHOLD = 0.0
        self.STUCK_FRAMES_TRIGGER = 8
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
                center = np.array([target_x, (sorted_y[i+1, 1] + sorted_y[i, 1]) / 2.0])

        return center, max_gap

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

        # Update Gap History
        current_gap_center, current_gap_size = self.find_largest_gap(grid_obstacles)
        self.gap_history.append((current_gap_center, current_gap_size, grid_obstacles.copy()))

        if len(self.gap_history) > 4:
            self.gap_history.pop(0)
        
        if len(self.gap_history) > 0:
            max_gap = max(self.gap_history, key=lambda g: g[1] if g[0] is not None else -np.inf)
            self.largest_gap_center, self.largest_gap_size = max_gap[0], max_gap[1]
        else:
            self.largest_gap_center, self.largest_gap_size = current_gap_center, current_gap_size

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

    def control_loop(self):
        """Main MPC Control Loop."""
        if not self.scan_received:
            return

        # ==========================================
        # 1. Action Sampling (Random Shooting)
        # ==========================================
        # Biased sampling based on vertical position to avoid boundaries naturally
        ax_samples = np.random.uniform(-0.3, self.SAFE_ACC_X, self.NUM_PATHS)

        if self.pos_y < (self.Y_GROUND + 0.2):
            # Biased Upward
            ay_samples = np.random.uniform(0.0, self.SAFE_ACC_Y, self.NUM_PATHS)
        elif self.pos_y > (self.Y_CEILING - 0.2):
            # Biased Downward
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

                COLLISION_RADIUS = 0.1
                collision_mask |= (min_dist < COLLISION_RADIUS)
            
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
        if self.largest_gap_center is not None:
            gap_x, gap_y = self.largest_gap_center[0], self.largest_gap_center[1]
            dist_x = max(0.1, gap_x)
            urgency_multiplier = np.exp(-dist_x / 1.5)
            y_error_sq = (p_y - gap_y)**2
            gap_dist_sq = self.Y_ERROR_WEIGHT * y_error_sq
            costs += gap_dist_sq

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
            alpha = 0.5  # Emergency: faster response
            self.logger.warn(f"EMERGENCY MODE: Ground={ground_dist:.3f}m, Ceiling={ceiling_dist:.3f}m")
        else:
            alpha = 0.25  # Normal: smoother response

        # Low-pass filter
        final_acc = (alpha * best_acc) + ((1.0 - alpha) * self.last_acc)

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
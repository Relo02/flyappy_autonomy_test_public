import numpy as np
from collections import defaultdict


class ProbabilisticGridMap:
    """
    Probabilistic occupancy grid mapping using log-odds representation.

    - Maintains a 2D grid where each cell has an occupancy probability
    - Uses ray-tracing to mark free space along laser beams
    - Updates probabilities incrementally as new scans arrive
    - Suitable for sparse laser scans (~9 points)
    """

    def __init__(self, logger=None):
        self.logger = logger

        # Grid parameters
        self.GRID_RESOLUTION = 0.1  # 10cm cells
        self.MAP_SIZE_X = 60  # 6m ahead (in cells)
        self.MAP_SIZE_Y = 60  # 6m vertically (in cells)
        self.MAX_MAP_RAYS = 2.5 # before 1.5
        self.MIN_MAP_RAYS = -2.5 # before -1.5

        # Game boundary constraints (global Y coordinates)
        # These must match the game geometry: ground=-1.35m, ceiling=+2.44m
        self.Y_GROUND = -1.30   # Ground collision boundary (with small margin)
        self.Y_CEILING = 2.40   # Ceiling collision boundary (with small margin)

        # Occupancy grid: dict of (gx, gy) -> log_odds
        # Using dict for sparse storage (only store cells we've observed)
        self.grid = defaultdict(float)  # Default log_odds = 0.0 (50% probability)

        # Accumulated displacement for sub-grid shifting (only X is used now)
        self.accumulated_displacement = np.array([0.0, 0.0])

        # Log-odds parameters (MORE RESPONSIVE to new data)
        self.LOG_ODDS_OCC = 2.5    # INCREASED: Single observation reaches ~92% probability
        self.LOG_ODDS_FREE = -1.0  # Clear free space
        self.LOG_ODDS_MIN = -3.0   # Faster to forget "very free", before: 3
        self.LOG_ODDS_MAX = 5.0    # Allow higher confidence for occupied

        # Probability thresholds for queries
        self.OCCUPIED_THRESHOLD = 0.5   # Lower threshold - 70% confidence marks as occupied
        self.FREE_THRESHOLD = 0.3       # Consider free if P < 0.3

        # Time-based decay and timeouts
        self.cell_timestamps = defaultdict(float)
        self.DECAY_RATE = 0.8      # Slower decay to keep obstacles visible longer
        self.DECAY_ENABLED = True
        self.CELL_TIMEOUT = 1.0    # 1 second - keep obstacles longer

        # Track last scan time for freshness detection
        self.last_update_time = None

    def _log_odds_to_prob(self, log_odds):
        """Convert log-odds to probability: P = 1 / (1 + exp(-log_odds))"""
        return 1.0 / (1.0 + np.exp(-log_odds))

    def _prob_to_log_odds(self, prob):
        """Convert probability to log-odds: log_odds = log(P / (1 - P))"""
        prob = np.clip(prob, 1e-6, 1.0 - 1e-6)  # Avoid division by zero
        return np.log(prob / (1.0 - prob))

    def _bresenham_ray(self, x0, y0, x1, y1):
        """
        Bresenham's line algorithm - returns all grid cells along ray.

        Args:
            x0, y0: Start position (bird position)
            x1, y1: End position (obstacle hit point)

        Returns:
            List of (gx, gy) tuples along the ray
        """
        # Convert to grid coordinates
        gx0 = int(np.round(x0 / self.GRID_RESOLUTION))
        gy0 = int(np.round(y0 / self.GRID_RESOLUTION))
        gx1 = int(np.round(x1 / self.GRID_RESOLUTION))
        gy1 = int(np.round(y1 / self.GRID_RESOLUTION))

        cells = []

        dx = abs(gx1 - gx0)
        dy = abs(gy1 - gy0)
        sx = 1 if gx0 < gx1 else -1
        sy = 1 if gy0 < gy1 else -1
        err = dx - dy

        gx, gy = gx0, gy0

        while True:
            cells.append((gx, gy))

            if gx == gx1 and gy == gy1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                gx += sx
            if e2 < dx:
                err += dx
                gy += sy

        return cells

    def update_with_scan(self, scan_points, current_velocity, current_y, current_time, dt):
        """
        Update the occupancy grid with a new laser scan.

        The grid uses GLOBAL Y coordinates to prevent vertical drift.
        X is shifted as the bird moves forward (scrolling compensation).
        Y is converted from local (sensor) to global (world) coordinates.

        Args:
            scan_points: Nx2 array of obstacle hit points [x, y] in LOCAL (sensor) frame
            current_velocity: [vx, vy] bird velocity
            current_y: Bird's current Y position in world frame
            current_time: Current timestamp (seconds)
            dt: Time since last update
        """
        # Shift grid based on bird X motion only (horizontal scroll)
        displacement = current_velocity * dt
        self._shift_grid(displacement)

        # Store current bird Y for ray tracing (bird position in global frame)
        bird_global_y = current_y

        # Track which cells were updated this scan (for freshness)
        cells_updated_this_scan = set()

        # Process each laser ray
        for obs_point in scan_points:
            # Local sensor coordinates (relative to bird)
            local_x, local_y = obs_point[0], obs_point[1]

            # Convert to GLOBAL Y coordinate
            # The obstacle's global Y = bird's global Y + local sensor Y
            global_y = local_y + bird_global_y

            # Ray-trace from bird position (0, bird_global_y) to obstacle (local_x, global_y)
            # Note: X is always relative (0 = bird position), Y is global
            ray_cells = self._bresenham_ray(0.0, bird_global_y, local_x, global_y)

            # Mark all cells along ray as FREE (except last one)
            for i, (gx, gy) in enumerate(ray_cells[:-1]):
                # Update log-odds (more confident it's free)
                self.grid[(gx, gy)] = np.clip(
                    self.grid[(gx, gy)] + self.LOG_ODDS_FREE,
                    self.LOG_ODDS_MIN,
                    self.LOG_ODDS_MAX
                )
                self.cell_timestamps[(gx, gy)] = current_time
                cells_updated_this_scan.add((gx, gy))

            # Mark the last cell as OCCUPIED
            if len(ray_cells) > 0:
                gx_hit, gy_hit = ray_cells[-1]
                self.grid[(gx_hit, gy_hit)] = np.clip(
                    self.grid[(gx_hit, gy_hit)] + self.LOG_ODDS_OCC,
                    self.LOG_ODDS_MIN,
                    self.LOG_ODDS_MAX
                )
                self.cell_timestamps[(gx_hit, gy_hit)] = current_time
                cells_updated_this_scan.add((gx_hit, gy_hit))

        # Detect stale data: if a long time has passed since last update,
        # aggressively clear nearby cells that weren't updated
        if self.last_update_time is not None:
            time_gap = current_time - self.last_update_time
            if time_gap > 0.5:  # >0.5s gap suggests stale data
                self._clear_stale_nearby_cells(current_time, cells_updated_this_scan)

        self.last_update_time = current_time

        # Optional: Apply time-based decay to all cells
        if self.DECAY_ENABLED:
            self._apply_decay(current_time, dt)

        # Cleanup: Remove cells outside the local map region and beyond boundaries
        self._cleanup_distant_cells()

    def _clear_stale_nearby_cells(self, current_time, updated_cells):
        """
        Clear occupied cells near the bird that weren't updated in the current scan.
        This handles the case where the bird moves to a new position and old
        obstacles should no longer be visible.

        Args:
            current_time: Current timestamp
            updated_cells: Set of (gx, gy) cells updated in this scan
        """
        cells_to_clear = []

        for (gx, gy) in self.grid.keys():
            # Only consider nearby cells (within 2m ahead, 1m behind)
            x = gx * self.GRID_RESOLUTION
            if x < -1.0 or x > 2.0:
                continue

            # Skip cells that were just updated
            if (gx, gy) in updated_cells:
                continue

            # Check if this is an occupied cell that's now stale
            log_odds = self.grid[(gx, gy)]
            if log_odds > 0:  # Occupied cell
                age = current_time - self.cell_timestamps.get((gx, gy), 0)
                if age > 0.15:  # Hasn't been confirmed in 0.15s - clear fast
                    cells_to_clear.append((gx, gy))

        # Decay these cells aggressively rather than deleting immediately
        for cell in cells_to_clear:
            self.grid[cell] *= 0.5  # Halve the log-odds

    def _shift_grid(self, displacement):
        """
        Shift grid when bird moves - ONLY in X direction (horizontal scroll).

        The Y-axis is kept in global coordinates because the game's vertical
        boundaries (ground/ceiling) are fixed. Shifting Y would cause obstacles
        to drift vertically, creating the "slanting" effect.

        Args:
            displacement: [dx, dy] bird displacement (meters) - only dx is used
        """
        # ONLY accumulate X displacement (horizontal scroll compensation)
        self.accumulated_displacement[0] += displacement[0]
        # Do NOT accumulate Y - we use global Y coordinates

        # Convert accumulated X displacement to grid units
        dgx = int(np.round(self.accumulated_displacement[0] / self.GRID_RESOLUTION))

        # Only shift if accumulated X displacement >= 1 grid cell
        if dgx == 0:
            return

        # Log the shift for debugging
        if self.logger:
            self.logger.debug(f"Grid shift: ({dgx}, 0) cells X, accumulated: {self.accumulated_displacement[0]:.4f}m")

        # Create new shifted grid - ONLY shift X coordinate
        new_grid = defaultdict(float)
        new_timestamps = defaultdict(float)

        for (gx, gy), log_odds in self.grid.items():
            new_gx = gx - dgx
            # Keep gy unchanged - Y is in global coordinates
            new_grid[(new_gx, gy)] = log_odds

            if (gx, gy) in self.cell_timestamps:
                new_timestamps[(new_gx, gy)] = self.cell_timestamps[(gx, gy)]

        self.grid = new_grid
        self.cell_timestamps = new_timestamps

        # Subtract the shifted amount from accumulated X displacement
        self.accumulated_displacement[0] -= dgx * self.GRID_RESOLUTION

    def _apply_decay(self, current_time, dt):
        """
        Apply time-based decay: old observations fade toward unknown (log_odds -> 0).
        Also remove cells that exceed timeout.

        This is CRITICAL for preventing stale scan data from persisting when
        the bird moves to new positions where old obstacles are no longer visible.

        Args:
            current_time: Current timestamp
            dt: Time step
        """
        cells_to_update = list(self.grid.keys())

        for cell in cells_to_update:
            age = current_time - self.cell_timestamps.get(cell, current_time)

            # Remove cells older than timeout (haven't been re-observed)
            # This clears obstacles that the bird has moved past or that
            # are no longer being scanned
            if age > self.CELL_TIMEOUT:
                if cell in self.grid:
                    del self.grid[cell]
                if cell in self.cell_timestamps:
                    del self.cell_timestamps[cell]
                continue

            # Exponential decay toward 0 (unknown) for remaining cells
            # Faster decay for occupied cells that haven't been re-confirmed
            log_odds = self.grid[cell]

            # Occupied cells (positive log_odds) decay faster if not refreshed
            if log_odds > 0 and age > 0.2:  # If occupied and >0.2s old
                decay_factor = np.exp(-self.DECAY_RATE * 3.0 * dt)  # 3x faster decay
            else:
                decay_factor = np.exp(-self.DECAY_RATE * dt)

            self.grid[cell] *= decay_factor

            # Remove cells that have decayed to near-zero (very uncertain)
            if abs(self.grid[cell]) < 0.1:
                del self.grid[cell]
                if cell in self.cell_timestamps:
                    del self.cell_timestamps[cell]

        # Track update time
        self.last_update_time = current_time

    def _cleanup_distant_cells(self):
        """
        Remove cells outside the local map region and beyond game boundaries.

        CRITICAL: Enforces hard boundaries to prevent obstacles from "leaking"
        below ground or above ceiling in the visualization.
        """
        cells_to_remove = []

        for (gx, gy) in self.grid.keys():
            # Convert to world coordinates
            x = gx * self.GRID_RESOLUTION
            y = gy * self.GRID_RESOLUTION

            # HARD BOUNDARY: Delete any cells outside game vertical limits
            # This prevents obstacles from drifting outside the playable area
            if y < self.Y_GROUND or y > self.Y_CEILING:
                cells_to_remove.append((gx, gy))
                continue

            # Keep cells within reasonable horizontal region
            # Behind: -1.5m (just passed), Ahead: +5m (upcoming obstacles)
            if x < self.MIN_MAP_RAYS or x > self.MAX_MAP_RAYS:
                cells_to_remove.append((gx, gy))

        for cell in cells_to_remove:
            del self.grid[cell]
            if cell in self.cell_timestamps:
                del self.cell_timestamps[cell]

    def get_occupied_cells(self):
        """
        Get all cells with high occupancy probability.

        Returns:
            Nx2 array of [x, y] positions of occupied cells
        """
        occupied = []

        for (gx, gy), log_odds in self.grid.items():
            prob = self._log_odds_to_prob(log_odds)

            if prob > self.OCCUPIED_THRESHOLD:
                x = gx * self.GRID_RESOLUTION
                y = gy * self.GRID_RESOLUTION
                occupied.append([x, y])

        if len(occupied) == 0:
            return np.empty((0, 2))

        return np.array(occupied)

    def is_occupied(self, position):
        """
        Check if a position is occupied.

        Args:
            position: [x, y] position to check

        Returns:
            bool: True if occupied (P > threshold)
        """
        gx = int(np.round(position[0] / self.GRID_RESOLUTION))
        gy = int(np.round(position[1] / self.GRID_RESOLUTION))

        log_odds = self.grid.get((gx, gy), 0.0)
        prob = self._log_odds_to_prob(log_odds)

        return prob > self.OCCUPIED_THRESHOLD

    def is_free(self, position):
        """
        Check if a position is free.

        Args:
            position: [x, y] position to check

        Returns:
            bool: True if free (P < threshold)
        """
        gx = int(np.round(position[0] / self.GRID_RESOLUTION))
        gy = int(np.round(position[1] / self.GRID_RESOLUTION))

        log_odds = self.grid.get((gx, gy), 0.0)
        prob = self._log_odds_to_prob(log_odds)

        return prob < self.FREE_THRESHOLD

    def get_occupancy_probability(self, position):
        """
        Get occupancy probability at a position.

        Args:
            position: [x, y] position to query

        Returns:
            float: Occupancy probability [0, 1]
        """
        gx = int(np.round(position[0] / self.GRID_RESOLUTION))
        gy = int(np.round(position[1] / self.GRID_RESOLUTION))

        log_odds = self.grid.get((gx, gy), 0.0)
        return self._log_odds_to_prob(log_odds)

    def get_map_statistics(self):
        """
        Get statistics about the current map.

        Returns:
            dict: Statistics (num_cells, num_occupied, num_free, num_unknown)
        """
        num_occupied = 0
        num_free = 0
        num_unknown = 0

        for log_odds in self.grid.values():
            prob = self._log_odds_to_prob(log_odds)

            if prob > self.OCCUPIED_THRESHOLD:
                num_occupied += 1
            elif prob < self.FREE_THRESHOLD:
                num_free += 1
            else:
                num_unknown += 1

        return {
            'total_cells': len(self.grid),
            'occupied': num_occupied,
            'free': num_free,
            'unknown': num_unknown
        }

    def get_visualization_data(self):
        """
        Get grid data for visualization.

        Returns:
            dict with keys:
                - 'occupied': Nx2 array of occupied cell positions
                - 'free': Nx2 array of free cell positions
                - 'unknown': Nx2 array of unknown cell positions
                - 'occupied_probs': N array of probabilities for occupied cells
        """
        occupied_cells = []
        occupied_probs = []
        free_cells = []
        unknown_cells = []

        for (gx, gy), log_odds in self.grid.items():
            prob = self._log_odds_to_prob(log_odds)
            x = gx * self.GRID_RESOLUTION
            y = gy * self.GRID_RESOLUTION

            if prob > self.OCCUPIED_THRESHOLD:
                occupied_cells.append([x, y])
                occupied_probs.append(prob)
            elif prob < self.FREE_THRESHOLD:
                free_cells.append([x, y])
            else:
                unknown_cells.append([x, y])

        return {
            'occupied': np.array(occupied_cells) if occupied_cells else np.empty((0, 2)),
            'occupied_probs': np.array(occupied_probs) if occupied_probs else np.array([]),
            'free': np.array(free_cells) if free_cells else np.empty((0, 2)),
            'unknown': np.array(unknown_cells) if unknown_cells else np.empty((0, 2))
        }

    def reset(self):
        """Clear the entire map."""
        self.grid.clear()
        self.cell_timestamps.clear()
        self.accumulated_displacement = np.array([0.0, 0.0])

        if self.logger:
            self.logger.info("ProbabilisticGridMap reset")

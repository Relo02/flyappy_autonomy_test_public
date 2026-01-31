import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import LaserScan
import numpy as np
from scipy.optimize import minimize

from flyappy_autonomy_code.probabilistic_grid_map import ProbabilisticGridMap

class NonlinearMPCController:
    def __init__(self, grid_map, logger=None):
        self.grid_map = grid_map
        self.logger = logger

        # ===== MPC PARAMETERS =====
        self.N = 15                 # Horizon steps
        self.dt = 0.033               # Time step
        self.state_dim = 4
        self.control_dim = 2

        # Limits
        self.max_acc = 1.5
        self.max_vel = 1.5

        # Desired forward velocity
        self.vx_ref = 0.5

        # Cost weights
        self.W_PROGRESS = 2.0
        self.W_VEL_TRACK = 10.0
        self.W_CONTROL = 50.0
        self.W_SMOOTH = 50.0
        self.W_OBSTACLE = 1.0
        self.W_HEIGHT = 10.0

        # Obstacle influence radius
        self.obs_radius = 1.0


    def solve(self, state):
        u0 = np.zeros(self.N * 2)
        bounds = [(-self.max_acc, self.max_acc)] * (self.N * 2)

        res = minimize(
            self._cost_function,
            u0,
            args=(state,),
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 50}
        )

        if not res.success:
            return np.array([0.0, 0.0])

        return res.x[:2]


    def _rollout(self, state, U):
        X = np.zeros((self.N+1, 4))
        X[0] = state

        for k in range(self.N):
            ax, ay = U[2*k:2*k+2]
            x, y, vx, vy = X[k]

            vx = np.clip(vx + ax*self.dt, -self.max_vel, self.max_vel)
            vy = np.clip(vy + ay*self.dt, -self.max_vel, self.max_vel)

            x += vx*self.dt
            y += vy*self.dt

            X[k+1] = [x, y, vx, vy]

        return X

    def _cost_function(self, U, state):
        X = self._rollout(state, U)
        cost = 0.0
        occupied = self.grid_map.get_occupied_cells()

        for k in range(self.N):
            x, y, vx, vy = X[k]
            ax, ay = U[2*k:2*k+2]

            cost -= self.W_PROGRESS * x
            cost += self.W_VEL_TRACK * (vx - self.vx_ref)**2
            cost += self.W_CONTROL * (ax**2 + ay**2)

            if k > 0:
                ax_prev, ay_prev = U[2*(k-1):2*(k-1)+2]
                cost += self.W_SMOOTH * ((ax-ax_prev)**2 + (ay-ay_prev)**2)

            mid_y = (self.grid_map.Y_GROUND + self.grid_map.Y_CEILING)/2
            cost += self.W_HEIGHT * (y - mid_y)**2

            for ox, oy in occupied:
                dx = x - ox
                dy = y - oy
                dist = np.hypot(dx, dy)
                if dist < self.obs_radius:
                    p = self.grid_map.get_occupancy_probability([ox, oy])
                    cost += self.W_OBSTACLE * p / (dist + 1e-2)**2

            if y < self.grid_map.Y_GROUND or y > self.grid_map.Y_CEILING:
                cost += 2000

        return cost
    
class MPCNode():
    def __init__(self, node):
        self.node = node

        self.grid_map = ProbabilisticGridMap(logger=self.node.get_logger())
        self.mpc = NonlinearMPCController(self.grid_map)

        self.pub_acc = self.node.create_publisher(Vector3, '/flyappy_acc', 10)
        self.node.create_subscription(Vector3, '/flyappy_vel', self.vel_callback, 10)
        self.node.create_subscription(LaserScan, '/flyappy_laser_scan', self.scan_callback, 10)

        self.state = np.zeros(4)  # [x, y, vx, vy]
        self.last_time = self.node.get_clock().now()

        self.timer = self.node.create_timer(0.1, self.control_loop)

    def vel_callback(self, msg):
        self.state[2] = msg.x
        self.state[3] = msg.y

    def scan_callback(self, scan):
        points = []
        angle = scan.angle_min

        for r in scan.ranges:
            if np.isfinite(r):
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                points.append([x, y])
            angle += scan.angle_increment

        points = np.array(points)

        now = self.node.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        self.grid_map.update_with_scan(
            points,
            current_velocity=self.state[2:4],
            current_y=self.state[1],
            current_time=now.nanoseconds * 1e-9,
            dt=dt
        )

    def control_loop(self):
        ax, ay = self.mpc.solve(self.state)

        msg = Vector3()
        msg.x = float(ax)
        msg.y = float(ay)
        msg.z = 0.0

        self.pub_acc.publish(msg)




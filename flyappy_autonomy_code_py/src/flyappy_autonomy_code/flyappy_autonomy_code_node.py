import rclpy

# Available controllers:
from flyappy_autonomy_code.flyappy_ros import FlyappyRos
from flyappy_autonomy_code.new_mpc import MPCNode

def main() -> None:
    rclpy.init()
    node = rclpy.node.Node('flyappy_autonomy_code_py')

    FlyappyRos(node)  # Alternative: MPC-based controller
    # MPCNode(node)

    rclpy.spin(node)


if __name__ == '__main__':
    main()

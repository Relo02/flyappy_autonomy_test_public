import rclpy

# Available controllers:
from flyappy_autonomy_code.flyappy_ros import FlyappyRos

def main() -> None:
    rclpy.init()
    node = rclpy.node.Node('flyappy_autonomy_code_py')

    FlyappyRos(node)  # Alternative: MPC-based controller
    rclpy.spin(node)

if __name__ == '__main__':
    main()

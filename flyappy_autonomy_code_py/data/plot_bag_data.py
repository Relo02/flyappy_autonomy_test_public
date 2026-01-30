#!/usr/bin/env python3

"""
Plot velocity and acceleration data from recorded ROS2 bag files
Usage: python3 plot_bag_data.py <bag_directory>
Example: python3 plot_bag_data.py flyappy_data_20240127_123456
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError:
    print("ERROR: ROS2 Python libraries not found!")
    print("Please install: sudo apt install ros-humble-rosbag2-py python3-rclpy")
    sys.exit(1)


class BagDataPlotter:
    """Extract and plot data from ROS2 bag files"""

    def __init__(self, bag_path: str):
        self.bag_path = bag_path

        # Data storage
        self.vel_times = []
        self.vel_x = []
        self.vel_y = []

        self.acc_times = []
        self.acc_x = []
        self.acc_y = []

        self.game_ended = False
        self.game_end_time = None

    def read_bag(self):
        """Read data from the bag file"""
        print(f"Reading bag: {self.bag_path}")

        # Setup reader
        storage_options = StorageOptions(uri=self.bag_path, storage_id='mcap')
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )

        reader = SequentialReader()
        reader.open(storage_options, converter_options)

        # Get topic types
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        print(f"Found topics: {list(type_map.keys())}")

        # Read messages
        start_time = None
        message_count = 0

        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            message_count += 1

            # Set start time from first message
            if start_time is None:
                start_time = timestamp

            # Convert timestamp to seconds relative to start
            time_sec = (timestamp - start_time) / 1e9

            # Deserialize based on topic
            if topic == '/flyappy_vel':
                msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, msg_type)
                self.vel_times.append(time_sec)
                self.vel_x.append(msg.x)
                self.vel_y.append(msg.y)

            elif topic == '/flyappy_acc':
                msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, msg_type)
                self.acc_times.append(time_sec)
                self.acc_x.append(msg.x)
                self.acc_y.append(msg.y)

            elif topic == '/flyappy_game_ended':
                msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, msg_type)
                if msg.data and not self.game_ended:
                    self.game_ended = True
                    self.game_end_time = time_sec

        print(f"Read {message_count} messages")
        print(f"Velocity samples: {len(self.vel_times)}")
        print(f"Acceleration samples: {len(self.acc_times)}")
        if self.game_ended:
            print(f"Game ended at: {self.game_end_time:.2f}s")

    def plot_data(self, save_path: str = None):
        """Create plots of the data"""

        # Create figure with subplots
        fig = plt.figure(figsize=(14, 10))

        # Convert to numpy arrays for calculations
        vel_x = np.array(self.vel_x)
        vel_y = np.array(self.vel_y)
        acc_x = np.array(self.acc_x)
        acc_y = np.array(self.acc_y)

        # Calculate speed magnitude
        speed = np.sqrt(vel_x**2 + vel_y**2)

        # 1. X Velocity
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(self.vel_times, vel_x, 'b-', linewidth=1.5, label='Velocity X')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        if self.game_end_time:
            ax1.axvline(x=self.game_end_time, color='r', linestyle='--', alpha=0.5, label='Game End')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Velocity X (m/s)')
        ax1.set_title('Forward Velocity (X)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. Y Velocity
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(self.vel_times, vel_y, 'g-', linewidth=1.5, label='Velocity Y')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        if self.game_end_time:
            ax2.axvline(x=self.game_end_time, color='r', linestyle='--', alpha=0.5, label='Game End')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity Y (m/s)')
        ax2.set_title('Vertical Velocity (Y)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 3. X Acceleration
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(self.acc_times, acc_x, 'b-', linewidth=1, alpha=0.7, label='Acceleration X')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        if self.game_end_time:
            ax3.axvline(x=self.game_end_time, color='r', linestyle='--', alpha=0.5, label='Game End')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Acceleration X (m/s²)')
        ax3.set_title('Forward Acceleration (X)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 4. Y Acceleration
        ax4 = plt.subplot(3, 2, 4)
        ax4.plot(self.acc_times, acc_y, 'g-', linewidth=1, alpha=0.7, label='Acceleration Y')
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        if self.game_end_time:
            ax4.axvline(x=self.game_end_time, color='r', linestyle='--', alpha=0.5, label='Game End')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Acceleration Y (m/s²)')
        ax4.set_title('Vertical Acceleration (Y)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        # 5. Speed Magnitude
        ax5 = plt.subplot(3, 2, 5)
        ax5.plot(self.vel_times, speed, 'purple', linewidth=1.5, label='Speed')
        if self.game_end_time:
            ax5.axvline(x=self.game_end_time, color='r', linestyle='--', alpha=0.5, label='Game End')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Speed (m/s)')
        ax5.set_title('Total Speed (Magnitude)')
        ax5.grid(True, alpha=0.3)
        ax5.legend()

        # 6. Velocity Vector Plot (X vs Y)
        ax6 = plt.subplot(3, 2, 6)
        scatter = ax6.scatter(vel_x, vel_y, c=self.vel_times, cmap='viridis',
                            s=10, alpha=0.6, label='Velocity Vector')
        ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax6.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax6.set_xlabel('Velocity X (m/s)')
        ax6.set_ylabel('Velocity Y (m/s)')
        ax6.set_title('Velocity Phase Plot')
        ax6.grid(True, alpha=0.3)
        ax6.axis('equal')
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('Time (s)')

        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

    def print_statistics(self):
        """Print statistics about the data"""
        print("\n" + "="*50)
        print("DATA STATISTICS")
        print("="*50)

        if len(self.vel_x) > 0:
            vel_x = np.array(self.vel_x)
            vel_y = np.array(self.vel_y)
            speed = np.sqrt(vel_x**2 + vel_y**2)

            print("\nVelocity X:")
            print(f"  Mean:  {np.mean(vel_x):.3f} m/s")
            print(f"  Std:   {np.std(vel_x):.3f} m/s")
            print(f"  Min:   {np.min(vel_x):.3f} m/s")
            print(f"  Max:   {np.max(vel_x):.3f} m/s")

            print("\nVelocity Y:")
            print(f"  Mean:  {np.mean(vel_y):.3f} m/s")
            print(f"  Std:   {np.std(vel_y):.3f} m/s")
            print(f"  Min:   {np.min(vel_y):.3f} m/s")
            print(f"  Max:   {np.max(vel_y):.3f} m/s")

            print("\nSpeed:")
            print(f"  Mean:  {np.mean(speed):.3f} m/s")
            print(f"  Std:   {np.std(speed):.3f} m/s")
            print(f"  Min:   {np.min(speed):.3f} m/s")
            print(f"  Max:   {np.max(speed):.3f} m/s")

        if len(self.acc_x) > 0:
            acc_x = np.array(self.acc_x)
            acc_y = np.array(self.acc_y)

            print("\nAcceleration X:")
            print(f"  Mean:  {np.mean(acc_x):.3f} m/s²")
            print(f"  Std:   {np.std(acc_x):.3f} m/s²")
            print(f"  Min:   {np.min(acc_x):.3f} m/s²")
            print(f"  Max:   {np.max(acc_x):.3f} m/s²")

            print("\nAcceleration Y:")
            print(f"  Mean:  {np.mean(acc_y):.3f} m/s²")
            print(f"  Std:   {np.std(acc_y):.3f} m/s²")
            print(f"  Min:   {np.min(acc_y):.3f} m/s²")
            print(f"  Max:   {np.max(acc_y):.3f} m/s²")

        print("="*50 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_bag_data.py <bag_directory>")
        print("Example: python3 plot_bag_data.py flyappy_data_20240127_123456")
        sys.exit(1)

    bag_path = sys.argv[1]

    # Check if path exists
    if not os.path.exists(bag_path):
        print(f"ERROR: Bag directory not found: {bag_path}")
        sys.exit(1)

    # Create plotter
    plotter = BagDataPlotter(bag_path)

    # Read and process data
    plotter.read_bag()

    # Print statistics
    plotter.print_statistics()

    # Create plot filename
    bag_name = Path(bag_path).name
    plot_filename = f"{bag_name}_plot.png"

    # Generate plots
    print(f"\nGenerating plots...")
    plotter.plot_data(save_path=plot_filename)

    print("\nDone!")


if __name__ == '__main__':
    main()

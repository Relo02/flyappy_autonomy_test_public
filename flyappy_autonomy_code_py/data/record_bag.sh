#!/bin/bash

# ROS2 Bag Recording Script for Flyappy Autonomy Data
# Records velocity and acceleration data for analysis

# Create data directory if it doesn't exist
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BAG_NAME="flyappy_data_${TIMESTAMP}"

echo "======================================"
echo "Flyappy Autonomy Data Recording"
echo "======================================"
echo "Recording bag: ${BAG_NAME}"
echo "Location: ${SCRIPT_DIR}"
echo ""
echo "Topics being recorded:"
echo "  - /flyappy_vel (x, y velocity)"
echo "  - /flyappy_acc (x, y acceleration)"
echo "  - /flyappy_laser_scan (obstacle detection)"
echo "  - /flyappy_game_ended (game status)"
echo ""
echo "Press Ctrl+C to stop recording"
echo "======================================"
echo ""

# Record the bag with specified topics
ros2 bag record \
  -o "${SCRIPT_DIR}/${BAG_NAME}" \
  /flyappy_vel \
  /flyappy_acc \
  /flyappy_laser_scan \
  /flyappy_game_ended

echo ""
echo "======================================"
echo "Recording stopped"
echo "Bag saved to: ${SCRIPT_DIR}/${BAG_NAME}"
echo "======================================"

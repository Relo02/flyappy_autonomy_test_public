#!/bin/bash

# Helper script to plot the most recent bag file

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================"
echo "Flyappy Bag Data Plotter"
echo "======================================"

# Find the most recent bag directory
LATEST_BAG=$(ls -td "${SCRIPT_DIR}"/flyappy_data_* 2>/dev/null | head -1)

if [ -z "$LATEST_BAG" ]; then
    echo "ERROR: No bag files found in ${SCRIPT_DIR}"
    echo ""
    echo "Please record data first using:"
    echo "  ./record_bag.sh"
    echo ""
    exit 1
fi

echo "Found latest bag: $(basename "$LATEST_BAG")"
echo ""

# Run the plotter
cd "${SCRIPT_DIR}"
python3 plot_bag_data.py "$LATEST_BAG"

echo ""
echo "======================================"

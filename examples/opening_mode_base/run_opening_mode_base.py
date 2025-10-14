#!/usr/bin/env python3
"""
Opening Mode Base Example

This script demonstrates the opening mode base case with a fracture oriented at 0 degrees.
The fracture grows over time with only normal stress (opening mode) and no shear stress.

Usage:
    python run_opening_mode_base.py [--recalculate] [--gauge_length GAUGE_LENGTH]

Parameters:
    --recalculate: Force recalculation instead of loading from HDF5 files
    --gauge_length: Channel spacing for interpolation (default: 10.0 meters)
"""

import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    setup_argument_parser,
    run_mode_simulation,
)


def main():
    """Main function to run opening mode base case."""
    parser = setup_argument_parser("Run opening mode base case")
    args = parser.parse_args()
    
    # Run the simulation
    run_mode_simulation(
        mode="opening_mode_base",
        mode_name="Opening Mode Base Case",
        orientation="0 degrees (no inclination)",
        stress_mode="Normal stress only (opening mode)",
        args=args
    )


if __name__ == "__main__":
    main()
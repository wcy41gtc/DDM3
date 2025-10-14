#!/usr/bin/env python3
"""
Shear Mode Incline Example

This script demonstrates the shear mode with a fracture inclined at -30 degrees.
The fracture experiences only shear stress (no normal stress) representing fault slip.

Usage:
    python run_shear_mode_incline.py [--recalculate] [--gauge_length GAUGE_LENGTH]

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
    """Main function to run shear mode incline case."""
    parser = setup_argument_parser("Run shear mode incline case")
    args = parser.parse_args()
    
    # Run the simulation
    run_mode_simulation(
        mode="shear_mode",
        mode_name="Shear Mode Incline Case",
        orientation="-30 degrees (inclined)",
        stress_mode="Shear stress only (fault slip)",
        args=args
    )


if __name__ == "__main__":
    main()
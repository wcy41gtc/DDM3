#!/usr/bin/env python3
"""
Standalone script for plotting fiber responses from saved HDF5 files.

This script allows you to plot results from previously saved HDF5 files
without needing to recalculate the entire workflow.
"""

import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .utils import (
    plot_from_h5_file,
    plot_from_h5_files_legacy,
    check_h5_files_exist,
    plot_geometry_and_stress_evolution,
    generate_geometry_and_stress_profiles,
    save_fibers_to_h5,
    plot_fiber_component
)


def main():
    """Main function for plotting from HDF5 files."""
    parser = argparse.ArgumentParser(
        description="Plot fiber responses from saved HDF5 files"
    )
    parser.add_argument(
        "mode",
        choices=["opening_mode_base", "opening_mode", "shear_mode", "mixed_mode", "opening_mode_base_test"],
        help="Mode to plot"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory containing HDF5 files (default: results)"
    )
    parser.add_argument(
        "--gauge-length",
        type=float,
        default=10.0,
        help="Gauge length for interpolation (default: 10.0)"
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=int,
        default=[12, 8],
        help="Figure size as width height (default: 12 8)"
    )
    parser.add_argument(
        "--plot-evolution",
        action="store_true",
        help="Also plot the geometry and stress evolution profiles"
    )
    
    args = parser.parse_args()
    
    print("DDM3D HDF5 Plotting Tool")
    print("=" * 40)
    print(f"Mode: {args.mode}")
    print(f"Output directory: {args.output_dir}")
    print(f"Gauge length: {args.gauge_length}")
    print(f"Figure size: {args.figsize}")
    print("=" * 40)
    
    # Check if HDF5 files exist
    if not check_h5_files_exist(args.mode, args.output_dir):
        print(f"Error: No HDF5 files found for mode '{args.mode}' in {args.output_dir}")
        print("Available files:")
        if os.path.exists(args.output_dir):
            for filename in os.listdir(args.output_dir):
                if filename.endswith(".h5"):
                    print(f"  {filename}")
        sys.exit(1)
    
    try:
        # Plot from HDF5 files
        plot_from_h5_files_legacy(
            mode=args.mode,
            output_dir=args.output_dir,
            gauge_length=args.gauge_length,
            figsize=tuple(args.figsize),
            fiber_id=1
        )
        
        # Optionally plot evolution profiles
        if args.plot_evolution:
            print("\nPlotting evolution profiles...")
            profiles = generate_geometry_and_stress_profiles()
            evolution_plot_path = os.path.join(
                args.output_dir, 
                f"{args.mode}_evolution_profiles.png"
            )
            plot_geometry_and_stress_evolution(
                profiles, 
                save_path=evolution_plot_path
            )
        
        print(f"\nPlotting completed successfully!")
        print(f"Results saved in '{args.output_dir}/' directory")
        
    except Exception as e:
        print(f"Error during plotting: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

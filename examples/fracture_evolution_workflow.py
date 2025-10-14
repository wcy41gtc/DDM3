"""
Fracture Evolution Workflow Examples

This script demonstrates the complete workflow for simulating fracture evolution
with different stress modes: opening_mode_base, opening_mode, shear_mode, and mixed_mode.

Based on the original DDM3D workflow with proper stress profile generation.

NOTE: Most functions have been moved to utils.py for better organization.
This script now serves as a legacy interface and example usage.
"""

import argparse
import os
from .utils import (
    generate_geometry_and_stress_profiles,
    make_fracture,
    create_fracture_series,
    create_fiber_network,
    calculate_fracture_evolution,
    save_fibers_to_h5,
    plot_fiber_component,
    save_results,
    check_h5_files_exist,
    plot_from_h5_file,
    plot_from_h5_files_legacy,
    create_material,
    setup_argument_parser,
    print_mode_header,
    run_mode_simulation,
)


def run_opening_mode_base(recalculate: bool = False, gauge_length: float = 10.0):
    """Run opening mode base case (0 degrees o1)."""
    print("=" * 60)
    print("RUNNING OPENING MODE BASE CASE")
    print("=" * 60)

    mode = "opening_mode_base"
    
    # Check if HDF5 files exist and recalculate is not forced
    if not recalculate and check_h5_files_exist(mode):
        print(f"HDF5 files found for {mode}. Loading and plotting from saved data...")
        try:
            plot_from_h5_files_legacy(mode, gauge_length=gauge_length, component="EYY_U", fiber_id=1)
            plot_from_h5_files_legacy(mode, gauge_length=gauge_length, component="EYY_U_Rate", fiber_id=1)
            print("Opening mode base case completed using saved data!")
            return
        except Exception as e:
            print(f"Error loading from HDF5 files: {e}")
            print("Falling back to recalculation...")

    # Generate stress profiles
    profiles = generate_geometry_and_stress_profiles()

    # Create material
    material = create_material()

    # Create fracture series
    fractures_series = create_fracture_series(mode, profiles, material)

    # Create fiber network
    fibers = create_fiber_network()

    # Calculate evolution
    calculate_fracture_evolution(fractures_series, fibers, mode)

    # Save results
    save_fibers_to_h5(fibers, mode)
    
    # Create plots for fiber 1 (across fracture)
    results_dir = f"results/{mode}"
    os.makedirs(results_dir, exist_ok=True)
    
    fiber1 = fibers[0]  # Fiber 1 (across fracture)
    plot_fiber_component(
        fiber1, 
        component="EYY_U", 
        gauge_length=gauge_length,
        save_path=f"{results_dir}/{mode}_fiber_1_EYY_U.png"
    )
    plot_fiber_component(
        fiber1, 
        component="EYY_U_Rate", 
        gauge_length=gauge_length,
        save_path=f"{results_dir}/{mode}_fiber_1_EYY_U_Rate.png"
    )

    print("Opening mode base case completed!")


def run_opening_mode(recalculate: bool = False, gauge_length: float = 10.0):
    """Run opening mode (-30 degrees o1)."""
    print("=" * 60)
    print("RUNNING OPENING MODE")
    print("=" * 60)

    mode = "opening_mode"
    
    # Check if HDF5 files exist and recalculate is not forced
    if not recalculate and check_h5_files_exist(mode):
        print(f"HDF5 files found for {mode}. Loading and plotting from saved data...")
        try:
            plot_from_h5_files_legacy(mode, gauge_length=gauge_length, component="EYY_U", fiber_id=1)
            plot_from_h5_files_legacy(mode, gauge_length=gauge_length, component="EYY_U_Rate", fiber_id=1)
            print("Opening mode completed using saved data!")
            return
        except Exception as e:
            print(f"Error loading from HDF5 files: {e}")
            print("Falling back to recalculation...")

    # Generate stress profiles
    profiles = generate_geometry_and_stress_profiles()

    # Create material
    material = create_material()

    # Create fracture series
    fractures_series = create_fracture_series(mode, profiles, material)

    # Create fiber network
    fibers = create_fiber_network()

    # Calculate evolution
    calculate_fracture_evolution(fractures_series, fibers, mode)

    # Save results
    save_fibers_to_h5(fibers, mode)
    # Create plots for fiber 1 (across fracture)
    results_dir = f"results/{mode}"
    os.makedirs(results_dir, exist_ok=True)
    
    fiber1 = fibers[0]  # Fiber 1 (across fracture)
    plot_fiber_component(
        fiber1, 
        component="EYY_U", 
        gauge_length=gauge_length,
        save_path=f"{results_dir}/{mode}_fiber_1_EYY_U.png"
    )
    plot_fiber_component(
        fiber1, 
        component="EYY_U_Rate", 
        gauge_length=gauge_length,
        save_path=f"{results_dir}/{mode}_fiber_1_EYY_U_Rate.png"
    )

    print("Opening mode completed!")


def run_shear_mode(recalculate: bool = False, gauge_length: float = 10.0):
    """Run shear mode."""
    print("=" * 60)
    print("RUNNING SHEAR MODE")
    print("=" * 60)

    mode = "shear_mode"
    
    # Check if HDF5 files exist and recalculate is not forced
    if not recalculate and check_h5_files_exist(mode):
        print(f"HDF5 files found for {mode}. Loading and plotting from saved data...")
        try:
            plot_from_h5_files_legacy(mode, gauge_length=gauge_length, component="EYY_U", fiber_id=1)
            plot_from_h5_files_legacy(mode, gauge_length=gauge_length, component="EYY_U_Rate", fiber_id=1)
            print("Shear mode completed using saved data!")
            return
        except Exception as e:
            print(f"Error loading from HDF5 files: {e}")
            print("Falling back to recalculation...")

    # Generate stress profiles
    profiles = generate_geometry_and_stress_profiles()

    # Create material
    material = create_material()

    # Create fracture series (30 time steps for shear mode)
    fractures_series = create_fracture_series(mode, profiles, material)

    # Create fiber network
    fibers = create_fiber_network()

    # Calculate evolution
    calculate_fracture_evolution(fractures_series, fibers, mode)

    # Save results
    save_fibers_to_h5(fibers, mode)
    # Create plots for fiber 1 (across fracture)
    results_dir = f"results/{mode}"
    os.makedirs(results_dir, exist_ok=True)
    
    fiber1 = fibers[0]  # Fiber 1 (across fracture)
    plot_fiber_component(
        fiber1, 
        component="EYY_U", 
        gauge_length=gauge_length,
        save_path=f"{results_dir}/{mode}_fiber_1_EYY_U.png"
    )
    plot_fiber_component(
        fiber1, 
        component="EYY_U_Rate", 
        gauge_length=gauge_length,
        save_path=f"{results_dir}/{mode}_fiber_1_EYY_U_Rate.png"
    )

    print("Shear mode completed!")


def run_mixed_mode(recalculate: bool = False, gauge_length: float = 10.0):
    """Run mixed mode (shear + normal stress)."""
    print("=" * 60)
    print("RUNNING MIXED MODE")
    print("=" * 60)

    mode = "mixed_mode"
    
    # Check if HDF5 files exist and recalculate is not forced
    if not recalculate and check_h5_files_exist(mode):
        print(f"HDF5 files found for {mode}. Loading and plotting from saved data...")
        try:
            plot_from_h5_files_legacy(mode, gauge_length=gauge_length, component="EYY_U", fiber_id=1)
            plot_from_h5_files_legacy(mode, gauge_length=gauge_length, component="EYY_U_Rate", fiber_id=1)
            print("Mixed mode completed using saved data!")
            return
        except Exception as e:
            print(f"Error loading from HDF5 files: {e}")
            print("Falling back to recalculation...")

    # Generate stress profiles
    profiles = generate_geometry_and_stress_profiles()

    # Create material
    material = create_material()

    # Create fracture series
    fractures_series = create_fracture_series(mode, profiles, material)

    # Create fiber network
    fibers = create_fiber_network()

    # Calculate evolution
    calculate_fracture_evolution(fractures_series, fibers, mode)

    # Save results
    save_fibers_to_h5(fibers, mode)
    # Create plots for fiber 1 (across fracture)
    results_dir = f"results/{mode}"
    os.makedirs(results_dir, exist_ok=True)
    
    fiber1 = fibers[0]  # Fiber 1 (across fracture)
    plot_fiber_component(
        fiber1, 
        component="EYY_U", 
        gauge_length=gauge_length,
        save_path=f"{results_dir}/{mode}_fiber_1_EYY_U.png"
    )
    plot_fiber_component(
        fiber1, 
        component="EYY_U_Rate", 
        gauge_length=gauge_length,
        save_path=f"{results_dir}/{mode}_fiber_1_EYY_U_Rate.png"
    )

    print("Mixed mode completed!")


def main():
    """Run all four fracture evolution modes."""
    parser = setup_argument_parser("DDM3D Fracture Evolution Workflow")
    parser.add_argument(
        "-m", "--mode", 
        choices=["opening_mode_base", "opening_mode", "shear_mode", "mixed_mode", "all"],
        default="all",
        help="Run specific mode or all modes (default: all)"
    )
    
    args = parser.parse_args()
    
    print("DDM3D Fracture Evolution Workflow")
    print("=" * 60)
    print(f"Recalculate: {args.recalculate}")
    print(f"Mode: {args.mode}")
    print(f"Gauge length: {args.gauge_length}")
    print("=" * 60)

    if args.mode == "all":
        # Run all modes
        run_opening_mode_base(args.recalculate, args.gauge_length)
        run_opening_mode(args.recalculate, args.gauge_length)
        run_shear_mode(args.recalculate, args.gauge_length)
        run_mixed_mode(args.recalculate, args.gauge_length)
    elif args.mode == "opening_mode_base":
        run_opening_mode_base(args.recalculate, args.gauge_length)
    elif args.mode == "opening_mode":
        run_opening_mode(args.recalculate, args.gauge_length)
    elif args.mode == "shear_mode":
        run_shear_mode(args.recalculate, args.gauge_length)
    elif args.mode == "mixed_mode":
        run_mixed_mode(args.recalculate, args.gauge_length)

    print("=" * 60)
    print("ALL MODES COMPLETED!")
    print("Results saved in 'results/' directory")
    print("=" * 60)


if __name__ == "__main__":
    main()
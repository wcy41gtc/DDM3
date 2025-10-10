"""
Test script for opening mode base case workflow.

This demonstrates the complete workflow with a smaller number of time steps
for testing purposes.
"""

import argparse
from examples.fracture_evolution_workflow import (
    generate_geometry_and_stress_profiles,
    create_fracture_series,
    create_fiber_network,
    calculate_fracture_evolution,
    save_results,
    check_h5_files_exist,
    plot_from_h5_files,
)
from ddm3d import Material


def test_opening_mode_base(recalculate: bool = False, gauge_length: float = 10.0):
    """Test opening mode base case with reduced time steps."""
    print("=" * 60)
    print("TESTING OPENING MODE BASE CASE")
    print("=" * 60)

    mode = "opening_mode_base_test"
    
    # Check if HDF5 files exist and recalculate is not forced
    if not recalculate and check_h5_files_exist(mode):
        print(f"HDF5 files found for {mode}. Loading and plotting from saved data...")
        try:
            plot_from_h5_files(mode, gauge_length=gauge_length)
            print("Opening mode base case test completed using saved data!")
            return
        except Exception as e:
            print(f"Error loading from HDF5 files: {e}")
            print("Falling back to recalculation...")

    # Generate stress profiles
    profiles = generate_geometry_and_stress_profiles()

    # Create material
    material = Material(shear_modulus=10e9, poisson_ratio=0.25)

    # Create fracture series
    fractures_series = []
    for i in range(90):  # 90 time steps for testing
        fracture = create_fracture_series("opening_mode_base", profiles, material)[i]
        fractures_series.append(fracture)

    # Create fiber network
    fibers = create_fiber_network()

    # Calculate evolution
    calculate_fracture_evolution(fractures_series, fibers, mode)

    # Save results
    save_results(fibers, mode)

    print("Opening mode base case test completed!")
    print(f"Results saved in 'results/' directory")


def main():
    """Main function with command line argument support."""
    parser = argparse.ArgumentParser(description="Test Opening Mode Base Case")
    parser.add_argument(
        "-r", "--recalculate", 
        action="store_true", 
        help="Force recalculation even if HDF5 files exist"
    )
    parser.add_argument(
        "-gl", "--gauge-length", 
        type=float, 
        default=10.0,
        help="Gauge length for plotting (default: 10.0)"
    )
    
    args = parser.parse_args()
    
    print(f"Recalculate: {args.recalculate}")
    print(f"Gauge length: {args.gauge_length}")
    print("=" * 60)
    
    test_opening_mode_base(args.recalculate, args.gauge_length)


if __name__ == "__main__":
    main()

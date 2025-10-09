"""
Test script for opening mode base case workflow.

This demonstrates the complete workflow with a smaller number of time steps
for testing purposes.
"""

from examples.fracture_evolution_workflow import (
    generate_geometry_and_stress_profiles,
    create_fracture_series,
    create_fiber_network,
    calculate_fracture_evolution,
    save_results,
)
from ddm3d import Material


def test_opening_mode_base():
    """Test opening mode base case with reduced time steps."""
    print("=" * 60)
    print("TESTING OPENING MODE BASE CASE")
    print("=" * 60)

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
    calculate_fracture_evolution(fractures_series, fibers, "opening_mode_base_test")

    # Save results
    save_results(fibers, "opening_mode_base_test")

    print("Opening mode base case test completed!")
    print(f"Results saved in 'results/' directory")


if __name__ == "__main__":
    test_opening_mode_base()

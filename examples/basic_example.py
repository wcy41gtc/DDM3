"""
Basic example demonstrating DDM3D usage.

This example shows how to:
1. Create a material
2. Create a rectangular fracture
3. Create a DAS fiber
4. Solve the displacement discontinuity problem
5. Calculate DAS response
"""

import numpy as np
import matplotlib.pyplot as plt
from ddm3d import Material, Fracture, Fiber, DDMCalculator
from .utils import plot_fracture_aperture, plot_fiber_response


def main():
    """Run the basic example."""
    print("DDM3D Basic Example")
    print("=" * 50)

    # 1. Create material properties
    print("1. Creating material...")
    material = Material(
        shear_modulus=10e9, poisson_ratio=0.25, name="Sandstone"  # 10 GPa
    )
    print(f"   Material: {material}")

    # 2. Create a rectangular fracture
    print("\n2. Creating rectangular fracture...")
    fracture = Fracture.create_rectangular(
        fracture_id=1,
        center=(0, 0, 0),  # meters
        length=100,  # meters
        height=50,  # meters
        element_size=(10, 10),  # meters
        orientation=(0, 90, 0),  # degrees (o1, o2, o3)
        material=material,
        initial_stress=(0, 0, 1e6),  # Pa (Ssl, Ssh, Snn)
    )
    print(f"   Fracture: {fracture}")

    # 3. Create a DAS fiber
    print("\n3. Creating DAS fiber...")
    fiber = Fiber.create_linear(
        fiber_id=1,
        start=(50, 10, -100),  # meters
        end=(50, 10, 100),  # meters
        n_channels=200,
    )
    print(f"   Fiber: {fiber}")

    # 4. Solve displacement discontinuities
    print("\n4. Solving displacement discontinuities...")
    calculator = DDMCalculator()
    calculator.solve_displacement_discontinuities([fracture])

    # Check some results
    element = fracture.get_element(1)
    print(f"   Element 1 displacement: {element.displacement}")

    # 5. Calculate DAS response
    print("\n5. Calculating DAS response...")
    calculator.calculate_fiber_response([fracture], [fiber])

    # Check some results
    channel = fiber.get_channel(100)  # Middle channel
    print(f"   Channel 100 stress SXX: {channel.get_stress_data('SXX')[-1]:.2e} Pa")
    print(f"   Channel 100 strain EXX: {channel.get_strain_data('EXX')[-1]:.2e}")

    # 6. Plot results
    print("\n6. Plotting results...")
    plot_fracture_aperture(fracture)
    plot_fiber_response(fiber)

    print("\nExample completed successfully!")


# Plotting functions are now imported from utils.py


if __name__ == "__main__":
    main()

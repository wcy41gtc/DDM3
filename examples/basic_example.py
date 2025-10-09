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


def plot_fracture_aperture(fracture):
    """Plot fracture aperture (normal displacement)."""
    # Get element centers and displacements
    centers = fracture.get_element_centers()
    displacements = [element.dnn for element in fracture.elements]

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        [c[0] for c in centers],
        [c[2] for c in centers],
        c=displacements,
        cmap="viridis",
        s=50,
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Fracture Aperture (Normal Displacement)")
    ax.set_aspect("equal")

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Displacement (m)")

    plt.tight_layout()
    plt.show()


def plot_fiber_response(fiber):
    """Plot fiber strain response."""
    # Get channel positions and strain data
    positions = fiber.get_channel_positions()
    strain_data = []

    for channel in fiber.channels:
        strain_data.append(channel.get_strain_data("EXX")[-1])

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate distance along fiber
    distances = [0]
    for i in range(1, len(positions)):
        dist = np.sqrt(
            (positions[i][0] - positions[i - 1][0]) ** 2
            + (positions[i][1] - positions[i - 1][1]) ** 2
            + (positions[i][2] - positions[i - 1][2]) ** 2
        )
        distances.append(distances[-1] + dist)

    ax.plot(distances, np.array(strain_data) * 1e6, "b-", linewidth=2)
    ax.set_xlabel("Distance along fiber (m)")
    ax.set_ylabel("Strain (μϵ)")
    ax.set_title("DAS Fiber Strain Response")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

"""
Fracture Evolution Workflow Examples

This script demonstrates the complete workflow for simulating fracture evolution
with different stress modes: opening_mode_base, opening_mode, shear_mode, and mixed_mode.

Based on the original DDM3D workflow with proper stress profile generation.
"""

import numpy as np
import h5py
import os
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt

from ddm3d import (
    Material,
    Fracture,
    Fiber,
    DDMCalculator,
    FracturePlotter,
    FiberPlotter,
)


def generate_geometry_and_stress_profiles(
    bsdt=60,
    asdt=30,
    l_scale_base=60,
    l_scale=20,
    h_scale=10,
    nl=10,
    nh=10,
    nn_scale=0.8e6,
    ss_scale=1.0e6,
) -> Dict[str, np.ndarray]:
    """
    Generate artificial stress profiles for fracture evolution.

    Parameters
    ----------
    bsdt : int, optional
        Before shut in dt, number of time steps before shut in (no real pressure,
        just assumed fracture opening and growth and stress profile) (unitless), by default 60
    asdt : int, optional
        After shut in dt, number of time steps after shut in (no real pressure,
        just assumed fracture closing and stress profile) (unitless), by default 30
    l_scale_base : int, optional
        Base length scale (m), by default 60
    l_scale : int, optional
        Length scale (m), by default 20
    h_scale : int, optional
        Height scale (m), by default 10
    nl : int, optional
        Number of elements along l, by default 10
    nh : int, optional
        Number of elements along h, by default 10
    nn_scale : float, optional
        Normal stress scale (Pa), by default 0.8e6
    ss_scale : float, optional
        Shear stress scale (Pa), by default 1.0e6

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing stress profiles and fracture parameters
    """
    # Define time steps
    base_for_geometry = np.linspace(1, l_scale_base, bsdt)
    base_for_snn_increase_before_shutin = np.linspace(0.1, 10, bsdt)
    base_for_snn_decay_after_shutin = np.linspace(0.1, 10, asdt)
    base_for_ssl_increase_after_shutin = np.linspace(0, 100, asdt)

    # Series of total length and total height for each time step
    _l = l_scale * np.sqrt(base_for_geometry)
    _h = h_scale * np.sqrt(base_for_geometry)
    _dl = l_scale / nl * np.sqrt(base_for_geometry)
    _dh = h_scale / nh * np.sqrt(base_for_geometry)

    # Stitch two parts together - first part fracture is growing,
    # second part fracture stopped growing
    l = np.concatenate((_l, np.ones(asdt) * _l[-1]))
    h = np.concatenate((_h, np.ones(asdt) * _h[-1]))
    dl = np.concatenate((_dl, np.ones(asdt) * _dl[-1]))
    dh = np.concatenate((_dh, np.ones(asdt) * _dh[-1]))

    # Assemble shear stress profile ssl
    _ssl_before_shutin = np.zeros(bsdt)
    _ssl_after_shutin = (
        -ss_scale * np.exp(0.01 * base_for_ssl_increase_after_shutin) + ss_scale
    )
    ssl = -np.concatenate((_ssl_before_shutin, _ssl_after_shutin))

    # Assemble shear stress profile ssh
    ssh = np.zeros(bsdt + asdt)

    # Assemble normal stress profile snn
    snn_before_shutin = nn_scale * np.arctan(base_for_snn_increase_before_shutin)
    snn_after_shutin = snn_before_shutin[-1] * np.exp(
        -1 * base_for_snn_decay_after_shutin
    )
    snn = np.concatenate((snn_before_shutin, snn_after_shutin))

    return {"l": l, "h": h, "dl": dl, "dh": dh, "ssl": ssl, "ssh": ssh, "snn": snn}


def make_fracture(
    tot_l: float,
    tot_h: float,
    dl: float,
    dh: float,
    c_x: float,
    c_y: float,
    c_z: float,
    o1: float,
    o2: float,
    o3: float,
    dsl: float,
    dsh: float,
    dnn: float,
    Ssl: float,
    Ssh: float,
    Snn: float,
    material: Material,
) -> Fracture:
    """
    Create a fracture with specified parameters matching the original DDM3D make_fracture function.

    Parameters
    ----------
    tot_l : float
        Total length
    tot_h : float
        Total height
    dl : float
        Element length
    dh : float
        Element height
    c_x, c_y, c_z : float
        Center coordinates
    o1 : float
        Orientation angle o1
    o2 : float
        Orientation angle o2
    o3 : float
        Orientation angle o3
    dsl, dsh, dnn : float
        Displacement discontinuities
    Ssl, Ssh, Snn : float
        Stress components
    material : Material
        Material properties

    Returns
    -------
    Fracture
        Created fracture object
    """
    # Create fracture using the new OOP approach
    fracture = Fracture.create_rectangular(
        fracture_id=0,
        center=(c_x, c_y, c_z),
        length=tot_l,
        height=tot_h,
        element_size=(dl, dh),
        orientation=(o1, o2, o3),
        material=material,
        initial_stress=(Ssl, Ssh, Snn),
    )

    # Set stress and displacement discontinuities for all elements
    for element in fracture.elements:
        element.set_stress((Ssl, Ssh, Snn))
        element.set_displacement((dsl, dsh, dnn))

    return fracture


def create_fracture_series(
    mode: str, profiles: Dict[str, np.ndarray], material: Material
) -> List[List[Fracture]]:
    """
    Create a series of fractures for different modes following the original workflow.

    Parameters
    ----------
    mode : str
        Mode type: 'opening_mode_base', 'opening_mode', 'shear_mode', 'mixed_mode'
    profiles : Dict[str, np.ndarray]
        Stress profiles and fracture parameters
    material : Material
        Material properties

    Returns
    -------
    List[List[Fracture]]
        List of fracture lists for each time step
    """
    growing_fractures = []

    if mode == "opening_mode_base":
        # Opening mode base case - 0 degrees o1
        x = 0.0
        y = 0.0
        z = 0.0
        o1 = 0.0
        o2 = 0.0
        o3 = 0.0

        for i in range(90):
            fracture = make_fracture(
                profiles["l"][i],
                profiles["h"][i],
                profiles["dl"][i],
                profiles["dh"][i],
                x,
                y,
                z,  # c_x, c_y, c_z
                o1,
                o2,
                o3,  # o1, o2, o3
                0.0,
                0.0,
                0.0,  # dsl, dsh, dnn
                0.0,
                0.0,
                profiles["snn"][i],  # Ssl, Ssh, Snn
                material,
            )
            growing_fractures.append([fracture])

    elif mode == "opening_mode":
        # Opening mode - -30 degrees o1
        x = 0
        y = -(50) * np.tan(np.deg2rad(30))
        z = 0
        o1 = -30.0
        o2 = 0.0
        o3 = 0.0

        for i in range(90):
            fracture = make_fracture(
                profiles["l"][i],
                profiles["h"][i],
                profiles["dl"][i],
                profiles["dh"][i],
                x,
                y,
                z,  # c_x, c_y, c_z
                o1,
                o2,
                o3,  # o1, o2, o3
                0.0,
                0.0,
                0.0,  # dsl, dsh, dnn
                0.0,
                0.0,
                profiles["snn"][i],  # Ssl, Ssh, Snn
                material,
            )
            growing_fractures.append([fracture])

    elif mode == "shear_mode":
        # Shear mode - only 30 time steps
        x = 0
        y = -(50) * np.tan(np.deg2rad(30))
        z = 0
        o1 = -30.0
        o2 = 0.0
        o3 = 0.0

        for i in range(30):
            fracture = make_fracture(
                profiles["l"][i],
                profiles["h"][i],
                profiles["dl"][i],
                profiles["dh"][i],
                x,
                y,
                z,  # c_x, c_y, c_z
                o1,
                o2,
                o3,  # o1, o2, o3
                0.0,
                0.0,
                0.0,  # dsl, dsh, dnn
                profiles["ssl"][i],
                0.0,
                0.0,  # Ssl, Ssh, Snn
                material,
            )
            growing_fractures.append([fracture])

    elif mode == "mixed_mode":
        # Mixed mode
        x = 0
        y = -(50) * np.tan(np.deg2rad(30))
        z = 0
        o1 = -30.0
        o2 = 0.0
        o3 = 0.0

        for i in range(90):
            fracture = make_fracture(
                profiles["l"][i],
                profiles["h"][i],
                profiles["dl"][i],
                profiles["dh"][i],
                x,
                y,
                z,  # c_x, c_y, c_z
                o1,
                o2,
                o3,  # o1, o2, o3
                0.0,
                0.0,
                0.0,  # dsl, dsh, dnn
                profiles["ssl"][i],
                0.0,
                profiles["snn"][i],  # Ssl, Ssh, Snn
                material,
            )
            growing_fractures.append([fracture])

    return growing_fractures


def create_fiber_network() -> List[Fiber]:
    """
    Create a network of DAS fibers for monitoring.
    Matches the original DDM3D fiber configuration.

    Returns
    -------
    List[Fiber]
        List of fiber objects
    """
    fibers = []

    # Fiber 1: fiber across the fracture
    fiber1 = Fiber.create_linear(
        fiber_id=1, start=(50, 100, 0), end=(50, -100, 0), n_channels=200
    )
    fibers.append(fiber1)

    # Fiber 2: fiber parallel to the fracture
    fiber2 = Fiber.create_linear(
        fiber_id=2, start=(50, 10, -100), end=(50, 10, 100), n_channels=200
    )
    fibers.append(fiber2)

    # Fiber 3: fiber parallel to the fracture
    fiber3 = Fiber.create_linear(
        fiber_id=3, start=(50, 50, -100), end=(50, 50, 100), n_channels=200
    )
    fibers.append(fiber3)

    return fibers


def calculate_fracture_evolution(
    fractures_series: List[List[Fracture]], fibers: List[Fiber], mode: str
) -> None:
    """
    Calculate stress, displacement, strain, and strain rate for fracture evolution.

    Parameters
    ----------
    fractures_series : List[List[Fracture]]
        Series of fractures for each time step
    fibers : List[Fiber]
        List of monitoring fibers
    mode : str
        Mode name for output files
    """
    calculator = DDMCalculator()

    print(f"Calculating {mode} evolution with {len(fractures_series)} time steps...")

    for i, fractures in enumerate(fractures_series):
        print(f"  Processing time step {i}/{len(fractures_series)}")

        # Solve displacement discontinuities
        calculator.solve_displacement_discontinuities(fractures)

        # Calculate fiber responses
        calculator.calculate_fiber_response(fractures, fibers)

        # Store time step information
        for fiber in fibers:
            fiber.add_time_step(i)

    print(f"Completed {mode} calculations")


def save_results(fibers: List[Fiber], mode: str, output_dir: str = "results") -> None:
    """
    Save results as plots and HDF5 files.

    Parameters
    ----------
    fibers : List[Fiber]
        List of fibers with calculated data
    mode : str
        Mode name for file naming
    output_dir : str
        Output directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving {mode} results...")

    for fiber in fibers:
        # Save HDF5 data
        h5_filename = os.path.join(output_dir, f"{mode}_fiber_{fiber.fiber_id}.h5")
        with h5py.File(h5_filename, "w") as f:
            # Save channel positions
            positions = fiber.get_channel_positions()
            f.create_dataset("positions", data=positions)

            # Save time series data for each channel
            for i, channel in enumerate(fiber.channels):
                if channel.stress_data:
                    stress_data = np.array(channel.stress_data)
                    f.create_dataset(f"channel_{i}_stress", data=stress_data)

                if channel.strain_data:
                    strain_data = np.array(channel.strain_data)
                    f.create_dataset(f"channel_{i}_strain", data=strain_data)

                if channel.displacement_data:
                    disp_data = np.array(channel.displacement_data)
                    f.create_dataset(f"channel_{i}_displacement", data=disp_data)

        # Create plots (without showing them)
        if fiber.fiber_id == 1:
            # Strain response contour plot (EYY_U)
            strain_plot_filename = os.path.join(
                output_dir, f"{mode}_fiber_{fiber.fiber_id}_EYY_U.png"
            )
            FiberPlotter.plot_fiber_contour(
                fiber,
                component="EYY_U",
                scale=20.0,
                gauge_length=10.0,
                figsize=(12, 8),
                save_path=strain_plot_filename,
            )

            # Stress response contour plot (EYY_U_Rate)
            stress_plot_filename = os.path.join(
                output_dir, f"{mode}_fiber_{fiber.fiber_id}_EYY_U_Rate.png"
            )
            FiberPlotter.plot_fiber_contour(
                fiber,
                component="EYY_U_Rate",
                scale=20.0,
                gauge_length=10.0,
                figsize=(12, 8),
                save_path=stress_plot_filename,
            )
        else:
            # Strain response contour plot (EZZ_U)
            strain_plot_filename = os.path.join(
                output_dir, f"{mode}_fiber_{fiber.fiber_id}_EZZ_U.png"
            )
            FiberPlotter.plot_fiber_contour(
                fiber,
                component="EZZ_U",
                scale=5.0,
                gauge_length=10.0,
                figsize=(12, 8),
                save_path=strain_plot_filename,
            )

            # Stress response contour plot (EZZ_U_Rate)
            stress_plot_filename = os.path.join(
                output_dir, f"{mode}_fiber_{fiber.fiber_id}_EZZ_U_Rate.png"
            )
            FiberPlotter.plot_fiber_contour(
                fiber,
                component="EZZ_U_Rate",
                scale=5.0,
                gauge_length=10.0,
                figsize=(12, 8),
                save_path=stress_plot_filename,
            )

    print(f"Saved {mode} results to {output_dir}/")


def run_opening_mode_base():
    """Run opening mode base case (0 degrees o1)."""
    print("=" * 60)
    print("RUNNING OPENING MODE BASE CASE")
    print("=" * 60)

    # Generate stress profiles
    profiles = generate_geometry_and_stress_profiles()

    # Create material
    material = Material(shear_modulus=10e9, poisson_ratio=0.25)

    # Create fracture series
    fractures_series = create_fracture_series("opening_mode_base", profiles, material)

    # Create fiber network
    fibers = create_fiber_network()

    # Calculate evolution
    calculate_fracture_evolution(fractures_series, fibers, "opening_mode_base")

    # Save results
    save_results(fibers, "opening_mode_base")

    print("Opening mode base case completed!")


def run_opening_mode():
    """Run opening mode (-30 degrees o1)."""
    print("=" * 60)
    print("RUNNING OPENING MODE")
    print("=" * 60)

    # Generate stress profiles
    profiles = generate_geometry_and_stress_profiles()

    # Create material
    material = Material(shear_modulus=10e9, poisson_ratio=0.25)

    # Create fracture series
    fractures_series = create_fracture_series("opening_mode", profiles, material)

    # Create fiber network
    fibers = create_fiber_network()

    # Calculate evolution
    calculate_fracture_evolution(fractures_series, fibers, "opening_mode")

    # Save results
    save_results(fibers, "opening_mode")

    print("Opening mode completed!")


def run_shear_mode():
    """Run shear mode."""
    print("=" * 60)
    print("RUNNING SHEAR MODE")
    print("=" * 60)

    # Generate stress profiles
    profiles = generate_geometry_and_stress_profiles()

    # Create material
    material = Material(shear_modulus=10e9, poisson_ratio=0.25)

    # Create fracture series (30 time steps for shear mode)
    fractures_series = create_fracture_series("shear_mode", profiles, material)

    # Create fiber network
    fibers = create_fiber_network()

    # Calculate evolution
    calculate_fracture_evolution(fractures_series, fibers, "shear_mode")

    # Save results
    save_results(fibers, "shear_mode")

    print("Shear mode completed!")


def run_mixed_mode():
    """Run mixed mode (shear + normal stress)."""
    print("=" * 60)
    print("RUNNING MIXED MODE")
    print("=" * 60)

    # Generate stress profiles
    profiles = generate_geometry_and_stress_profiles()

    # Create material
    material = Material(shear_modulus=10e9, poisson_ratio=0.25)

    # Create fracture series
    fractures_series = create_fracture_series("mixed_mode", profiles, material)

    # Create fiber network
    fibers = create_fiber_network()

    # Calculate evolution
    calculate_fracture_evolution(fractures_series, fibers, "mixed_mode")

    # Save results
    save_results(fibers, "mixed_mode")

    print("Mixed mode completed!")


def main():
    """Run all four fracture evolution modes."""
    print("DDM3D Fracture Evolution Workflow")
    print("=" * 60)

    # Run all modes
    run_opening_mode_base()
    run_opening_mode()
    run_shear_mode()
    run_mixed_mode()

    print("=" * 60)
    print("ALL MODES COMPLETED!")
    print("Results saved in 'results/' directory")
    print("=" * 60)


if __name__ == "__main__":
    main()

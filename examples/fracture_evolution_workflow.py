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


def plot_geometry_and_stress_evolution(
    profiles: Dict[str, np.ndarray],
    save_path: str = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Plot the geometry and stress evolution profiles.
    
    Parameters
    ----------
    profiles : Dict[str, np.ndarray]
        Dictionary containing geometry and stress profiles from generate_geometry_and_stress_profiles
    save_path : str, optional
        Path to save the plot. If None, displays the plot, by default None
    figsize : Tuple[int, int], optional
        Figure size (width, height), by default (15, 10)
    """
    # Create time array (assuming unit time steps)
    time_steps = np.arange(len(profiles['l']))
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Fracture Geometry and Stress Evolution Profiles', fontsize=16, fontweight='bold')
    
    # Plot 1: Geometry evolution (l, h)
    ax1 = axes[0, 0]
    ax1.plot(time_steps, profiles['l'], 'b-', linewidth=2, label='Length (l)')
    ax1.plot(time_steps, profiles['h'], 'r-', linewidth=2, label='Height (h)')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Dimension (m)')
    ax1.set_title('Fracture Dimensions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Element size evolution (dl, dh)
    ax2 = axes[0, 1]
    ax2.plot(time_steps, profiles['dl'], 'b-', linewidth=2, label='Element Length (dl)')
    ax2.plot(time_steps, profiles['dh'], 'r-', linewidth=2, label='Element Height (dh)')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Element Size (m)')
    ax2.set_title('Element Dimensions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Normal stress evolution (snn)
    ax3 = axes[0, 2]
    ax3.plot(time_steps, profiles['snn'] / 1e6, 'g-', linewidth=2, label='Normal Stress (snn)')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Stress (MPa)')
    ax3.set_title('Normal Stress Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Shear stress evolution (ssl)
    ax4 = axes[1, 0]
    ax4.plot(time_steps, profiles['ssl'] / 1e6, 'm-', linewidth=2, label='Shear Stress (ssl)')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Stress (MPa)')
    ax4.set_title('Shear Stress Evolution (ssl)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Shear stress evolution (ssh)
    ax5 = axes[1, 1]
    ax5.plot(time_steps, profiles['ssh'] / 1e6, 'c-', linewidth=2, label='Shear Stress (ssh)')
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Stress (MPa)')
    ax5.set_title('Shear Stress Evolution (ssh)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Combined stress evolution
    ax6 = axes[1, 2]
    ax6.plot(time_steps, profiles['snn'] / 1e6, 'g-', linewidth=2, label='Normal (snn)')
    ax6.plot(time_steps, profiles['ssl'] / 1e6, 'm-', linewidth=2, label='Shear (ssl)')
    ax6.plot(time_steps, profiles['ssh'] / 1e6, 'c-', linewidth=2, label='Shear (ssh)')
    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('Stress (MPa)')
    ax6.set_title('All Stress Components')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evolution plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_evolution_comparison(
    profiles_list: List[Dict[str, np.ndarray]],
    labels: List[str],
    save_path: str = None,
    figsize: Tuple[int, int] = (15, 12)
) -> None:
    """
    Plot comparison of multiple geometry and stress evolution profiles.
    
    Parameters
    ----------
    profiles_list : List[Dict[str, np.ndarray]]
        List of profile dictionaries from generate_geometry_and_stress_profiles
    labels : List[str]
        Labels for each profile set
    save_path : str, optional
        Path to save the plot. If None, displays the plot, by default None
    figsize : Tuple[int, int], optional
        Figure size (width, height), by default (15, 12)
    """
    if len(profiles_list) != len(labels):
        raise ValueError("Number of profiles must match number of labels")
    
    # Create time arrays for each profile (they may have different lengths)
    time_steps_list = [np.arange(len(profiles['l'])) for profiles in profiles_list]
    
    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle('Fracture Evolution Comparison', fontsize=16, fontweight='bold')
    
    # Colors for different profiles
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    
    # Plot 1: Geometry evolution (l, h)
    ax1 = axes[0, 0]
    for i, (profiles, label, time_steps) in enumerate(zip(profiles_list, labels, time_steps_list)):
        color = colors[i % len(colors)]
        ax1.plot(time_steps, profiles['l'], f'{color}-', linewidth=2, 
                label=f'{label} - Length', alpha=0.8)
        ax1.plot(time_steps, profiles['h'], f'{color}--', linewidth=2, 
                label=f'{label} - Height', alpha=0.8)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Dimension (m)')
    ax1.set_title('Fracture Dimensions')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Element size evolution (dl, dh)
    ax2 = axes[0, 1]
    for i, (profiles, label, time_steps) in enumerate(zip(profiles_list, labels, time_steps_list)):
        color = colors[i % len(colors)]
        ax2.plot(time_steps, profiles['dl'], f'{color}-', linewidth=2, 
                label=f'{label} - dl', alpha=0.8)
        ax2.plot(time_steps, profiles['dh'], f'{color}--', linewidth=2, 
                label=f'{label} - dh', alpha=0.8)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Element Size (m)')
    ax2.set_title('Element Dimensions')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Normal stress evolution (snn)
    ax3 = axes[1, 0]
    for i, (profiles, label, time_steps) in enumerate(zip(profiles_list, labels, time_steps_list)):
        color = colors[i % len(colors)]
        ax3.plot(time_steps, profiles['snn'] / 1e6, f'{color}-', linewidth=2, 
                label=f'{label} - snn', alpha=0.8)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Stress (MPa)')
    ax3.set_title('Normal Stress Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Shear stress evolution (ssl)
    ax4 = axes[1, 1]
    for i, (profiles, label, time_steps) in enumerate(zip(profiles_list, labels, time_steps_list)):
        color = colors[i % len(colors)]
        ax4.plot(time_steps, profiles['ssl'] / 1e6, f'{color}-', linewidth=2, 
                label=f'{label} - ssl', alpha=0.8)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Stress (MPa)')
    ax4.set_title('Shear Stress Evolution (ssl)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Shear stress evolution (ssh)
    ax5 = axes[2, 0]
    for i, (profiles, label, time_steps) in enumerate(zip(profiles_list, labels, time_steps_list)):
        color = colors[i % len(colors)]
        ax5.plot(time_steps, profiles['ssh'] / 1e6, f'{color}-', linewidth=2, 
                label=f'{label} - ssh', alpha=0.8)
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Stress (MPa)')
    ax5.set_title('Shear Stress Evolution (ssh)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Combined stress evolution
    ax6 = axes[2, 1]
    for i, (profiles, label, time_steps) in enumerate(zip(profiles_list, labels, time_steps_list)):
        color = colors[i % len(colors)]
        ax6.plot(time_steps, profiles['snn'] / 1e6, f'{color}-', linewidth=2, 
                label=f'{label} - snn', alpha=0.8)
        ax6.plot(time_steps, profiles['ssl'] / 1e6, f'{color}--', linewidth=1, 
                label=f'{label} - ssl', alpha=0.6)
        ax6.plot(time_steps, profiles['ssh'] / 1e6, f'{color}:', linewidth=1, 
                label=f'{label} - ssh', alpha=0.6)
    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('Stress (MPa)')
    ax6.set_title('All Stress Components')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evolution comparison plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


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

"""
Common utility functions for DDM3D examples.

This module contains shared functions used across different fracture mode examples
to avoid code duplication and improve maintainability.
"""

import numpy as np
import h5py
import os
import sys
import argparse
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt

# Add project root to path for ddm3d imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ddm3d import (
    Material,
    Fracture,
    Fiber,
    DDMCalculator,
    FracturePlotter,
    FiberPlotter,
)


def generate_geometry_and_stress_profiles(
    bsdt=60.0,
    asdt=30.0,
    l_scale_base=60.0,
    l_scale=20.0,
    h_scale=10.0,
    nl=10,
    nh=10,
    nn_scale=0.8e6,
    ss_scale=1.0e6,
) -> Dict[str, np.ndarray]:
    """
    Generate artificial stress profiles for fracture evolution.

    Parameters
    ----------
    bsdt : float, optional
        Before shut in dt, number of time steps before shut in (no real pressure,
        just assumed fracture opening and growth and stress profile) (unitless), by default 60.0
    asdt : float, optional
        After shut in dt, number of time steps after shut in (no real pressure,
        just assumed fracture closing and stress profile) (unitless), by default 30.0
    l_scale_base : float, optional
        Base length scale (m), by default 60.0
    l_scale : float, optional
        Length scale (m), by default 20.0
    h_scale : float, optional
        Height scale (m), by default 10.0
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
    # Convert float inputs to integers for numpy functions
    bsdt = int(bsdt)
    asdt = int(asdt)
    
    # Define time steps
    base_for_geometry = np.linspace(1, l_scale_base, bsdt)
    base_for_snn_increase_before_shutin = np.linspace(0.1, 10, bsdt)
    base_for_snn_decay_after_shutin = np.linspace(0.1, 10, asdt)
    base_for_ssl_increase_after_shutin = np.linspace(0, 100, asdt)

    # Series of total length and total height for each time step
    _l = l_scale * np.sqrt(base_for_geometry)
    _h = h_scale * np.sqrt(base_for_geometry)
    _dl = l_scale / float(nl) * np.sqrt(base_for_geometry)
    _dh = h_scale / float(nh) * np.sqrt(base_for_geometry)

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
    snn_after_shutin = snn_before_shutin[-1] * np.exp(-1 * base_for_snn_decay_after_shutin)
    snn = np.concatenate((snn_before_shutin, snn_after_shutin))

    return {
        'l': l,
        'h': h,
        'dl': dl,
        'dh': dh,
        'ssl': ssl,
        'ssh': ssh,
        'snn': snn
    }


def make_fracture(
    length: float,
    height: float,
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
    """Create a fracture with specified parameters."""
    fracture = Fracture.create_elliptical(
        fracture_id=0,
        center=(c_x, c_y, c_z),
        length=length,
        height=height,
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
        print(f"Time step {i+1}/{len(fractures_series)}")

        # Calculate stress, displacement, strain, and strain rate
        calculator.solve_displacement_discontinuities(fractures)
        calculator.calculate_fiber_response(fractures, fibers)

        # Store time step information
        for fiber in fibers:
            fiber.add_time_step(i)

    print(f"Calculation completed for {mode}!")


def save_fibers_to_h5(fibers: List[Fiber], mode: str, output_dir: str = "results") -> None:
    """
    Save fiber data to HDF5 files.

    Parameters
    ----------
    fibers : List[Fiber]
        List of fibers with stored results
    mode : str
        Mode name for output files
    output_dir : str
        Output directory for HDF5 files
    """
    # Create results directory
    results_dir = f"{output_dir}/{mode}"
    os.makedirs(results_dir, exist_ok=True)

    # Save HDF5 files
    for fiber in fibers:
        h5_filename = f"{results_dir}/{mode}_fiber_{fiber.fiber_id}.h5"
        fiber.save_to_h5(h5_filename)
        print(f"Saved HDF5 file: {h5_filename}")

    print(f"HDF5 files saved to {results_dir}/")


def plot_fiber_component(
    fiber: Fiber,
    component: str = "EXX",
    scale: float = 1.0,
    gauge_length: float = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot a specific component from a fiber object using FiberPlotter.plot_fiber_contour().

    Parameters
    ----------
    fiber : Fiber
        Fiber object with stored results
    component : str, optional
        Component to plot ('SXX', 'SYY', 'SZZ', 'SXY', 'SXZ', 'SYZ',
                          'UXX', 'UYY', 'UZZ', 'EXX', 'EYY', 'EZZ',
                          'EXX_U', 'EYY_U', 'EZZ_U', 'EXX_Rate', 'EYY_Rate', 'EZZ_Rate',
                          'EXX_U_Rate', 'EYY_U_Rate', 'EZZ_U_Rate'), by default 'EXX'
    scale : float, optional
        Scale factor for the data, by default 1.0
    gauge_length : float, optional
        Desired channel spacing for interpolation in meters. If None, uses original channel spacing, by default None
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 8)
    save_path : str, optional
        Path to save the plot, by default None
    """
    plotter = FiberPlotter()
    plotter.plot_fiber_contour(
        fiber, 
        component=component,
        scale=scale,
        gauge_length=gauge_length,
        figsize=figsize,
        save_path=save_path
    )
    plt.close('all')  # Close all figures to free memory


def save_results(fibers: List[Fiber], mode: str, gauge_length: float = 10.0) -> None:
    """
    Save results as plots and HDF5 files (legacy function for backward compatibility).

    Parameters
    ----------
    fibers : List[Fiber]
        List of fibers with stored results
    mode : str
        Mode name for output files
    gauge_length : float
        Channel spacing for interpolation
    """
    # Save HDF5 files
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


def check_h5_files_exist(mode: str) -> bool:
    """Check if HDF5 files exist for the given mode."""
    results_dir = f"results/{mode}"
    if not os.path.exists(results_dir):
        return False
    
    # Check for at least one HDF5 file
    for fiber_id in [1, 2, 3]:
        h5_file = f"{results_dir}/{mode}_fiber_{fiber_id}.h5"
        if not os.path.exists(h5_file):
            return False
    
    return True


def load_fiber_from_h5(h5_file_path: str) -> Fiber:
    """
    Load fiber data from HDF5 file and return a Fiber object.

    Parameters
    ----------
    h5_file_path : str
        Path to the HDF5 file

    Returns
    -------
    Fiber
        Fiber object with loaded data
    """
    if not os.path.exists(h5_file_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_file_path}")
    
    # Read fiber metadata from HDF5 file
    with h5py.File(h5_file_path, 'r') as f:
        fiber_id = f.attrs['fiber_id']
        n_channels = f.attrs['n_channels']
        positions = f['positions'][:]
    
    # Create fiber with correct geometry
    start = (positions[0, 0], positions[0, 1], positions[0, 2])
    end = (positions[-1, 0], positions[-1, 1], positions[-1, 2])
    
    fiber = Fiber.create_linear(fiber_id=fiber_id, start=start, end=end, n_channels=n_channels)
    fiber.load_from_h5(h5_file_path)
    return fiber


def load_fiber_from_h5_legacy(mode: str, fiber_id: int) -> Fiber:
    """Legacy function for backward compatibility."""
    h5_file = f"results/{mode}/{mode}_fiber_{fiber_id}.h5"
    return load_fiber_from_h5(h5_file)


def plot_from_h5_file(
    h5_file_path: str,
    component: str = "EXX",
    scale: float = 1.0,
    gauge_length: float = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Load fiber from HDF5 file and plot specified component.
    
    Parameters
    ----------
    h5_file_path : str
        Path to the HDF5 file
    component : str, optional
        Component to plot, by default "EXX"
    scale : float, optional
        Scale factor for the data, by default 1.0
    gauge_length : float, optional
        Channel spacing for interpolation, by default None
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 8)
    save_path : str, optional
        Path to save the plot, by default None
    """
    # Load fiber from HDF5 file
    fiber = load_fiber_from_h5(h5_file_path)
    
    # Plot the specified component
    plot_fiber_component(
        fiber,
        component=component,
        scale=scale,
        gauge_length=gauge_length,
        figsize=figsize,
        save_path=save_path
    )


def plot_from_h5_files_legacy(
    mode: str, 
    gauge_length: float = 10.0,
    component: str = "EXX",
    scale: float = 1.0,
    figsize: Tuple[int, int] = (12, 8),
    output_dir: str = "results",
    fiber_id: int = 1
) -> None:
    """Legacy function for backward compatibility."""
    print(f"Loading and plotting {mode} from HDF5 files...")
    
    h5_file = f"{output_dir}/{mode}/{mode}_fiber_{fiber_id}.h5"
    save_path = f"{output_dir}/{mode}/{mode}_fiber_{fiber_id}_{component}.png"
    
    plot_from_h5_file(
        h5_file,
        component=component,
        scale=scale,
        gauge_length=gauge_length,
        figsize=figsize,
        save_path=save_path
    )


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
        # Opening mode incline - -30 degrees o1
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
        # Shear mode incline - -30 degrees o1
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
                0.0,  # Ssl, Ssh, Snn
                material,
            )
            growing_fractures.append([fracture])

    elif mode == "mixed_mode":
        # Mixed mode incline - -30 degrees o1
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


def create_material() -> Material:
    """Create standard material properties."""
    return Material(shear_modulus=10e9, poisson_ratio=0.25)


def setup_argument_parser(description: str) -> argparse.ArgumentParser:
    """Setup common argument parser for all examples."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-r", "--recalculate", 
        action="store_true", 
        help="Force recalculation instead of loading from HDF5 files"
    )
    parser.add_argument(
        "-gl", "--gauge_length", 
        type=float, 
        default=5.0, 
        help="Channel spacing for interpolation in plotting only (default: 10.0 meters)"
    )
    parser.add_argument(
        "-s", "--scale", 
        type=float, 
        default=20.0, 
        help="Scale factor for the data for plotting, by default 1.0"
    )
    return parser


def print_mode_header(mode_name: str, orientation: str, stress_mode: str, args) -> None:
    """Print standardized header for each mode."""
    print("=" * 60)
    print(f"RUNNING {mode_name.upper()}")
    print("=" * 60)
    print(f"Fracture orientation: {orientation}")
    print(f"Stress mode: {stress_mode}")
    print(f"Recalculate: {args.recalculate}")
    print(f"Gauge length: {args.gauge_length} meters")
    print("=" * 60)


def run_mode_simulation(mode: str, mode_name: str, orientation: str, stress_mode: str, args) -> None:
    """
    Run a complete simulation for any fracture mode.
    
    Parameters
    ----------
    mode : str
        Mode identifier for file naming
    mode_name : str
        Display name for the mode
    orientation : str
        Fracture orientation description
    stress_mode : str
        Stress mode description
    args : argparse.Namespace
        Command line arguments
    """
    print_mode_header(mode_name, orientation, stress_mode, args)
    
    # Check if HDF5 files exist and recalculate is not forced
    if not args.recalculate and check_h5_files_exist(mode):
        print(f"HDF5 files found for {mode}. Loading and plotting from saved data...")
        try:
            plot_from_h5_files_legacy(mode, gauge_length=args.gauge_length, component="EYY_U", scale=args.scale, fiber_id=1)
            plot_from_h5_files_legacy(mode, gauge_length=args.gauge_length, component="EYY_U_Rate", scale=args.scale, fiber_id=1)
            plot_from_h5_files_legacy(mode, gauge_length=args.gauge_length, component="EZZ_U", scale=args.scale, fiber_id=2)
            plot_from_h5_files_legacy(mode, gauge_length=args.gauge_length, component="EZZ_U_Rate", scale=args.scale, fiber_id=2)
            plot_from_h5_files_legacy(mode, gauge_length=args.gauge_length, component="EZZ_U", scale=args.scale, fiber_id=3)
            plot_from_h5_files_legacy(mode, gauge_length=args.gauge_length, component="EZZ_U_Rate", scale=args.scale, fiber_id=3)
            print(f"{mode_name} completed using saved data!")
            return
        except Exception as e:
            print(f"Error loading from HDF5 files: {e}")
            print("Falling back to recalculation...")

    # Generate stress profiles
    profiles = generate_geometry_and_stress_profiles()
    print(f"Generated stress profiles with {len(profiles['l'])} time steps")

    # Create material
    material = create_material()

    # Create fracture series
    fractures_series = create_fracture_series(mode, profiles, material)
    print(f"Created {len(fractures_series)} fracture time steps")

    # Create fiber network
    fibers = create_fiber_network()
    print(f"Created {len(fibers)} monitoring fibers")

    # Calculate fracture evolution
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
        scale=args.scale, 
        gauge_length=args.gauge_length,
        save_path=f"{results_dir}/{mode}_fiber_1_EYY_U.png"
    )
    plot_fiber_component(
        fiber1, 
        component="EYY_U_Rate", 
        scale=args.scale, 
        gauge_length=args.gauge_length,
        save_path=f"{results_dir}/{mode}_fiber_1_EYY_U_Rate.png"
    )
    fiber2 = fibers[1]  # Fiber 2 (across fracture)
    plot_fiber_component(
        fiber2, 
        component="EZZ_U", 
        scale=args.scale, 
        gauge_length=args.gauge_length,
        save_path=f"{results_dir}/{mode}_fiber_2_EZZ_U.png"
    )
    plot_fiber_component(
        fiber2, 
        component="EZZ_U_Rate", 
        scale=args.scale, 
        gauge_length=args.gauge_length,
        save_path=f"{results_dir}/{mode}_fiber_2_EZZ_U_Rate.png"
    )
    fiber3 = fibers[2]  # Fiber 3 (across fracture)
    plot_fiber_component(
        fiber3, 
        component="EZZ_U", 
        scale=args.scale, 
        gauge_length=args.gauge_length,
        save_path=f"{results_dir}/{mode}_fiber_3_EZZ_U.png"
    )
    plot_fiber_component(
        fiber3, 
        component="EXX_U_Rate", 
        scale=args.scale, 
        gauge_length=args.gauge_length,
        save_path=f"{results_dir}/{mode}_fiber_3_EZZ_U_Rate.png"
    )

    print(f"{mode_name} completed successfully!")


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

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


# =============================================================================
# MODE-SPECIFIC PLOTTING FUNCTIONS
# =============================================================================

def plot_fracture_dimensions(profiles, save_path, title):
    """Plot fracture length and height evolution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    time_steps = np.arange(len(profiles['l']))
    
    # Length evolution
    ax1.plot(time_steps, profiles['l'], 'b-', linewidth=2, label='Length')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Length (m)')
    ax1.set_title('Fracture Length Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Height evolution
    ax2.plot(time_steps, profiles['h'], 'r-', linewidth=2, label='Height')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Height (m)')
    ax2.set_title('Fracture Height Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(f'{title} - Fracture Dimensions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_opening_mode_stresses(profiles, save_path, title):
    """Plot normal stress for opening mode (no shear stress)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    time_steps = np.arange(len(profiles['snn']))
    
    # Normal stress evolution
    ax.plot(time_steps, profiles['snn'] / 1e6, 'g-', linewidth=2, label='Normal Stress (Snn)')
    ax.axvline(x=60, color='k', linestyle='--', alpha=0.7, label='Shut-in Time')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Stress (MPa)')
    ax.set_title('Normal Stress Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.suptitle(f'{title} - Normal Stress Only', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_shear_mode_stresses(profiles, save_path, title):
    """Plot shear stress for shear mode (no normal stress)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    time_steps = np.arange(len(profiles['ssl']))
    
    # Shear stress evolution
    ax.plot(time_steps, profiles['ssl'] / 1e6, 'm-', linewidth=2, label='Shear Stress (Ssl)')
    ax.axvline(x=60, color='k', linestyle='--', alpha=0.7, label='Shut-in Time')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Stress (MPa)')
    ax.set_title('Shear Stress Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.suptitle(f'{title} - Shear Stress Only', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_mixed_mode_stresses(profiles, save_path, title):
    """Plot both normal and shear stresses for mixed mode."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    time_steps = np.arange(len(profiles['snn']))
    
    # Normal stress evolution
    ax1.plot(time_steps, profiles['snn'] / 1e6, 'g-', linewidth=2, label='Normal Stress (Snn)')
    ax1.axvline(x=60, color='k', linestyle='--', alpha=0.7, label='Shut-in Time')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Stress (MPa)')
    ax1.set_title('Normal Stress Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Shear stress evolution
    ax2.plot(time_steps, profiles['ssl'] / 1e6, 'm-', linewidth=2, label='Shear Stress (Ssl)')
    ax2.axvline(x=60, color='k', linestyle='--', alpha=0.7, label='Shut-in Time')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Stress (MPa)')
    ax2.set_title('Shear Stress Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(f'{title} - Normal and Shear Stresses', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# BASIC EXAMPLE UTILITIES
# =============================================================================

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

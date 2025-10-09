"""Plotting utilities for DDM3D visualization."""

from typing import List, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from ..core.fracture import Fracture
from ..core.fiber import Fiber


class FracturePlotter:
    """Plotting utilities for fractures."""

    @staticmethod
    def plot_aperture(
        fracture: Fracture,
        component: str = "dnn",
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot fracture aperture (displacement discontinuities).

        Parameters
        ----------
        fracture : Fracture
            Fracture to plot
        component : str, optional
            Displacement component to plot ('dsl', 'dsh', 'dnn'), by default 'dnn'
        title : str, optional
            Plot title, by default None
        figsize : Tuple[int, int], optional
            Figure size, by default (10, 8)
        save_path : str, optional
            Path to save the plot, by default None
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )

        if component not in ["dsl", "dsh", "dnn"]:
            raise ValueError("Component must be 'dsl', 'dsh', or 'dnn'")

        # Get element data
        centers = fracture.get_element_centers()
        displacements = []

        for element in fracture.elements:
            if component == "dsl":
                displacements.append(element.dsl)
            elif component == "dsh":
                displacements.append(element.dsh)
            else:  # dnn
                displacements.append(element.dnn)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        scatter = ax.scatter(
            [c[0] for c in centers],
            [c[2] for c in centers],
            c=displacements,
            cmap="viridis",
            s=50,
            edgecolors="black",
            linewidth=0.5,
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(title or f"Fracture {component.upper()} Displacement")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Displacement (m)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

    @staticmethod
    def plot_3d_aperture(
        fracture: Fracture,
        component: str = "dnn",
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> None:
        """
        Plot fracture aperture in 3D.

        Parameters
        ----------
        fracture : Fracture
            Fracture to plot
        component : str, optional
            Displacement component to plot ('dsl', 'dsh', 'dnn'), by default 'dnn'
        title : str, optional
            Plot title, by default None
        figsize : Tuple[int, int], optional
            Figure size, by default (12, 8)
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )

        if component not in ["dsl", "dsh", "dnn"]:
            raise ValueError("Component must be 'dsl', 'dsh', or 'dnn'")

        # Get element data
        centers = fracture.get_element_centers()
        displacements = []

        for element in fracture.elements:
            if component == "dsl":
                displacements.append(element.dsl)
            elif component == "dsh":
                displacements.append(element.dsh)
            else:  # dnn
                displacements.append(element.dnn)

        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            [c[0] for c in centers],
            [c[1] for c in centers],
            [c[2] for c in centers],
            c=displacements,
            cmap="viridis",
            s=50,
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title or f"Fracture {component.upper()} Displacement (3D)")

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label("Displacement (m)")

        plt.tight_layout()


class FiberPlotter:
    """Plotting utilities for DAS fibers."""

    @staticmethod
    def _interpolate_fiber_data(
        fiber: Fiber, data_list: List[List[float]], target_gauge_length: float
    ) -> Tuple[np.ndarray, int]:
        """
        Interpolate fiber data to a new gauge length.

        Parameters
        ----------
        fiber : Fiber
            Fiber object
        data_list : List[List[float]]
            List of time series data for each channel
        target_gauge_length : float
            Target gauge length for interpolation

        Returns
        -------
        Tuple[np.ndarray, int]
            Interpolated data array and number of interpolated channels
        """
        # Get original channel positions
        positions = fiber.get_channel_positions()
        n_original_channels = len(positions)
        n_time_steps = len(data_list[0])

        # Calculate distances along fiber
        distances = [0.0]
        for i in range(1, n_original_channels):
            pos1 = np.array(positions[i - 1])
            pos2 = np.array(positions[i])
            dist = np.linalg.norm(pos2 - pos1)
            distances.append(distances[-1] + dist)

        # Create target positions
        fiber_length = distances[-1]
        n_interp_channels = int(fiber_length / target_gauge_length) + 1
        target_distances = np.linspace(0, fiber_length, n_interp_channels)

        # Interpolate data for each time step
        interpolated_data = np.zeros((n_time_steps, n_interp_channels))

        for t in range(n_time_steps):
            # Get data for this time step
            time_data = [data_list[i][t] for i in range(n_original_channels)]

            # Interpolate using linear interpolation
            interpolated_data[t, :] = np.interp(target_distances, distances, time_data)

        return interpolated_data, n_interp_channels

    @staticmethod
    def plot_strain_response(
        fiber: Fiber,
        component: str = "EXX",
        time_step: int = -1,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot strain response along the fiber.

        Parameters
        ----------
        fiber : Fiber
            Fiber to plot
        component : str, optional
            Strain component to plot ('EXX', 'EYY', 'EZZ'), by default 'EXX'
        time_step : int, optional
            Time step to plot (-1 for last), by default -1
        title : str, optional
            Plot title, by default None
        figsize : Tuple[int, int], optional
            Figure size, by default (12, 6)
        save_path : str, optional
            Path to save the plot, by default None
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )

        if component not in ["EXX", "EYY", "EZZ"]:
            raise ValueError("Component must be 'EXX', 'EYY', or 'EZZ'")

        # Get channel data
        positions = fiber.get_channel_positions()
        strain_data = []

        for channel in fiber.channels:
            strain_values = channel.get_strain_data(component)
            if strain_values and len(strain_values) > abs(time_step):
                strain_data.append(strain_values[time_step])
            else:
                strain_data.append(0.0)

        # Calculate distance along fiber
        distances = [0]
        for i in range(1, len(positions)):
            dist = np.sqrt(
                (positions[i][0] - positions[i - 1][0]) ** 2
                + (positions[i][1] - positions[i - 1][1]) ** 2
                + (positions[i][2] - positions[i - 1][2]) ** 2
            )
            distances.append(distances[-1] + dist)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            distances,
            np.array(strain_data) * 1e6,
            "b-",
            linewidth=2,
            marker="o",
            markersize=3,
        )
        ax.set_xlabel("Distance along fiber (m)")
        ax.set_ylabel("Strain (μϵ)")
        ax.set_title(title or f"DAS Fiber {component} Strain Response")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

    @staticmethod
    def plot_stress_response(
        fiber: Fiber,
        component: str = "SXX",
        time_step: int = -1,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot stress response along the fiber.

        Parameters
        ----------
        fiber : Fiber
            Fiber to plot
        component : str, optional
            Stress component to plot ('SXX', 'SYY', 'SZZ', 'SXY', 'SXZ', 'SYZ'), by default 'SXX'
        time_step : int, optional
            Time step to plot (-1 for last), by default -1
        title : str, optional
            Plot title, by default None
        figsize : Tuple[int, int], optional
            Figure size, by default (12, 6)
        save_path : str, optional
            Path to save the plot, by default None
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )

        if component not in ["SXX", "SYY", "SZZ", "SXY", "SXZ", "SYZ"]:
            raise ValueError(
                "Component must be one of 'SXX', 'SYY', 'SZZ', 'SXY', 'SXZ', 'SYZ'"
            )

        # Get channel data
        positions = fiber.get_channel_positions()
        stress_data = []

        for channel in fiber.channels:
            stress_values = channel.get_stress_data(component)
            if stress_values and len(stress_values) > abs(time_step):
                stress_data.append(stress_values[time_step])
            else:
                stress_data.append(0.0)

        # Calculate distance along fiber
        distances = [0]
        for i in range(1, len(positions)):
            dist = np.sqrt(
                (positions[i][0] - positions[i - 1][0]) ** 2
                + (positions[i][1] - positions[i - 1][1]) ** 2
                + (positions[i][2] - positions[i - 1][2]) ** 2
            )
            distances.append(distances[-1] + dist)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            distances,
            np.array(stress_data) / 1e6,
            "r-",
            linewidth=2,
            marker="o",
            markersize=3,
        )
        ax.set_xlabel("Distance along fiber (m)")
        ax.set_ylabel("Stress (MPa)")
        ax.set_title(title or f"DAS Fiber {component} Stress Response")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

    @staticmethod
    def plot_fiber_geometry(
        fiber: Fiber,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot fiber geometry in 3D.

        Parameters
        ----------
        fiber : Fiber
            Fiber to plot
        title : str, optional
            Plot title, by default None
        figsize : Tuple[int, int], optional
            Figure size, by default (10, 8)
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )
        # Get channel positions
        positions = fiber.get_channel_positions()

        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Plot fiber path
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        z_coords = [pos[2] for pos in positions]

        ax.plot(x_coords, y_coords, z_coords, "b-", linewidth=2, label="Fiber path")
        ax.scatter(x_coords, y_coords, z_coords, c="red", s=20, label="Channels")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title or f"DAS Fiber {fiber.fiber_id} Geometry")
        ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

    @staticmethod
    def plot_fiber_contour(
        fiber: Fiber,
        component: str = "EXX",
        scale: float = 1.0,
        gauge_length: float = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot a color contour image of the quantity stored in the fiber channels.
        This method replicates the original fibre_plot function format with dynamic interpolation.

        Parameters
        ----------
        fiber : Fiber
            Fiber that stores quantities
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
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )

        # Get channel positions and data
        positions = fiber.get_channel_positions()
        n_channels = len(fiber.channels)

        if n_channels == 0:
            raise ValueError("Fiber has no channels")

        # Get data from first channel to determine time steps
        first_channel = fiber.channels[0]

        # Determine interpolation parameters
        if gauge_length is None:
            # Use original channel spacing
            target_gauge_length = first_channel.gauge_length
            interpolate = False
        else:
            # Use specified gauge length for interpolation
            target_gauge_length = gauge_length
            interpolate = True

        # Calculate fiber length and number of interpolated channels
        fiber_length = fiber.get_total_length()
        n_interp_channels = int(fiber_length / target_gauge_length) + 1

        # Determine data type and get data
        if component.startswith("S"):
            # Stress components
            data_list = []
            for channel in fiber.channels:
                stress_data = channel.get_stress_data(component)
                data_list.append(stress_data)

            if not data_list[0]:
                raise ValueError(f"No {component} data available")

            n_time_steps = len(data_list[0])

            # Apply interpolation if requested
            if interpolate:
                _img, n_plot_channels = FiberPlotter._interpolate_fiber_data(
                    fiber, data_list, target_gauge_length
                )
            else:
                _img = np.zeros((n_time_steps, n_channels))
                for i, channel_data in enumerate(data_list):
                    _img[:, i] = channel_data
                n_plot_channels = n_channels

            # Create time and channel arrays
            t = np.linspace(0, n_time_steps, n_time_steps)
            chn = np.linspace(0, n_plot_channels * target_gauge_length, n_plot_channels)
            T, CHN = np.meshgrid(t, chn)

            # Scale and plot
            vmax = np.max(np.abs(_img)) / 1e6 / scale
            levels = MaxNLocator(nbins=200).tick_values(-vmax, vmax)

            # Create figure
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

            # Plot with appropriate sign convention
            if component in ["SXX", "SYY", "SZZ"]:
                img = ax.contourf(
                    T, CHN, -_img.T / 1e6, cmap="bwr", levels=levels, extend="both"
                )
            else:
                img = ax.contourf(
                    T, CHN, _img.T / 1e6, cmap="bwr", levels=levels, extend="both"
                )

            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=0.2, pad=0.05)
            cbar = plt.colorbar(img, cax=cax)
            cbar.ax.set_ylabel("Stress(MPa)", labelpad=15, rotation=270)

        elif component.startswith("U"):
            # Displacement components
            data_list = []
            for channel in fiber.channels:
                disp_data = channel.get_displacement_data(component)
                data_list.append(disp_data)

            if not data_list[0]:
                raise ValueError(f"No {component} data available")

            n_time_steps = len(data_list[0])

            # Apply interpolation if requested
            if interpolate:
                _img, n_plot_channels = FiberPlotter._interpolate_fiber_data(
                    fiber, data_list, target_gauge_length
                )
            else:
                _img = np.zeros((n_time_steps, n_channels))
                for i, channel_data in enumerate(data_list):
                    _img[:, i] = channel_data
                n_plot_channels = n_channels

            # Create time and channel arrays
            t = np.linspace(0, n_time_steps, n_time_steps)
            chn = np.linspace(0, n_plot_channels * target_gauge_length, n_plot_channels)
            T, CHN = np.meshgrid(t, chn)

            # Scale and plot
            vmax = np.max(np.abs(_img)) * 1e3 / scale
            levels = MaxNLocator(nbins=200).tick_values(-vmax, vmax)

            # Create figure
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

            img = ax.contourf(
                T, CHN, _img.T * 1e3, cmap="bwr", levels=levels, extend="both"
            )

            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=0.2, pad=0.05)
            cbar = plt.colorbar(img, cax=cax)
            cbar.ax.set_ylabel("Displacement(mm)", labelpad=15, rotation=270)

        elif component.startswith("E"):
            # Strain components
            if component.endswith("_U"):
                # Strain calculated from displacement
                base_component = component.replace("_U", "")
                disp_component = base_component.replace("E", "U")

                data_list = []
                for channel in fiber.channels:
                    disp_data = channel.get_displacement_data(disp_component)
                    data_list.append(disp_data)

                if not data_list[0]:
                    raise ValueError(f"No {disp_component} data available")

                n_time_steps = len(data_list[0])

                # Apply interpolation if requested
                if interpolate:
                    _img, n_plot_channels = FiberPlotter._interpolate_fiber_data(
                        fiber, data_list, target_gauge_length
                    )
                else:
                    _img = np.zeros((n_time_steps, n_channels))
                    for i, channel_data in enumerate(data_list):
                        _img[:, i] = channel_data
                    n_plot_channels = n_channels

                # Calculate strain from displacement using target gauge length
                for i in range(_img.shape[0]):
                    for j in range(_img.shape[1] - 1):
                        _img[i][j] = (
                            (_img[i][j + 1] - _img[i][j]) / target_gauge_length * 1e6
                        )

                # Create time and channel arrays
                t = np.linspace(0, n_time_steps, n_time_steps)
                chn = np.linspace(
                    0, n_plot_channels * target_gauge_length, n_plot_channels
                )
                T, CHN = np.meshgrid(t, chn)

                # Scale and plot
                vmax = np.max(np.abs(_img)) / scale
                levels = MaxNLocator(nbins=100).tick_values(-vmax, vmax)

                # Create figure
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111)

                if base_component == "EZZ":
                    img = ax.contourf(
                        T, CHN, -_img.T, cmap="bwr", levels=levels, extend="both"
                    )
                else:
                    img = ax.contourf(
                        T, CHN, -_img.T, cmap="bwr", levels=levels, extend="both"
                    )

                # Add colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size=0.2, pad=0.05)
                cbar = plt.colorbar(img, cax=cax)
                cbar.ax.set_ylabel("$\\mu\\epsilon$", labelpad=15)

            elif component.endswith("_Rate"):
                # Strain rate
                base_component = component.replace("_Rate", "")

                if base_component.endswith("_U"):
                    # Strain rate from displacement
                    disp_component = base_component.replace("E", "U").replace("_U", "")

                    data_list = []
                    for channel in fiber.channels:
                        disp_data = channel.get_displacement_data(disp_component)
                        data_list.append(disp_data)

                    if not data_list[0]:
                        raise ValueError(f"No {disp_component} data available")

                    n_time_steps = len(data_list[0])

                    # Apply interpolation if requested
                    if interpolate:
                        _img, n_plot_channels = FiberPlotter._interpolate_fiber_data(
                            fiber, data_list, target_gauge_length
                        )
                    else:
                        _img = np.zeros((n_time_steps, n_channels))
                        for i, channel_data in enumerate(data_list):
                            _img[:, i] = channel_data
                        n_plot_channels = n_channels

                    # Calculate strain from displacement using target gauge length
                    for i in range(_img.shape[0]):
                        for j in range(_img.shape[1] - 1):
                            _img[i][j] = (
                                (_img[i][j + 1] - _img[i][j])
                                / target_gauge_length
                                * 1e6
                            )

                    # Calculate strain rate
                    _img = np.diff(_img, axis=0)

                    # Create time and channel arrays
                    t = np.linspace(0, n_time_steps, n_time_steps - 1)
                    chn = np.linspace(
                        0, n_plot_channels * target_gauge_length, n_plot_channels
                    )
                    T, CHN = np.meshgrid(t, chn)

                    # Scale and plot
                    vmax = np.max(np.abs(_img)) / scale
                    levels = MaxNLocator(nbins=200).tick_values(-vmax, vmax)

                    # Create figure
                    fig = plt.figure(figsize=figsize)
                    ax = fig.add_subplot(111)

                    if base_component == "EZZ_U":
                        img = ax.contourf(
                            T, CHN, -_img.T, cmap="bwr", levels=levels, extend="both"
                        )
                    else:
                        img = ax.contourf(
                            T, CHN, _img.T, cmap="bwr", levels=levels, extend="both"
                        )

                    # Add colorbar
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size=0.2, pad=0.05)
                    cbar = plt.colorbar(img, cax=cax)
                    cbar.ax.set_ylabel("$\\mu\\epsilon$/min", labelpad=15)

                else:
                    # Strain rate from strain data
                    data_list = []
                    for channel in fiber.channels:
                        strain_data = channel.get_strain_data(base_component)
                        data_list.append(strain_data)

                    if not data_list[0]:
                        raise ValueError(f"No {base_component} data available")

                    n_time_steps = len(data_list[0])
                    _img = np.zeros((n_time_steps, n_channels))

                    for i, channel_data in enumerate(data_list):
                        _img[:, i] = channel_data

                    # Calculate strain rate
                    _img = np.diff(_img, axis=0) * 1e6

                    # Create time and channel arrays
                    t = np.linspace(0, n_time_steps, n_time_steps - 1)
                    chn = np.linspace(
                        0, n_channels * first_channel.gauge_length, n_channels
                    )
                    T, CHN = np.meshgrid(t, chn)

                    # Scale and plot
                    vmax = np.max(np.abs(_img)) / scale
                    levels = MaxNLocator(nbins=200).tick_values(-vmax, vmax)

                    # Create figure
                    fig = plt.figure(figsize=figsize)
                    ax = fig.add_subplot(111)

                    img = ax.contourf(
                        T, CHN, -_img.T, cmap="bwr", levels=levels, extend="both"
                    )

                    # Add colorbar
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size=0.2, pad=0.05)
                    cbar = plt.colorbar(img, cax=cax)
                    cbar.ax.set_ylabel("$\\mu\\epsilon$/min", labelpad=15)

            else:
                # Direct strain data
                data_list = []
                for channel in fiber.channels:
                    strain_data = channel.get_strain_data(component)
                    data_list.append(strain_data)

                if not data_list[0]:
                    raise ValueError(f"No {component} data available")

                n_time_steps = len(data_list[0])

                # Apply interpolation if requested
                if interpolate:
                    _img, n_plot_channels = FiberPlotter._interpolate_fiber_data(
                        fiber, data_list, target_gauge_length
                    )
                else:
                    _img = np.zeros((n_time_steps, n_channels))
                    for i, channel_data in enumerate(data_list):
                        _img[:, i] = channel_data
                    n_plot_channels = n_channels

                # Convert to microstrain
                _img = _img * 1e6

                # Create time and channel arrays
                t = np.linspace(0, n_time_steps, n_time_steps)
                chn = np.linspace(
                    0, n_plot_channels * target_gauge_length, n_plot_channels
                )
                T, CHN = np.meshgrid(t, chn)

                # Scale and plot
                vmax = np.max(np.abs(_img)) / scale
                levels = MaxNLocator(nbins=200).tick_values(-vmax, vmax)

                # Create figure
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111)

                img = ax.contourf(
                    T, CHN, -_img.T, cmap="bwr", levels=levels, extend="both"
                )

                # Add colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size=0.2, pad=0.05)
                cbar = plt.colorbar(img, cax=cax)
                cbar.ax.set_ylabel("$\\mu\\epsilon$", labelpad=15)

        else:
            raise ValueError(f"Unsupported component: {component}")

        # Set axis properties
        ax.invert_yaxis()
        ax.set_xlabel("Time(min)")
        ax.set_ylabel("Fibre Length(m)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)

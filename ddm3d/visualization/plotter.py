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
        component: str = 'dnn',
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
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
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
        
        if component not in ['dsl', 'dsh', 'dnn']:
            raise ValueError("Component must be 'dsl', 'dsh', or 'dnn'")
        
        # Get element data
        centers = fracture.get_element_centers()
        displacements = []
        
        for element in fracture.elements:
            if component == 'dsl':
                displacements.append(element.dsl)
            elif component == 'dsh':
                displacements.append(element.dsh)
            else:  # dnn
                displacements.append(element.dnn)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        scatter = ax.scatter(
            [c[0] for c in centers],
            [c[2] for c in centers],
            c=displacements,
            cmap='viridis',
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(title or f'Fracture {component.upper()} Displacement')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Displacement (m)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_3d_aperture(
        fracture: Fracture,
        component: str = 'dnn',
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
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
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
        
        if component not in ['dsl', 'dsh', 'dnn']:
            raise ValueError("Component must be 'dsl', 'dsh', or 'dnn'")
        
        # Get element data
        centers = fracture.get_element_centers()
        displacements = []
        
        for element in fracture.elements:
            if component == 'dsl':
                displacements.append(element.dsl)
            elif component == 'dsh':
                displacements.append(element.dsh)
            else:  # dnn
                displacements.append(element.dnn)
        
        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            [c[0] for c in centers],
            [c[1] for c in centers],
            [c[2] for c in centers],
            c=displacements,
            cmap='viridis',
            s=50
        )
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title or f'Fracture {component.upper()} Displacement (3D)')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Displacement (m)')
        
        plt.tight_layout()
        plt.show()


class FiberPlotter:
    """Plotting utilities for DAS fibers."""
    
    @staticmethod
    def plot_strain_response(
        fiber: Fiber,
        component: str = 'EXX',
        time_step: int = -1,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
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
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
        
        if component not in ['EXX', 'EYY', 'EZZ']:
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
                (positions[i][0] - positions[i-1][0])**2 +
                (positions[i][1] - positions[i-1][1])**2 +
                (positions[i][2] - positions[i-1][2])**2
            )
            distances.append(distances[-1] + dist)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(distances, np.array(strain_data) * 1e6, 'b-', linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Distance along fiber (m)')
        ax.set_ylabel('Strain (μϵ)')
        ax.set_title(title or f'DAS Fiber {component} Strain Response')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_stress_response(
        fiber: Fiber,
        component: str = 'SXX',
        time_step: int = -1,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
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
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
        
        if component not in ['SXX', 'SYY', 'SZZ', 'SXY', 'SXZ', 'SYZ']:
            raise ValueError("Component must be one of 'SXX', 'SYY', 'SZZ', 'SXY', 'SXZ', 'SYZ'")
        
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
                (positions[i][0] - positions[i-1][0])**2 +
                (positions[i][1] - positions[i-1][1])**2 +
                (positions[i][2] - positions[i-1][2])**2
            )
            distances.append(distances[-1] + dist)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(distances, np.array(stress_data) / 1e6, 'r-', linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Distance along fiber (m)')
        ax.set_ylabel('Stress (MPa)')
        ax.set_title(title or f'DAS Fiber {component} Stress Response')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_fiber_geometry(
        fiber: Fiber,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
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
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
        # Get channel positions
        positions = fiber.get_channel_positions()
        
        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot fiber path
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        z_coords = [pos[2] for pos in positions]
        
        ax.plot(x_coords, y_coords, z_coords, 'b-', linewidth=2, label='Fiber path')
        ax.scatter(x_coords, y_coords, z_coords, c='red', s=20, label='Channels')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title or f'DAS Fiber {fiber.fiber_id} Geometry')
        ax.legend()
        
        plt.tight_layout()
        plt.show()

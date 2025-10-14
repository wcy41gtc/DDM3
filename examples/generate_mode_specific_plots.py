#!/usr/bin/env python3
"""
Generate mode-specific evolution plots for each example.
Each mode gets two plots: fracture dimensions and relevant stress components.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path for ddm3d imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .utils import (
    generate_geometry_and_stress_profiles,
    plot_fracture_dimensions,
    plot_opening_mode_stresses,
    plot_shear_mode_stresses,
    plot_mixed_mode_stresses
)


def main():
    """Generate mode-specific plots for all examples."""
    
    # Generate base profiles
    profiles = generate_geometry_and_stress_profiles()
    
    # Define modes and their specific parameters
    modes = {
        'opening_mode_base': {
            'title': 'Opening Mode Base',
            'stress_plotter': plot_opening_mode_stresses
        },
        'opening_mode_incline': {
            'title': 'Opening Mode Incline',
            'stress_plotter': plot_opening_mode_stresses
        },
        'shear_mode_incline': {
            'title': 'Shear Mode Incline',
            'stress_plotter': plot_shear_mode_stresses
        },
        'mixed_mode_incline': {
            'title': 'Mixed Mode Incline',
            'stress_plotter': plot_mixed_mode_stresses
        }
    }
    
    # Generate plots for each mode
    for mode_name, mode_info in modes.items():
        mode_dir = Path(mode_name)
        figures_dir = mode_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        print(f"Generating plots for {mode_name}...")
        
        # Plot 1: Fracture dimensions
        dimensions_path = figures_dir / f'{mode_name}_dimensions.png'
        plot_fracture_dimensions(profiles, dimensions_path, mode_info['title'])
        
        # Plot 2: Stress evolution (mode-specific)
        stress_path = figures_dir / f'{mode_name}_stresses.png'
        mode_info['stress_plotter'](profiles, stress_path, mode_info['title'])
        
        print(f"  ✓ Generated: {dimensions_path}")
        print(f"  ✓ Generated: {stress_path}")
    
    print("\n✅ All mode-specific plots generated successfully!")


if __name__ == "__main__":
    main()

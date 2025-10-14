#!/usr/bin/env python3
"""
Generate evolution plots for all example modes.

This script creates evolution plots showing fracture geometry and stress evolution
for each of the four example modes and saves them to the appropriate figures directories.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .utils import (
    generate_geometry_and_stress_profiles,
    plot_geometry_and_stress_evolution
)


def generate_all_evolution_plots():
    """Generate evolution plots for all example modes."""
    
    # Define the modes and their parameters
    modes = {
        "opening_mode_base": {
            "description": "Opening Mode Base (0° inclination)",
            "parameters": {},
            "stress_description": "Normal stress only (pure opening mode)"
        },
        "opening_mode_incline": {
            "description": "Opening Mode Incline (-30° inclination)",
            "parameters": {},
            "stress_description": "Normal stress only (pure opening mode)"
        },
        "shear_mode_incline": {
            "description": "Shear Mode Incline (-30° inclination)",
            "parameters": {},
            "stress_description": "Shear stress only (fault slip)"
        },
        "mixed_mode_incline": {
            "description": "Mixed Mode Incline (-30° inclination)",
            "parameters": {},
            "stress_description": "Combined normal and shear stress"
        }
    }
    
    print("Generating evolution plots for all example modes...")
    print("=" * 60)
    
    for mode, config in modes.items():
        print(f"\nGenerating plots for {config['description']}...")
        
        # Create figures directory if it doesn't exist
        figures_dir = f"{mode}/figures"
        os.makedirs(figures_dir, exist_ok=True)
        
        # Generate profiles (all modes use the same base parameters)
        profiles = generate_geometry_and_stress_profiles(**config["parameters"])
        
        # Generate evolution plot
        plot_path = os.path.join(figures_dir, f"{mode}_evolution.png")
        plot_geometry_and_stress_evolution(
            profiles, 
            save_path=plot_path
        )
        
        print(f"  ✓ Saved: {plot_path}")
        
        # Also create a comparison plot showing all modes
        if mode == "mixed_mode_incline":  # Generate comparison plot last
            print(f"\nGenerating comparison plot...")
            generate_comparison_plot()
    
    print(f"\n" + "=" * 60)
    print("All evolution plots generated successfully!")
    print("\nGenerated files:")
    for mode in modes.keys():
        print(f"  - {mode}/figures/{mode}_evolution.png")
    print("  - figures/all_modes_comparison.png")


def generate_comparison_plot():
    """Generate a comparison plot showing all modes."""
    
    # Create main figures directory
    os.makedirs("figures", exist_ok=True)
    
    # Generate profiles for all modes
    profiles_all = generate_geometry_and_stress_profiles()
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Fracture Evolution Comparison - All Modes', fontsize=16, fontweight='bold')
    
    # Plot fracture length evolution
    axes[0, 0].plot(profiles_all['l'], 'b-', linewidth=2, label='Length')
    axes[0, 0].set_title('Fracture Length Evolution')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Length (m)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot fracture height evolution
    axes[0, 1].plot(profiles_all['h'], 'r-', linewidth=2, label='Height')
    axes[0, 1].set_title('Fracture Height Evolution')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Height (m)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot normal stress evolution
    axes[1, 0].plot(profiles_all['snn']/1e6, 'g-', linewidth=2, label='Normal Stress')
    axes[1, 0].set_title('Normal Stress Evolution')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Normal Stress (MPa)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot shear stress evolution
    axes[1, 1].plot(profiles_all['ssl']/1e6, 'm-', linewidth=2, label='Shear Stress')
    axes[1, 1].set_title('Shear Stress Evolution')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Shear Stress (MPa)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_path = "figures/all_modes_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {comparison_path}")


if __name__ == "__main__":
    generate_all_evolution_plots()

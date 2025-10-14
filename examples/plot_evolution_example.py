#!/usr/bin/env python3
"""
Example script demonstrating how to use the evolution plotting functions.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .utils import (
    generate_geometry_and_stress_profiles,
    plot_geometry_and_stress_evolution,
    plot_evolution_comparison
)

def main():
    """Demonstrate evolution plotting functions."""
    print("DDM3D Evolution Plotting Example")
    print("=" * 40)
    
    # Create output directory
    output_dir = "examples/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Example 1: Single profile with default parameters
    print("\n1. Generating single evolution profile...")
    profiles_default = generate_geometry_and_stress_profiles()
    
    # Plot single profile
    plot_path = os.path.join(output_dir, "single_evolution.png")
    plot_geometry_and_stress_evolution(profiles_default, save_path=plot_path)
    
    # Example 2: Custom parameters
    print("\n2. Generating custom evolution profile...")
    profiles_custom = generate_geometry_and_stress_profiles(
        bsdt=40.0,         # 40 time steps before shut-in
        asdt=20.0,         # 20 time steps after shut-in
        l_scale=25.0,      # Larger length scale
        h_scale=12.0,      # Larger height scale
        nn_scale=1.0e6,    # Higher normal stress scale
        ss_scale=1.2e6     # Higher shear stress scale
    )
    
    # Plot custom profile
    custom_plot_path = os.path.join(output_dir, "custom_evolution.png")
    plot_geometry_and_stress_evolution(profiles_custom, save_path=custom_plot_path)
    
    # Example 3: Compare different scenarios
    print("\n3. Comparing different evolution scenarios...")
    
    # Generate different scenarios
    scenarios = {
        "Conservative": generate_geometry_and_stress_profiles(
            l_scale=15.0, h_scale=8.0, nn_scale=0.6e6, ss_scale=0.8e6
        ),
        "Moderate": generate_geometry_and_stress_profiles(
            l_scale=20.0, h_scale=10.0, nn_scale=0.8e6, ss_scale=1.0e6
        ),
        "Aggressive": generate_geometry_and_stress_profiles(
            l_scale=30.0, h_scale=15.0, nn_scale=1.2e6, ss_scale=1.5e6
        )
    }
    
    # Create comparison plot
    profiles_list = list(scenarios.values())
    labels = list(scenarios.keys())
    comparison_path = os.path.join(output_dir, "scenario_comparison.png")
    plot_evolution_comparison(profiles_list, labels, save_path=comparison_path)
    
    # Example 4: Show parameter effects
    print("\n4. Analyzing parameter effects...")
    
    # Test different time scales
    time_scenarios = {
        "Slow Evolution": generate_geometry_and_stress_profiles(bsdt=80.0, asdt=40.0),
        "Default": generate_geometry_and_stress_profiles(bsdt=60.0, asdt=30.0),
        "Fast Evolution": generate_geometry_and_stress_profiles(bsdt=40.0, asdt=20.0)
    }
    
    # Create time comparison plot
    time_profiles_list = list(time_scenarios.values())
    time_labels = list(time_scenarios.keys())
    time_comparison_path = os.path.join(output_dir, "time_scale_comparison.png")
    plot_evolution_comparison(time_profiles_list, time_labels, save_path=time_comparison_path)
    
    print(f"\nAll plots saved to: {output_dir}/")
    print("\nGenerated files:")
    print("- single_evolution.png: Default parameter evolution")
    print("- custom_evolution.png: Custom parameter evolution")
    print("- scenario_comparison.png: Conservative vs Moderate vs Aggressive")
    print("- time_scale_comparison.png: Different time scale effects")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()

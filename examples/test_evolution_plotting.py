#!/usr/bin/env python3
"""
Test script for the geometry and stress evolution plotting functions.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.fracture_evolution_workflow import (
    generate_geometry_and_stress_profiles,
    plot_geometry_and_stress_evolution,
    plot_evolution_comparison
)

def test_evolution_plotting():
    """Test the evolution plotting functions."""
    print("Testing geometry and stress evolution plotting...")
    
    # Create output directory
    output_dir = "examples/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test 1: Single profile plotting
    print("\n1. Testing single profile plotting...")
    profiles_default = generate_geometry_and_stress_profiles()
    
    print(f"Generated profiles with {len(profiles_default['l'])} time steps")
    print(f"Length range: {profiles_default['l'].min():.2f} - {profiles_default['l'].max():.2f} m")
    print(f"Height range: {profiles_default['h'].min():.2f} - {profiles_default['h'].max():.2f} m")
    print(f"Normal stress range: {profiles_default['snn'].min()/1e6:.2f} - {profiles_default['snn'].max()/1e6:.2f} MPa")
    print(f"Shear stress (ssl) range: {profiles_default['ssl'].min()/1e6:.2f} - {profiles_default['ssl'].max()/1e6:.2f} MPa")
    
    # Plot and save single profile
    plot_path = os.path.join(output_dir, "geometry_and_stress_evolution.png")
    plot_geometry_and_stress_evolution(profiles_default, save_path=plot_path)
    
    # Test 2: Multiple profiles comparison
    print("\n2. Testing multiple profiles comparison...")
    
    # Generate different profiles with different parameters
    profiles_small = generate_geometry_and_stress_profiles(
        l_scale=10, h_scale=5, nn_scale=0.4e6, ss_scale=0.5e6
    )
    
    profiles_large = generate_geometry_and_stress_profiles(
        l_scale=30, h_scale=15, nn_scale=1.2e6, ss_scale=1.5e6
    )
    
    profiles_fast = generate_geometry_and_stress_profiles(
        bsdt=30, asdt=15  # Faster evolution
    )
    
    # Create comparison plot
    profiles_list = [profiles_default, profiles_small, profiles_large, profiles_fast]
    labels = ["Default", "Small Scale", "Large Scale", "Fast Evolution"]
    
    comparison_path = os.path.join(output_dir, "evolution_comparison.png")
    plot_evolution_comparison(profiles_list, labels, save_path=comparison_path)
    
    print("\nEvolution plotting tests completed successfully!")
    print(f"Plots saved to: {output_dir}/")

if __name__ == "__main__":
    test_evolution_plotting()

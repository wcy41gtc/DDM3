# Opening Mode Base Example

This example demonstrates the **opening mode base case** with a fracture oriented at 0 degrees (vertical fracture). The fracture grows over time with only normal stress (opening mode) and no shear stress.

## Overview

- **Fracture Orientation**: 0 degrees (vertical, no inclination)
- **Stress Mode**: Normal stress only (pure opening mode)
- **Fracture Center**: (0, 0, 0)
- **Time Steps**: 90 (60 before shut-in + 30 after shut-in)
- **Fracture Growth**: Square root growth pattern

## Parameters

### Fracture Geometry
- **Length Scale**: 20.0 m
- **Height Scale**: 10.0 m
- **Element Count**: 10 × 10 elements
- **Growth Pattern**: `length = 20 * sqrt(time)`, `height = 10 * sqrt(time)`

### Stress Profile
- **Normal Stress (Snn)**: 
  - Before shut-in: `0.8e6 * arctan(time)` (increasing)
  - After shut-in: Exponential decay
- **Shear Stress (Ssl, Ssh)**: 0 (pure opening mode)

### Material Properties
- **Shear Modulus**: 10 GPa
- **Poisson's Ratio**: 0.25

## Fiber Network

Three DAS fibers are deployed for monitoring:

1. **Fiber 1**: Across the fracture
   - Start: (50, 100, 0)
   - End: (50, -100, 0)
   - Channels: 200

2. **Fiber 2**: Parallel to fracture (close)
   - Start: (50, 10, -100)
   - End: (50, 10, 100)
   - Channels: 200

3. **Fiber 3**: Parallel to fracture (far)
   - Start: (50, 50, -100)
   - End: (50, 50, 100)
   - Channels: 200

## Fiber and Fracture Dimensions

The following figure illustrates the fracture and fiber configuration for the opening mode base case.

![Base Fracture and Fiber Configuration](figures/base_fracture_fiber_config.png)

- **Fracture Center**: (0, 0, 0)
- **Fracture Orientation**: o1=0°, o2=0°, o3=0° (vertical fracture)
- **Block Dimensions**: 400m × 400m × 200m
- **Fiber Layout**: Three fibers positioned to capture different aspects of the vertical fracture response

## Usage

### Basic Usage
```bash
python run_opening_mode_base.py
```

### With Custom Parameters
```bash
# Force recalculation
python run_opening_mode_base.py --recalculate

# Custom gauge length for interpolation
python run_opening_mode_base.py --gauge_length 5.0

# Both options
python run_opening_mode_base.py --recalculate --gauge_length 5.0
```

### Command Line Options
- `--recalculate` or `-r`: Force recalculation instead of loading from HDF5 files
- `--gauge_length` or `-gl`: Channel spacing for interpolation (default: 10.0 meters)
- `--scale` or `-s`: Scale factor for the data for plotting (default: 20.0)

## Output

The script generates:

### HDF5 Files
- `./results/opening_mode_base/opening_mode_base_fiber_1.h5`
- `./results/opening_mode_base/opening_mode_base_fiber_2.h5`
- `./results/opening_mode_base/opening_mode_base_fiber_3.h5`

### Plots
- `./results/opening_mode_base/opening_mode_base_fiber_1_EYY_U.png` - Strain contour
- `./results/opening_mode_base/opening_mode_base_fiber_1_EYY_U_Rate.png` - Strain rate contour

## Fracture Geometry Evolution and Stress Profiles

The simulation uses synthetic geometry evolution and stress profiles derived from pure mathematic approximations, which mimics the real-world cases. The code is capable and the users are encouraged to use physics-based fracture geometry and stress evolution data. The following plots show the fracture geometry and stress evolution for the opening mode base case:

### Fracture Dimensions Evolution

![Fracture Dimensions - Opening Mode Base](figures/opening_mode_base_dimensions.png)

**Fracture Geometry Characteristics:**
- **Length**: Square root growth from 0 to ~20m over 60 time steps, then constant
- **Height**: Square root growth from 0 to ~10m over 60 time steps, then constant
- **Growth Pattern**: Highest growth rate at early times, decreasing as fracture grows
- **Orientation**: 0°, 0°, 0° in o1, o2, o3 (vertical fracture)

### Normal Stress Evolution

![Normal Stress - Opening Mode Base](figures/opening_mode_base_stresses.png)

**Stress Characteristics:**
- **Normal Stress (Snn)**: 
  - Increases with arctangent function during growth phase (0-60 time steps (minutes in this simulation), the code does not enforce the unit of time steps)
  - Exponential decay after shut-in (60-90 time steps)
  - Peak stress at shut-in time
- **Shear Stresses**: Zero throughout (pure opening mode)
- **Pure Opening**: Only normal stress component, no shear components

## Expected Results

### Fracture Behavior
- **Pure Opening**: No shear displacement, only normal opening
- **Symmetric Growth**: Fracture grows symmetrically in all directions
- **Stress Evolution**: Normal stress increases during growth, then decays after shut-in

### DAS Response
- **Fiber 1** (across fracture): Strongest response due to direct crossing
- **Fiber 2** (close parallel): Moderate response from proximity
- **Fiber 3** (far parallel): Weakest response due to distance

### Time-Space Patterns
- **Strain**: Positive values during opening, negative during closing
- **Strain Rate**: High rates during active growth, low rates during stable periods
- **Spatial Distribution**: Concentrated near fracture center, decaying with distance

## Physical Interpretation

This example represents a **vertical hydraulic fracture** in a homogeneous medium with:
- **No tectonic stress**: Pure hydraulic opening
- **No faulting**: No shear components
- **Symmetric growth**: Equal growth in all directions
- **Standard DAS monitoring**: Typical fiber deployment for vertical hydraulic fracturing

## Comparison with Other Modes

- **vs. Opening Mode Incline**: This has no inclination (0° vs -30°)
- **vs. Shear Mode**: This has no shear stress (pure opening vs pure shear)
- **vs. Mixed Mode**: This has only normal stress (opening vs opening+shear)

## Troubleshooting

### Common Issues
1. **Memory Issues**: Reduce time steps or use smaller gauge_length
2. **File Not Found**: Ensure DDM3D package is properly installed
3. **Plot Errors**: Check matplotlib backend and display settings

### Performance Tips
- Use `--gauge_length` to control interpolation resolution
- Use HDF5 files for repeated runs (don't use `--recalculate`)
- Monitor memory usage for large simulations

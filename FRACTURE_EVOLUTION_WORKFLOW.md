# Fracture Evolution Workflow

This document describes the complete workflow for simulating fracture evolution using the DDM3D package, following the original DDM3D methodology.

## Overview

The workflow simulates the evolution of fractures over time with different stress modes:
- **opening_mode_base**: Fracture with 0° strike angle
- **opening_mode**: Fracture with -30° strike angle  
- **shear_mode**: Fracture with shear stress loading
- **mixed_mode**: Fracture with combined shear and normal stress loading

## Workflow Steps

### 1. Geometry and Stress Profile Generation

The workflow starts by generating artificial stress profiles that define how the fracture evolves over time:

```python
def generate_geometry_and_stress_profiles(
    bsdt=60, asdt=30, l_scale_base=60, l_scale=20, h_scale=10, 
    nl=10, nh=10, nn_scale=0.8e6, ss_scale=1.0e6
):
    """
    Generate artificial stress profiles for fracture evolution.
    
    Parameters:
    bsdt: before shut in dt, number of time steps before shut in (unitless)
    asdt: after shut in dt, number of time steps after shut in (unitless)
    l_scale_base: base length scale (m)
    l_scale: length scale (m)
    h_scale: height scale (m)
    nl: number of elements along l
    nh: number of elements along h
    nn_scale: normal stress scale (Pa)
    ss_scale: shear stress scale (Pa)
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
    _ssl_after_shutin = -ss_scale * np.exp(0.01 * base_for_ssl_increase_after_shutin) + ss_scale
    ssl = -np.concatenate((_ssl_before_shutin, _ssl_after_shutin))

    # Assemble shear stress profile ssh
    ssh = np.zeros(bsdt + asdt)
    
    # Assemble normal stress profile snn
    snn_before_shutin = nn_scale * np.arctan(base_for_snn_increase_before_shutin)
    snn_after_shutin = snn_before_shutin[-1] * np.exp(-1 * base_for_snn_decay_after_shutin)
    snn = np.concatenate((snn_before_shutin, snn_after_shutin))
    
    return {
        'l': l, 'h': h, 'dl': dl, 'dh': dh,
        'ssl': ssl, 'ssh': ssh, 'snn': snn
    }
```

### 2. Fracture Series Creation

For each mode, a series of fractures is created following the original DDM3D `make_fracture` function signature:

```python
def make_fracture(
    tot_l, tot_h, dl, dh,           # Fracture dimensions and element sizes
    c_x, c_y, c_z,                  # Center coordinates
    sk, dp, yw,                     # Strike, dip, yaw angles
    dsl, dsh, dnn,                  # Displacement discontinuities
    Ssl, Ssh, Snn,                  # Stress components
    material                         # Material properties
):
```

#### Mode-Specific Fracture Creation:

**Opening Mode Base (0° strike):**
```python
# Fracture center and orientation
x = 0.0
y = 0.0
z = 0.0
sk = 0.0
dp = 0.0
yw = 0.0

for i in range(90):
    growing_fracture.append([make_fracture(
        profiles['l'][i], profiles['h'][i], profiles['dl'][i], profiles['dh'][i],
        x, y, z,                     # c_x, c_y, c_z
        sk, dp, yw,                  # sk, dp, yw
        0.0, 0.0, 0.0,              # dsl, dsh, dnn
        0.0, 0.0, profiles['snn'][i], # Ssl, Ssh, Snn
        material
    )])
```

**Opening Mode (-30° strike):**
```python
# Fracture center and orientation
x = 0
y = -(50) * np.tan(np.deg2rad(30))
z = 0
sk = -30.0
dp = 0.0
yw = 0.0

for i in range(90):
    growing_fracture.append([make_fracture(
        profiles['l'][i], profiles['h'][i], profiles['dl'][i], profiles['dh'][i],
        x, y, z,                     # c_x, c_y, c_z
        sk, dp, yw,                  # sk, dp, yw
        0.0, 0.0, 0.0,              # dsl, dsh, dnn
        0.0, 0.0, profiles['snn'][i], # Ssl, Ssh, Snn
        material
    )])
```

**Shear Mode (30 time steps):**
```python
# Fracture center and orientation
x = 0
y = -(50) * np.tan(np.deg2rad(30))
z = 0
sk = -30.0
dp = 0.0
yw = 0.0

for i in range(30):
    growing_fracture.append([make_fracture(
        profiles['l'][i], profiles['h'][i], profiles['dl'][i], profiles['dh'][i],
        x, y, z,                     # c_x, c_y, c_z
        sk, dp, yw,                  # sk, dp, yw
        0.0, 0.0, 0.0,              # dsl, dsh, dnn
        profiles['ssl'][i], 0.0, 0.0, # Ssl, Ssh, Snn
        material
    )])
```

**Mixed Mode:**
```python
# Fracture center and orientation
x = 0
y = -(50) * np.tan(np.deg2rad(30))
z = 0
sk = -30.0
dp = 0.0
yw = 0.0

for i in range(90):
    growing_fracture.append([make_fracture(
        profiles['l'][i], profiles['h'][i], profiles['dl'][i], profiles['dh'][i],
        x, y, z,                     # c_x, c_y, c_z
        sk, dp, yw,                  # sk, dp, yw
        0.0, 0.0, 0.0,              # dsl, dsh, dnn
        profiles['ssl'][i], 0.0, profiles['snn'][i], # Ssl, Ssh, Snn
        material
    )])
```

### 3. DDM Calculator Processing

For each time step:

1. **Solve Displacement Discontinuities**: The DDM calculator solves for the displacement discontinuities in all fracture elements
2. **Calculate Fiber Responses**: Stress, displacement, and strain fields are calculated at all DAS fiber channels
3. **Store Time Series Data**: Results are stored in the Fiber objects for later analysis

```python
calculator = DDMCalculator()

for i, fractures in enumerate(fractures_series):
    # Solve displacement discontinuities
    calculator.solve_displacement_discontinuities(fractures)
    
    # Calculate fiber responses
    calculator.calculate_fiber_response(fractures, fibers)
    
    # Store time step information
    for fiber in fibers:
        fiber.add_time_step(i)
```

### 4. Results Storage

Results are saved in two formats:

#### HDF5 Files
- Channel positions
- Time series stress data (SXX, SYY, SZZ, SXY, SXZ, SYZ)
- Time series strain data (EXX, EYY, EZZ)
- Time series displacement data (UXX, UYY, UZZ)

#### Plot Files
- Strain evolution plots for each fiber
- Stress evolution plots for each fiber
- Plots show time series data for selected channels

## Usage

### Running Individual Modes

```python
from examples.fracture_evolution_workflow import (
    run_opening_mode_base,
    run_opening_mode,
    run_shear_mode,
    run_mixed_mode
)

# Run individual modes
run_opening_mode_base()
run_opening_mode()
run_shear_mode()
run_mixed_mode()
```

### Running All Modes

```python
from examples.fracture_evolution_workflow import main

# Run all four modes
main()
```

### Custom Testing

```python
from examples.fracture_evolution_workflow import (
    generate_geometry_and_stress_profiles,
    create_fracture_series,
    create_fiber_network,
    calculate_fracture_evolution,
    save_results
)
from ddm3d import Material

# Generate profiles
profiles = generate_geometry_and_stress_profiles()
material = Material(shear_modulus=10e9, poisson_ratio=0.25)

# Create limited fracture series for testing
fractures_series = []
for i in range(5):  # Reduced time steps
    fracture = create_fracture_series('opening_mode_base', profiles, material)[i]
    fractures_series.append(fracture)

# Create fiber network
fibers = create_fiber_network()

# Calculate evolution
calculate_fracture_evolution(fractures_series, fibers, 'test_mode')

# Save results
save_results(fibers, 'test_mode')
```

## Output Files

The workflow generates the following files in the `results/` directory:

### HDF5 Files
- `{mode}_fiber_{id}.h5`: Contains all time series data for each fiber

### Plot Files
- `{mode}_fiber_{id}_EXX.png`: Strain evolution plots (EXX component)
- `{mode}_fiber_{id}_SXX.png`: Stress evolution plots (SXX component)

### Fiber Network Configuration
The workflow uses a 4-fiber monitoring network matching the original DDM3D configuration:

1. **Fiber 1**: Parallel to fracture at (50, 10, -100) to (50, 10, 100)
2. **Fiber 2**: Parallel to fracture at (50, 50, -100) to (50, 50, 100)  
3. **Fiber 3**: Across above fracture at (50, -100, 50) to (50, 100, 50)
4. **Fiber 4**: Across fracture at (50, 100, 0) to (50, -100, 0)

Each fiber has 200 channels for high-resolution monitoring.

## Key Features

1. **Exact Workflow Match**: Follows the original DDM3D workflow exactly
2. **Multiple Stress Modes**: Supports opening, shear, and mixed mode loading
3. **Time Series Data**: Complete temporal evolution of fracture and fiber responses
4. **Flexible Output**: Both HDF5 and plot formats for analysis
5. **Modular Design**: Easy to customize for different scenarios

## Dependencies

- NumPy: Numerical calculations
- Matplotlib: Plotting (optional)
- H5Py: HDF5 file I/O
- DDM3D: Core displacement discontinuity method calculations

## Notes

- The workflow includes warning suppression for numerical divide-by-zero warnings that are normal in DDM calculations
- Results are saved without displaying plots (non-interactive mode)
- The workflow is designed to handle large time series efficiently
- All calculations use the object-oriented DDM3D implementation while maintaining compatibility with the original workflow

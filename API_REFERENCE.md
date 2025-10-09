# DDM3D API Reference

This document provides a comprehensive reference for the DDM3D API.

## Core Classes

### Material

Represents material properties for geological formations.

```python
from ddm3d import Material

material = Material(
    shear_modulus=10e9,  # Pa
    poisson_ratio=0.25
)
```

**Parameters:**
- `shear_modulus` (float): Shear modulus in Pascals
- `poisson_ratio` (float): Poisson's ratio (dimensionless)

### Fracture

Represents a 3D fracture with displacement discontinuity elements.

#### Creating Fractures

```python
from ddm3d import Fracture

# Rectangular fracture
fracture = Fracture.create_rectangular(
    fracture_id=1,
    center=(0, 0, 0),           # (x, y, z) in meters
    length=100,                 # meters
    height=50,                  # meters
    element_size=(5, 5),        # (length, height) in meters
    orientation=(0, 90, 0),     # (strike, dip, yaw) in degrees
    material=material,
    initial_stress=(0, 0, 0)    # (Ssl, Ssh, Snn) in Pa
)
```

**Parameters:**
- `fracture_id` (int): Unique identifier
- `center` (tuple): Center coordinates (x, y, z)
- `length` (float): Fracture length in meters
- `height` (float): Fracture height in meters
- `element_size` (tuple): Element dimensions (length, height)
- `orientation` (tuple): Orientation angles (strike, dip, yaw)
- `material` (Material): Material properties
- `initial_stress` (tuple): Initial stress components

#### Fracture Methods

```python
# Get element centers
centers = fracture.get_element_centers()

# Get number of elements
n_elements = fracture.n_elements

# Access elements
for element in fracture.elements:
    print(f"Element stress: {element.Ssl}, {element.Ssh}, {element.Snn}")
```

### Fiber

Represents a DAS fiber optic cable with measurement channels.

#### Creating Fibers

```python
from ddm3d import Fiber

# Linear fiber
fiber = Fiber.create_linear(
    fiber_id=1,
    start=(50, 10, -100),       # Start coordinates
    end=(50, 10, 100),          # End coordinates
    n_channels=200              # Number of channels
)

# Curved fiber
fiber = Fiber.create_curved(
    fiber_id=2,
    points=[(0, 0, 0), (50, 0, 0), (50, 50, 0)],  # Waypoints
    n_channels=100
)
```

#### Fiber Methods

```python
# Get channel positions
positions = fiber.get_channel_positions()

# Get specific channel
channel = fiber.get_channel(channel_id=1)

# Get total length
length = fiber.get_total_length()

# Clear all data
fiber.clear_all_data()

# Add time step marker
fiber.add_time_step(time_step=0)
```

### Channel

Represents a single measurement point along a DAS fiber.

```python
# Access channel data
channel = fiber.channels[0]

# Get stress data
stress_data = channel.get_stress_data('SXX')  # or 'SYY', 'SZZ', etc.

# Get strain data
strain_data = channel.get_strain_data('EXX')  # or 'EYY', 'EZZ'

# Get displacement data
disp_data = channel.get_displacement_data('UXX')  # or 'UYY', 'UZZ'

# Get all data
all_stress = channel.get_stress_data()  # Returns all components
all_strain = channel.get_strain_data()
all_disp = channel.get_displacement_data()
```

## Calculation Engine

### DDMCalculator

Main calculation engine for solving displacement discontinuity problems.

```python
from ddm3d import DDMCalculator

calculator = DDMCalculator(tolerance=1e-10)

# Solve displacement discontinuities
calculator.solve_displacement_discontinuities([fracture])

# Calculate fiber response
calculator.calculate_fiber_response([fracture], [fiber])

# Calculate plane response (for monitoring planes)
calculator.calculate_plane_response([fracture], [plane])
```

**Parameters:**
- `tolerance` (float): Numerical tolerance for calculations

## Visualization

### FracturePlotter

Plotting utilities for fractures.

```python
from ddm3d import FracturePlotter

# Plot aperture (displacement discontinuities)
FracturePlotter.plot_aperture(
    fracture,
    component='dnn',           # 'dsl', 'dsh', or 'dnn'
    title='Fracture Aperture',
    figsize=(10, 8),
    save_path='aperture.png'
)

# 3D aperture plot
FracturePlotter.plot_3d_aperture(
    fracture,
    component='dnn',
    title='3D Fracture Aperture',
    figsize=(12, 8)
)
```

### FiberPlotter

Plotting utilities for DAS fibers with dynamic interpolation.

#### Basic Plots

```python
from ddm3d import FiberPlotter

# Strain response along fiber
FiberPlotter.plot_strain_response(
    fiber,
    component='EXX',           # 'EXX', 'EYY', or 'EZZ'
    time_step=-1,              # -1 for last time step
    title='Strain Response',
    figsize=(12, 6),
    save_path='strain.png'
)

# Stress response along fiber
FiberPlotter.plot_stress_response(
    fiber,
    component='SXX',           # Any stress component
    time_step=-1,
    title='Stress Response',
    figsize=(12, 6),
    save_path='stress.png'
)

# Fiber geometry in 3D
FiberPlotter.plot_fiber_geometry(
    fiber,
    title='Fiber Geometry',
    figsize=(10, 8),
    save_path='geometry.png'
)
```

#### Time-Space Contour Plots (NEW)

The most powerful feature for DAS analysis with dynamic interpolation:

```python
# Basic contour plot
FiberPlotter.plot_fiber_contour(
    fiber,
    component='EXX',           # Any component
    scale=1.0,                 # Scale factor
    gauge_length=None,         # Use original spacing
    figsize=(12, 8),
    save_path='contour.png'
)

# With dynamic interpolation
FiberPlotter.plot_fiber_contour(
    fiber,
    component='EYY_U',         # Strain from displacement
    scale=20.0,
    gauge_length=10.0,         # 10m channel spacing
    figsize=(12, 8),
    save_path='interpolated.png'
)
```

**Supported Components:**
- **Stress**: `SXX`, `SYY`, `SZZ`, `SXY`, `SXZ`, `SYZ`
- **Displacement**: `UXX`, `UYY`, `UZZ`
- **Strain**: `EXX`, `EYY`, `EZZ`
- **Strain from Displacement**: `EXX_U`, `EYY_U`, `EZZ_U`
- **Strain Rate**: `EXX_Rate`, `EYY_Rate`, `EZZ_Rate`
- **Strain Rate from Displacement**: `EXX_U_Rate`, `EYY_U_Rate`, `EZZ_U_Rate`

**Dynamic Interpolation Examples:**

```python
# Original fiber: 200m length, 200 channels (1m spacing)
fiber = Fiber.create_linear(
    start=(50, 10, -100),
    end=(50, 10, 100),
    n_channels=200
)

# Different interpolation options
FiberPlotter.plot_fiber_contour(fiber, 'EYY_U', gauge_length=None)    # Original 1m
FiberPlotter.plot_fiber_contour(fiber, 'EYY_U', gauge_length=2.0)     # 2m spacing
FiberPlotter.plot_fiber_contour(fiber, 'EYY_U', gauge_length=5.0)     # 5m spacing
FiberPlotter.plot_fiber_contour(fiber, 'EYY_U', gauge_length=10.0)    # 10m spacing
FiberPlotter.plot_fiber_contour(fiber, 'EYY_U', gauge_length=20.0)    # 20m spacing
```

## Fracture Evolution Workflow

Complete workflow for time-series fracture simulation:

```python
from examples.fracture_evolution_workflow import (
    generate_geometry_and_stress_profiles,
    create_fracture_series,
    create_fiber_network,
    calculate_fracture_evolution,
    save_results
)

# Generate stress profiles
profiles = generate_geometry_and_stress_profiles(
    bsdt=60,                    # Before shut-in time steps
    asdt=30,                    # After shut-in time steps
    l_scale_base=60,            # Base length scale
    l_scale=20,                 # Length scale
    h_scale=10,                 # Height scale
    nn_scale=0.8e6,             # Normal stress scale
    ss_scale=1.0e6              # Shear stress scale
)

# Create material
material = Material(shear_modulus=10e9, poisson_ratio=0.25)

# Create fracture series for different modes
fractures_series = create_fracture_series('opening_mode_base', profiles, material)

# Create fiber network
fibers = create_fiber_network()

# Calculate evolution
calculate_fracture_evolution(fractures_series, fibers, 'opening_mode_base')

# Save results
save_results(fibers, 'opening_mode_base')
```

**Available Modes:**
- `opening_mode_base`: 0° strike angle
- `opening_mode`: -30° strike angle
- `shear_mode`: Shear stress loading
- `mixed_mode`: Combined shear and normal stress

## Error Handling

```python
# Common exceptions
from ddm3d import DDM3DError

try:
    # Your DDM3D code here
    pass
except ValueError as e:
    print(f"Invalid parameter: {e}")
except ImportError as e:
    print(f"Missing dependency: {e}")
```

## Performance Tips

1. **Use appropriate element sizes**: Smaller elements give better accuracy but require more computation
2. **Optimize fiber channel count**: More channels give better resolution but slower plotting
3. **Use dynamic interpolation**: Plot at different resolutions without recalculating
4. **Memory management**: Use `plt.close()` for large simulations
5. **Batch processing**: Process multiple time steps efficiently

## Examples

See the `examples/` directory for comprehensive usage examples:

- `fracture_evolution_workflow.py`: Complete workflow with 4 stress modes
- `test_opening_mode_base.py`: Individual mode testing

## Troubleshooting

### Common Issues

1. **Memory issues**: Use `plt.close()` after plotting
2. **Slow interpolation**: Reduce number of channels or use coarser gauge_length
3. **Import errors**: Install required dependencies with `pip install -e ".[all]"`
4. **Plotting issues**: Check matplotlib backend and dependencies

### Getting Help

- Check the documentation files in the repository
- Review the examples in `examples/`
- Open an issue on GitHub for bugs or feature requests

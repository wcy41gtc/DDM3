# DDM3D: 3D Displacement Discontinuity Method for DAS Simulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

DDM3D is a modern, object-oriented Python library for simulating Distributed Acoustic Sensing (DAS) responses using the 3D Displacement Discontinuity Method (DDM). It provides comprehensive tools for modeling fractures, faults, and other discontinuities in geological formations and calculating their effects on DAS fiber optic sensors.

## Features

- **Object-Oriented Design**: Clean, modular architecture with well-defined classes
- **3D Fracture Modeling**: Support for rectangular and elliptical fractures with arbitrary orientations
- **DAS Fiber Simulation**: Calculate stress and displacement fields along fiber optic cables
- **Dynamic Interpolation**: Plot fiber data at any desired channel spacing through interpolation
- **Advanced Visualization**: Comprehensive plotting tools including time-space contour plots
- **Fracture Evolution Workflow**: Complete time-series simulation with multiple stress modes
- **Professional Plotting**: Time-space contour plots matching original DDM3D format
- **Comprehensive Examples**: Four complete example modes with detailed documentation and 3D visualizations
- **Command-line Interface**: Flexible parameter control for all examples
- **HDF5 Data Storage**: Efficient time-series data storage and retrieval
- **Extensible**: Easy to add new fracture geometries and calculation methods

## Installation

```bash
# Clone the repository
git clone https://github.com/wcy41gtc/DDM3.git
cd DDM3

# Install core package
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with all optional dependencies
pip install -e ".[all]"

# Install with specific extras
pip install -e ".[optional]"  # Optional scientific packages
```

## Quick Start

```python
import numpy as np
from ddm3d import Fracture, Fiber, DDMCalculator, Material

# Define material properties
material = Material(
    shear_modulus=10e9,  # Pa
    poisson_ratio=0.25
)

# Create a fracture
fracture = Fracture.create_rectangular(
    center=(0, 0, 0),
    length=100,  # meters
    height=50,   # meters
    element_size=(5, 5),  # meters
    o1=0,        # degrees
    o2=90,       # degrees
    material=material
)

# Create a DAS fiber
fiber = Fiber.create_linear(
    start=(50, 10, -100),
    end=(50, 10, 100),
    n_channels=200
)

# Calculate displacement discontinuities
calculator = DDMCalculator()
calculator.solve_displacement_discontinuities([fracture])

# Calculate DAS response
calculator.calculate_fiber_response([fracture], [fiber])

# Visualize results
fracture.plot_aperture()
fiber.plot_strain_response()

# Create professional time-space contour plots with dynamic interpolation
from ddm3d import FiberPlotter

# Plot with original channel spacing
FiberPlotter.plot_fiber_contour(fiber, component='EXX', save_path='strain_contour.png')

# Plot with interpolated 10m channel spacing
FiberPlotter.plot_fiber_contour(fiber, component='EXX', gauge_length=10.0, save_path='strain_10m.png')

# Plot stress with 5m channel spacing
FiberPlotter.plot_fiber_contour(fiber, component='SXX', gauge_length=5.0, save_path='stress_5m.png')
```

## Documentation

### Main Documentation
- **Project Overview**: This README.md file provides comprehensive project information
- **API Reference**: Detailed docstrings throughout the codebase
- **Examples Guide**: Complete examples in the `examples/` directory with individual README files

### Examples Documentation
Each example directory contains detailed documentation:
- **3D Configuration Diagrams**: Visual fracture and fiber layouts
- **Evolution Plots**: Fracture geometry and stress evolution over time
- **Command-line Options**: Complete parameter reference
- **Expected Results**: Detailed output descriptions
- **Physical Interpretation**: Real-world applications and use cases

### Coordinate System
The package uses a custom coordinate system (o1, o2, o3) to avoid confusion with geological strike-dip-rake conventions. See the [Examples README](examples/README.md) for detailed orientation definitions.

## Examples

The `examples/` directory contains comprehensive, ready-to-run examples demonstrating different fracture modes and their DAS responses. Each example includes complete Python scripts, detailed documentation, and command-line interfaces.

### Quick Start with Examples

```bash
# Navigate to any example directory
cd examples/opening_mode_base

# Run the simulation
python run_opening_mode_base.py

# With custom parameters
python run_opening_mode_base.py --recalculate --gauge_length 5.0 --scale 20.0
```

### Available Examples

1. **[Opening Mode Base](examples/opening_mode_base/)** - Vertical fracture with pure opening (0°, 0°, 0°)
2. **[Opening Mode Incline](examples/opening_mode_incline/)** - Inclined fracture with pure opening (-30°, 0°, 0°)
3. **[Shear Mode Incline](examples/shear_mode_incline/)** - Inclined fracture with pure shear slip (-30°, 0°, 0°)
4. **[Mixed Mode Incline](examples/mixed_mode_incline/)** - Inclined fracture with combined opening and shear (-30°, 0°, 0°)

### Example Features

Each example includes:
- **Complete Python Script** (`run_*.py`) - Ready to run simulation
- **Detailed README.md** - Comprehensive documentation with 3D configuration diagrams
- **Command-line Interface** - Flexible parameter control with `--recalculate` and `--gauge_length` options
- **Evolution Plots** - Fracture geometry and stress evolution visualizations
- **HDF5 Output** - Time-series data storage for efficient analysis

### Fracture Evolution Workflow

The examples demonstrate a complete workflow for simulating fracture evolution over time with different stress modes:

```python
from examples.utils import run_mode_simulation

# Run individual modes
run_mode_simulation('opening_mode_base')
run_mode_simulation('opening_mode_incline')
run_mode_simulation('shear_mode_incline')
run_mode_simulation('mixed_mode_incline')
```

The workflow generates:
- Time-series HDF5 data files for each fiber
- Time-space contour plots with dynamic interpolation
- Strain and strain rate plots (EYY_U, EZZ_U, EYY_U_Rate, EZZ_U_Rate)
- Complete fracture evolution analysis with 4 different stress modes
- 3D configuration diagrams showing fracture and fiber layouts

### Dynamic Interpolation

The new dynamic interpolation feature allows you to plot fiber data at any desired channel spacing:

```python
# Original fiber: 200m length, 200 channels (1m spacing)
fiber = Fiber.create_linear(
    start=(50, 10, -100),
    end=(50, 10, 100),
    n_channels=200
)

# Plot with different channel spacings
FiberPlotter.plot_fiber_contour(fiber, component='EYY_U', gauge_length=None)    # Original 1m spacing
FiberPlotter.plot_fiber_contour(fiber, component='EYY_U', gauge_length=5.0)     # 5m spacing
FiberPlotter.plot_fiber_contour(fiber, component='EYY_U', gauge_length=10.0)    # 10m spacing
FiberPlotter.plot_fiber_contour(fiber, component='EYY_U', gauge_length=20.0)    # 20m spacing
```

This enables both high-resolution analysis and overview visualization without recalculating DDM results.

### Synthetic vs. Physics-Based Data

The examples use synthetic geometry evolution and stress profiles derived from mathematical approximations that mimic real-world cases. The code is designed to be flexible and users are encouraged to use physics-based fracture geometry and stress evolution data when available. The synthetic approach provides:

- **Educational Value**: Clear understanding of DDM principles and DAS responses
- **Reproducible Results**: Consistent baseline for comparison and validation
- **Flexible Framework**: Easy integration of real-world data and physics-based models
- **Research Foundation**: Solid base for developing more complex, physics-based simulations

## Contributing

Contributions are welcome! Please see the [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=ddm3d

```

### Development Features

- **Dynamic Interpolation**: Plot fiber data at any channel spacing
- **Memory Management**: Proper matplotlib figure cleanup for large simulations
- **Comprehensive Testing**: Full workflow testing with multiple fracture modes
- **Type Hints**: Complete type annotations for better IDE support
- **Documentation**: Extensive docstrings and examples
- **Example Framework**: Complete example system with utilities and common functions
- **3D Visualization**: Configuration diagrams for all fracture and fiber setups
- **Evolution Analysis**: Time-series fracture geometry and stress evolution plots

## Citation

The code is inspired by the the following dissertation:
```bibtex
@phdthesis{wu2014numerical,
  title={Numerical modeling of complex hydraulic fracture development in unconventional reservoirs},
  author={Wu, Kan},
  year={2014}
}
```

If you use DDM3D in your research, please cite:

```bibtex
@software{ddm3d2025,
  title={DDM3D: 3D Displacement Discontinuity Method for DAS Simulation with Comprehensive Examples and Dynamic Interpolation},
  author={DDM3D Contributors},
  year={2025},
  version={0.3.0},
  url={https://github.com/wcy41gtc/DDM3},
  license={MIT}
}
```

Please also consider citing this study where the original code was developed:

```bibtex
@article{wang2023numerical,
  title={Numerical modeling of low-frequency distributed acoustic sensing signals for mixed-mode reactivation},
  author={Wang, Chaoyi and Eaton, David W and Ma, Yuanyuan},
  journal={Geophysics},
  volume={88},
  number={6},
  pages={WC25--WC36},
  year={2023},
  publisher={Society of Exploration Geophysicists}
}
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the original DDM3D implementation
- Built with NumPy and Matplotlib
- Inspired by the geomechanics and DAS communities

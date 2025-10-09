# DDM3D: 3D Displacement Discontinuity Method for DAS Simulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

DDM3D is a modern, object-oriented Python library for simulating Distributed Acoustic Sensing (DAS) responses using the 3D Displacement Discontinuity Method (DDM). It provides tools for modeling fractures, faults, and other discontinuities in geological formations and calculating their effects on DAS fiber optic sensors.

## Features

- **Object-Oriented Design**: Clean, modular architecture with well-defined classes
- **3D Fracture Modeling**: Support for rectangular and elliptical fractures with arbitrary orientations
- **DAS Fiber Simulation**: Calculate stress and displacement fields along fiber optic cables
- **Advanced Visualization**: Comprehensive plotting tools including time-space contour plots
- **Fracture Evolution Workflow**: Complete time-series simulation with multiple stress modes
- **Professional Plotting**: Time-space contour plots matching original DDM3D format
- **High Performance**: Optimized numerical calculations using NumPy
- **Extensible**: Easy to add new fracture geometries and calculation methods

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ddm3d.git
cd ddm3d

# Install in development mode
pip install -e .

# Or install with dependencies
pip install -e ".[dev]"
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
    strike=0,    # degrees
    dip=90,      # degrees
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

# Create professional time-space contour plots
from ddm3d import FiberPlotter
FiberPlotter.plot_fiber_contour(fiber, component='EXX', save_path='strain_contour.png')
FiberPlotter.plot_fiber_contour(fiber, component='SXX', save_path='stress_contour.png')
```

## Documentation

Full documentation is available at [https://ddm3d.readthedocs.io](https://ddm3d.readthedocs.io)

## Examples

Check out the `examples/` directory for comprehensive usage examples:

- `basic_example.py` - Simple fracture modeling and DAS response calculation
- `fracture_evolution_workflow.py` - Complete fracture evolution simulation with multiple stress modes
- `test_opening_mode_base.py` - Test script for individual fracture evolution modes

### Fracture Evolution Workflow

The package includes a complete workflow for simulating fracture evolution over time with different stress modes:

```python
from examples.fracture_evolution_workflow import main

# Run all four fracture evolution modes
main()
```

Available modes:
- **opening_mode_base**: Fracture with 0° strike angle
- **opening_mode**: Fracture with -30° strike angle  
- **shear_mode**: Fracture with shear stress loading
- **mixed_mode**: Fracture with combined shear and normal stress loading

The workflow generates:
- Time-series HDF5 data files
- Professional time-space contour plots (EXX, SXX components)
- Complete fracture evolution analysis

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

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

# Run linting
black ddm3d tests
flake8 ddm3d tests

# Build documentation
cd docs
make html
```

## Citation

If you use DDM3D in your research, please cite:

```bibtex
@software{ddm3d2024,
  title={DDM3D: 3D Displacement Discontinuity Method for DAS Simulation},
  author={Your Name and Contributors},
  year={2024},
  url={https://github.com/yourusername/ddm3d},
  license={MIT}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the original DDM3D implementation
- Built with NumPy and Matplotlib
- Inspired by the geomechanics and DAS communities

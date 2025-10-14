# Contributing to DDM3D

Thank you for your interest in contributing to DDM3D! This document provides guidelines and information for contributors.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/ddm3d.git
   cd ddm3d
   ```
3. Install the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git

### Installing Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=ddm3d

# Run specific test file
pytest tests/test_basic.py
```

### Code Style

We use Black for code formatting and flake8 for linting:

```bash
# Format code
black ddm3d tests examples

# Check linting
flake8 ddm3d tests examples
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-fracture-geometry`
- `bugfix/fix-calculation-error`
- `docs/update-readme`

### Commit Messages

Follow conventional commit format:
- `feat: add elliptical fracture support`
- `fix: resolve numerical instability in DDM solver`
- `docs: update installation instructions`
- `test: add unit tests for Material class`

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request

## Code Guidelines

### Python Style

- Follow PEP 8
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and small
- Use meaningful variable names

### Documentation

- Update docstrings when changing function signatures
- Add examples to complex functions
- Update README.md for user-facing changes
- Update this file for process changes

### Testing

- Write unit tests for new functionality
- Aim for high test coverage
- Test edge cases and error conditions
- Use descriptive test names

## Project Structure

```
DDM3/
├── ddm3d/                    # Main package
│   ├── __init__.py          # Package initialization
│   ├── core/                # Core classes (Material, Fracture, Fiber, etc.)
│   ├── calculations/        # DDM calculation engine
│   ├── visualization/       # Plotting utilities and FiberPlotter
│   └── utils/              # Utility functions and geometry helpers
├── tests/                   # Test suite
├── examples/                # Comprehensive usage examples
│   ├── __init__.py         # Examples package initialization
│   ├── README.md           # Examples documentation
│   ├── utils.py            # Common utilities for examples
│   ├── basic_example.py    # Simple usage example
│   ├── fracture_evolution_workflow.py  # Complete workflow script
│   ├── test_opening_mode_base.py       # Test script
│   ├── plot_from_h5.py     # HDF5 plotting utility
│   ├── plot_evolution_example.py       # Evolution plotting example
│   ├── test_evolution_plotting.py      # Evolution plotting tests
│   ├── figures/            # Shared evolution plots
│   ├── opening_mode_base/  # Vertical fracture example
│   │   ├── run_opening_mode_base.py
│   │   ├── README.md
│   │   ├── figures/        # Mode-specific plots
│   │   └── results/        # Output files
│   ├── opening_mode_incline/  # Inclined opening fracture example
│   │   ├── run_opening_mode_incline.py
│   │   ├── README.md
│   │   └── figures/
│   ├── shear_mode_incline/    # Inclined shear fracture example
│   │   ├── run_shear_mode_incline.py
│   │   ├── README.md
│   │   └── figures/
│   └── mixed_mode_incline/    # Inclined mixed-mode fracture example
│       ├── run_mixed_mode_incline.py
│       ├── README.md
│       └── figures/
├── docs/                   # Documentation
├── backup/                 # Original code backup
│   └── original_code/      # Legacy implementation and test outputs
├── README.md              # Main project documentation
├── CONTRIBUTING.md        # This file
├── LICENSE                # MIT License
├── .gitignore            # Git ignore rules
├── setup.py              # Package setup
├── pyproject.toml        # Modern Python packaging
└── requirements.txt      # Dependencies
```

## Examples System

The project includes a comprehensive examples system with four complete fracture evolution scenarios:

### Example Structure
Each example directory contains:
- **Complete Python Script** (`run_*.py`) - Ready-to-run simulation
- **Detailed README.md** - Comprehensive documentation with 3D configuration diagrams
- **Command-line Interface** - Flexible parameter control
- **Evolution Plots** - Fracture geometry and stress evolution visualizations
- **HDF5 Output** - Time-series data storage

### Adding New Examples
When adding new examples:
1. Create a new directory following the naming convention (`mode_type_description/`)
2. Include a complete Python script with command-line interface
3. Add comprehensive README.md with 3D configuration diagrams
4. Follow the established code structure using `examples/utils.py`
5. Test thoroughly before submitting

### Example Utilities
The `examples/utils.py` file contains common functions used across all examples:
- `generate_geometry_and_stress_profiles()` - Synthetic stress profile generation
- `make_fracture()` - Fracture creation with proper parameters
- `create_fiber_network()` - Standard DAS fiber network setup
- `calculate_fracture_evolution()` - Time-series simulation workflow
- `save_results()` - HDF5 storage and plotting
- `setup_argument_parser()` - Command-line interface setup

## Areas for Contribution

### High Priority

- Performance optimization of DDM calculations
- Additional fracture geometries (circular, polygonal)
- Enhanced time-dependent fracture growth models
- Parallel processing support for large simulations
- Advanced visualization features and 3D plotting
- Physics-based fracture evolution models
- Additional DAS fiber configurations

### Medium Priority

- Enhanced input/output file formats (VTK, additional HDF5 features)
- Integration with other geomechanics tools
- Uncertainty quantification and sensitivity analysis
- Additional example scenarios and use cases
- Command-line interface improvements
- Documentation enhancements

### Low Priority

- GUI interface for interactive fracture modeling
- Web-based visualization and analysis tools
- Cloud deployment and distributed computing
- Mobile applications for field monitoring

## Reporting Issues

When reporting issues, please include:

1. Python version
2. Operating system
3. DDM3D version
4. Minimal code example that reproduces the issue
5. Expected vs. actual behavior
6. Error messages (if any)

## Getting Help

- Check existing issues and discussions
- Join our community discussions
- Contact maintainers for complex issues

## License

By contributing to DDM3D, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to DDM3D!

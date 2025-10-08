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
ddm3d/
├── ddm3d/                 # Main package
│   ├── core/             # Core classes (Material, Fracture, etc.)
│   ├── calculations/     # DDM calculation engine
│   ├── visualization/    # Plotting utilities
│   └── utils/           # Utility functions
├── tests/               # Test suite
├── examples/            # Usage examples
├── docs/               # Documentation
└── backup/             # Original code backup
```

## Areas for Contribution

### High Priority

- Performance optimization of DDM calculations
- Additional fracture geometries (circular, polygonal)
- Time-dependent fracture growth
- Parallel processing support
- Advanced visualization features

### Medium Priority

- Input/output file formats (VTK, HDF5)
- Integration with other geomechanics tools
- Uncertainty quantification
- Sensitivity analysis tools

### Low Priority

- GUI interface
- Web-based visualization
- Cloud deployment tools

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

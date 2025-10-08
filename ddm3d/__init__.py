"""
DDM3D: 3D Displacement Discontinuity Method for DAS Simulation

A modern, object-oriented Python library for simulating Distributed Acoustic Sensing (DAS)
responses using the 3D Displacement Discontinuity Method (DDM).
"""

__version__ = "0.1.0"
__author__ = "DDM3D Contributors"
__email__ = "your.email@example.com"

# Import main classes for easy access
from .core.material import Material
from .core.element import DisplacementDiscontinuityElement
from .core.fracture import Fracture
from .core.fiber import Fiber, Channel
from .core.plane import Plane
from .calculations.ddm_calculator import DDMCalculator
from .visualization.plotter import FracturePlotter, FiberPlotter

__all__ = [
    "Material",
    "DisplacementDiscontinuityElement", 
    "Fracture",
    "Fiber",
    "Channel",
    "Plane",
    "DDMCalculator",
    "FracturePlotter",
    "FiberPlotter",
]

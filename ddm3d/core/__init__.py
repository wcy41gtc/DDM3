"""Core classes for DDM3D package."""

from .material import Material
from .element import DisplacementDiscontinuityElement
from .fracture import Fracture
from .fiber import Fiber, Channel
from .plane import Plane

__all__ = [
    "Material",
    "DisplacementDiscontinuityElement",
    "Fracture", 
    "Fiber",
    "Channel",
    "Plane",
]

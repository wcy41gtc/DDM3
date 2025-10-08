"""Displacement discontinuity element for DDM3D calculations."""

from typing import Tuple, Optional
import numpy as np


class DisplacementDiscontinuityElement:
    """
    A displacement discontinuity element representing a small patch of a fracture.
    
    This class encapsulates the geometric and mechanical properties of a single
    element in a discretized fracture, including its position, orientation,
    displacement discontinuities, and stress components.
    """
    
    def __init__(
        self,
        element_id: int,
        center: Tuple[float, float, float],
        dimensions: Tuple[float, float],
        orientation: Tuple[float, float, float],
        displacement: Optional[Tuple[float, float, float]] = None,
        stress: Optional[Tuple[float, float, float]] = None
    ):
        """
        Initialize a displacement discontinuity element.
        
        Parameters
        ----------
        element_id : int
            Unique identifier for the element
        center : Tuple[float, float, float]
            Center coordinates (x, y, z) in meters
        dimensions : Tuple[float, float]
            Element dimensions (length, height) in meters
        orientation : Tuple[float, float, float]
            Element orientation (strike, dip, yaw) in degrees
        displacement : Tuple[float, float, float], optional
            Initial displacement discontinuities (dsl, dsh, dnn) in meters
        stress : Tuple[float, float, float], optional
            Initial stress components (Ssl, Ssh, Snn) in Pa
        """
        self._element_id = int(element_id)
        
        # Validate and store center coordinates
        if len(center) != 3:
            raise ValueError("Center must be a 3-tuple (x, y, z)")
        self._center = tuple(float(c) for c in center)
        
        # Validate and store dimensions
        if len(dimensions) != 2:
            raise ValueError("Dimensions must be a 2-tuple (length, height)")
        if any(d <= 0 for d in dimensions):
            raise ValueError("Dimensions must be positive")
        self._dimensions = tuple(float(d) for d in dimensions)
        
        # Validate and store orientation
        if len(orientation) != 3:
            raise ValueError("Orientation must be a 3-tuple (strike, dip, yaw)")
        self._orientation = tuple(float(o) for o in orientation)
        
        # Initialize displacement discontinuities
        if displacement is None:
            displacement = (0.0, 0.0, 0.0)
        if len(displacement) != 3:
            raise ValueError("Displacement must be a 3-tuple (dsl, dsh, dnn)")
        self._displacement = tuple(float(d) for d in displacement)
        
        # Initialize stress components
        if stress is None:
            stress = (0.0, 0.0, 0.0)
        if len(stress) != 3:
            raise ValueError("Stress must be a 3-tuple (Ssl, Ssh, Snn)")
        self._stress = tuple(float(s) for s in stress)
    
    @property
    def element_id(self) -> int:
        """Element identifier."""
        return self._element_id
    
    @property
    def center(self) -> Tuple[float, float, float]:
        """Center coordinates (x, y, z) in meters."""
        return self._center
    
    @property
    def x(self) -> float:
        """X coordinate in meters."""
        return self._center[0]
    
    @property
    def y(self) -> float:
        """Y coordinate in meters."""
        return self._center[1]
    
    @property
    def z(self) -> float:
        """Z coordinate in meters."""
        return self._center[2]
    
    @property
    def dimensions(self) -> Tuple[float, float]:
        """Element dimensions (length, height) in meters."""
        return self._dimensions
    
    @property
    def length(self) -> float:
        """Element length in meters."""
        return self._dimensions[0]
    
    @property
    def height(self) -> float:
        """Element height in meters."""
        return self._dimensions[1]
    
    @property
    def area(self) -> float:
        """Element area in square meters."""
        return self._dimensions[0] * self._dimensions[1]
    
    @property
    def orientation(self) -> Tuple[float, float, float]:
        """Element orientation (strike, dip, yaw) in degrees."""
        return self._orientation
    
    @property
    def strike(self) -> float:
        """Strike angle in degrees."""
        return self._orientation[0]
    
    @property
    def dip(self) -> float:
        """Dip angle in degrees."""
        return self._orientation[1]
    
    @property
    def yaw(self) -> float:
        """Yaw angle in degrees."""
        return self._orientation[2]
    
    @property
    def displacement(self) -> Tuple[float, float, float]:
        """Displacement discontinuities (dsl, dsh, dnn) in meters."""
        return self._displacement
    
    @property
    def dsl(self) -> float:
        """Strike-slip displacement discontinuity in meters."""
        return self._displacement[0]
    
    @property
    def dsh(self) -> float:
        """Dip-slip displacement discontinuity in meters."""
        return self._displacement[1]
    
    @property
    def dnn(self) -> float:
        """Normal displacement discontinuity (opening) in meters."""
        return self._displacement[2]
    
    @property
    def stress(self) -> Tuple[float, float, float]:
        """Stress components (Ssl, Ssh, Snn) in Pa."""
        return self._stress
    
    @property
    def Ssl(self) -> float:
        """Strike-slip stress component in Pa."""
        return self._stress[0]
    
    @property
    def Ssh(self) -> float:
        """Dip-slip stress component in Pa."""
        return self._stress[1]
    
    @property
    def Snn(self) -> float:
        """Normal stress component in Pa."""
        return self._stress[2]
    
    def set_displacement(self, displacement: Tuple[float, float, float]) -> None:
        """
        Set displacement discontinuities.
        
        Parameters
        ----------
        displacement : Tuple[float, float, float]
            Displacement discontinuities (dsl, dsh, dnn) in meters
        """
        if len(displacement) != 3:
            raise ValueError("Displacement must be a 3-tuple (dsl, dsh, dnn)")
        self._displacement = tuple(float(d) for d in displacement)
    
    def set_stress(self, stress: Tuple[float, float, float]) -> None:
        """
        Set stress components.
        
        Parameters
        ----------
        stress : Tuple[float, float, float]
            Stress components (Ssl, Ssh, Snn) in Pa
        """
        if len(stress) != 3:
            raise ValueError("Stress must be a 3-tuple (Ssl, Ssh, Snn)")
        self._stress = tuple(float(s) for s in stress)
    
    def get_rotation_matrix(self) -> np.ndarray:
        """
        Get the rotation matrix for coordinate transformation.
        
        Returns
        -------
        np.ndarray
            3x3 rotation matrix
        """
        strike_rad = np.deg2rad(self._orientation[0])
        dip_rad = np.deg2rad(self._orientation[1])
        yaw_rad = np.deg2rad(self._orientation[2])
        
        # Strike rotation matrix
        cos_strike = np.cos(strike_rad)
        sin_strike = np.sin(strike_rad)
        strike_matrix = np.array([
            [cos_strike, sin_strike, 0.0],
            [-sin_strike, cos_strike, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Dip rotation matrix
        cos_dip = np.cos(dip_rad)
        sin_dip = np.sin(dip_rad)
        dip_matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cos_dip, sin_dip],
            [0.0, -sin_dip, cos_dip]
        ])
        
        # Yaw rotation matrix
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)
        yaw_matrix = np.array([
            [cos_yaw, sin_yaw, 0.0],
            [-sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Combined rotation matrix
        return np.matmul(np.matmul(strike_matrix, dip_matrix), yaw_matrix)
    
    def distance_to_point(self, point: Tuple[float, float, float]) -> float:
        """
        Calculate distance from element center to a point.
        
        Parameters
        ----------
        point : Tuple[float, float, float]
            Point coordinates (x, y, z)
            
        Returns
        -------
        float
            Distance in meters
        """
        return np.sqrt(
            (point[0] - self._center[0])**2 +
            (point[1] - self._center[1])**2 +
            (point[2] - self._center[2])**2
        )
    
    def __repr__(self) -> str:
        return (
            f"DisplacementDiscontinuityElement(id={self._element_id}, "
            f"center={self._center}, dimensions={self._dimensions})"
        )
    
    def __str__(self) -> str:
        return (
            f"Element {self._element_id}:\n"
            f"  Center: {self._center} m\n"
            f"  Dimensions: {self._dimensions} m\n"
            f"  Orientation: {self._orientation}°\n"
            f"  Displacement: {self._displacement} m\n"
            f"  Stress: {self._stress} Pa"
        )

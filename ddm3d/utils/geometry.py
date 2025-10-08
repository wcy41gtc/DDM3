"""Geometry utility functions for DDM3D calculations."""

from typing import Tuple
import numpy as np


def create_rotation_matrix(
    strike: float,
    dip: float,
    yaw: float
) -> np.ndarray:
    """
    Create a 3D rotation matrix from strike, dip, and yaw angles.
    
    Parameters
    ----------
    strike : float
        Strike angle in degrees
    dip : float
        Dip angle in degrees
    yaw : float
        Yaw angle in degrees
        
    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    strike_rad = np.deg2rad(strike)
    dip_rad = np.deg2rad(dip)
    yaw_rad = np.deg2rad(yaw)
    
    # Strike rotation matrix (rotation around Z-axis)
    cos_strike = np.cos(strike_rad)
    sin_strike = np.sin(strike_rad)
    strike_matrix = np.array([
        [cos_strike, sin_strike, 0.0],
        [-sin_strike, cos_strike, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Dip rotation matrix (rotation around X-axis)
    cos_dip = np.cos(dip_rad)
    sin_dip = np.sin(dip_rad)
    dip_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_dip, sin_dip],
        [0.0, -sin_dip, cos_dip]
    ])
    
    # Yaw rotation matrix (rotation around Z-axis)
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    yaw_matrix = np.array([
        [cos_yaw, sin_yaw, 0.0],
        [-sin_yaw, cos_yaw, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Combined rotation matrix: R = R_strike * R_dip * R_yaw
    return np.matmul(np.matmul(strike_matrix, dip_matrix), yaw_matrix)


def transform_coordinates(
    points: np.ndarray,
    rotation_matrix: np.ndarray,
    translation: np.ndarray
) -> np.ndarray:
    """
    Transform coordinates using rotation and translation.
    
    Parameters
    ----------
    points : np.ndarray
        Array of points to transform (N x 3)
    rotation_matrix : np.ndarray
        3x3 rotation matrix
    translation : np.ndarray
        Translation vector (3,)
        
    Returns
    -------
    np.ndarray
        Transformed points (N x 3)
    """
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    if points.shape[1] != 3:
        raise ValueError("Points must have 3 coordinates")
    
    if rotation_matrix.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3")
    
    if translation.shape != (3,):
        raise ValueError("Translation vector must have 3 components")
    
    # Apply rotation and translation
    transformed_points = np.matmul(points, rotation_matrix.T) + translation
    
    return transformed_points


def calculate_distance(
    point1: Tuple[float, float, float],
    point2: Tuple[float, float, float]
) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Parameters
    ----------
    point1 : Tuple[float, float, float]
        First point (x, y, z)
    point2 : Tuple[float, float, float]
        Second point (x, y, z)
        
    Returns
    -------
    float
        Distance between points
    """
    p1 = np.array(point1)
    p2 = np.array(point2)
    return np.linalg.norm(p2 - p1)


def calculate_angle_between_vectors(
    vector1: Tuple[float, float, float],
    vector2: Tuple[float, float, float]
) -> float:
    """
    Calculate the angle between two vectors in degrees.
    
    Parameters
    ----------
    vector1 : Tuple[float, float, float]
        First vector (x, y, z)
    vector2 : Tuple[float, float, float]
        Second vector (x, y, z)
        
    Returns
    -------
    float
        Angle between vectors in degrees
    """
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    
    # Calculate dot product
    dot_product = np.dot(v1, v2)
    
    # Calculate magnitudes
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # Calculate angle in radians
    cos_angle = dot_product / (magnitude1 * magnitude2)
    
    # Clamp to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_angle)
    
    # Convert to degrees
    return np.rad2deg(angle_rad)


def calculate_plane_normal(
    point1: Tuple[float, float, float],
    point2: Tuple[float, float, float],
    point3: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """
    Calculate the normal vector of a plane defined by three points.
    
    Parameters
    ----------
    point1 : Tuple[float, float, float]
        First point on the plane
    point2 : Tuple[float, float, float]
        Second point on the plane
    point3 : Tuple[float, float, float]
        Third point on the plane
        
    Returns
    -------
    Tuple[float, float, float]
        Normal vector (x, y, z)
    """
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)
    
    # Calculate two vectors in the plane
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Calculate cross product to get normal
    normal = np.cross(v1, v2)
    
    # Normalize
    magnitude = np.linalg.norm(normal)
    if magnitude > 0:
        normal = normal / magnitude
    
    return tuple(normal)


def calculate_plane_angles(normal: Tuple[float, float, float]) -> Tuple[float, float]:
    """
    Calculate strike and dip angles from a plane normal vector.
    
    Parameters
    ----------
    normal : Tuple[float, float, float]
        Normal vector (x, y, z)
        
    Returns
    -------
    Tuple[float, float]
        Strike and dip angles in degrees
    """
    nx, ny, nz = normal
    
    # Calculate dip angle
    dip_rad = np.arccos(abs(nz))
    dip = np.rad2deg(dip_rad)
    
    # Calculate strike angle
    if abs(nz) < 1e-10:  # Horizontal plane
        strike = 0.0
    else:
        strike_rad = np.arctan2(nx, ny)
        strike = np.rad2deg(strike_rad)
        if strike < 0:
            strike += 360.0
    
    return (strike, dip)


def interpolate_linear(
    x: float,
    x1: float,
    x2: float,
    y1: float,
    y2: float
) -> float:
    """
    Linear interpolation between two points.
    
    Parameters
    ----------
    x : float
        X coordinate to interpolate at
    x1, x2 : float
        X coordinates of known points
    y1, y2 : float
        Y coordinates of known points
        
    Returns
    -------
    float
        Interpolated Y value
    """
    if abs(x2 - x1) < 1e-10:
        return y1
    
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def interpolate_bilinear(
    x: float,
    y: float,
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    z11: float,
    z12: float,
    z21: float,
    z22: float
) -> float:
    """
    Bilinear interpolation in a rectangular grid.
    
    Parameters
    ----------
    x, y : float
        Coordinates to interpolate at
    x1, x2 : float
        X coordinates of grid corners
    y1, y2 : float
        Y coordinates of grid corners
    z11, z12, z21, z22 : float
        Z values at grid corners (z11 at (x1,y1), z12 at (x1,y2), etc.)
        
    Returns
    -------
    float
        Interpolated Z value
    """
    # Linear interpolation in x direction
    z1 = interpolate_linear(x, x1, x2, z11, z21)
    z2 = interpolate_linear(x, x1, x2, z12, z22)
    
    # Linear interpolation in y direction
    return interpolate_linear(y, y1, y2, z1, z2)

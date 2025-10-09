"""Fracture class for DDM3D calculations."""

from typing import List, Tuple, Optional, Union
import numpy as np

from .element import DisplacementDiscontinuityElement
from .material import Material


class Fracture:
    """
    A fracture represented as a collection of displacement discontinuity elements.

    This class provides methods for creating different fracture geometries
    and managing the discretized elements that represent the fracture.
    """

    def __init__(
        self,
        fracture_id: int,
        elements: List[DisplacementDiscontinuityElement],
        material: Material,
    ):
        """
        Initialize a fracture.

        Parameters
        ----------
        fracture_id : int
            Unique identifier for the fracture
        elements : List[DisplacementDiscontinuityElement]
            List of elements representing the fracture
        material : Material
            Material properties for the fracture
        """
        self._fracture_id = int(fracture_id)
        self._elements = list(elements)
        self._material = material

        if not self._elements:
            raise ValueError("Fracture must contain at least one element")

    @classmethod
    def create_rectangular(
        cls,
        fracture_id: int,
        center: Tuple[float, float, float],
        length: float,
        height: float,
        element_size: Tuple[float, float],
        orientation: Tuple[float, float, float] = (0.0, 90.0, 0.0),
        material: Optional[Material] = None,
        initial_stress: Optional[Tuple[float, float, float]] = None,
    ) -> "Fracture":
        """
        Create a rectangular fracture.

        Parameters
        ----------
        fracture_id : int
            Unique identifier for the fracture
        center : Tuple[float, float, float]
            Center coordinates (x, y, z) in meters
        length : float
            Fracture length in meters
        height : float
            Fracture height in meters
        element_size : Tuple[float, float]
            Element size (length, height) in meters
        orientation : Tuple[float, float, float], optional
            Fracture orientation (o1, o2, o3) in degrees, by default (0, 90, 0)
        material : Material, optional
            Material properties, by default None (creates default material)
        initial_stress : Tuple[float, float, float], optional
            Initial stress (Ssl, Ssh, Snn) in Pa, by default (0, 0, 0)

        Returns
        -------
        Fracture
            Rectangular fracture instance
        """
        if material is None:
            material = Material(shear_modulus=10e9, poisson_ratio=0.25)

        if initial_stress is None:
            initial_stress = (0.0, 0.0, 0.0)

        # Calculate number of elements
        n_elements_length = max(1, int(np.ceil(length / element_size[0])))
        n_elements_height = max(1, int(np.ceil(height / element_size[1])))

        # Adjust element size to fit exactly
        actual_element_length = length / n_elements_length
        actual_element_height = height / n_elements_height

        # Create rotation matrix
        strike_rad = np.deg2rad(orientation[0])
        dip_rad = np.deg2rad(orientation[1])
        yaw_rad = np.deg2rad(orientation[2])

        cos_strike = np.cos(strike_rad)
        sin_strike = np.sin(strike_rad)
        cos_dip = np.cos(dip_rad)
        sin_dip = np.sin(dip_rad)
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)

        # Rotation matrices
        strike_matrix = np.array(
            [
                [cos_strike, sin_strike, 0.0],
                [-sin_strike, cos_strike, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        dip_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, cos_dip, sin_dip], [0.0, -sin_dip, cos_dip]]
        )

        yaw_matrix = np.array(
            [[cos_yaw, sin_yaw, 0.0], [-sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]]
        )

        rotation_matrix = np.matmul(np.matmul(strike_matrix, dip_matrix), yaw_matrix)

        # Create elements
        elements = []
        element_id = 1

        for i in range(n_elements_length):
            for j in range(n_elements_height):
                # Local coordinates (before rotation)
                local_x = (
                    -length / 2 + actual_element_length / 2 + i * actual_element_length
                )
                local_y = 0.0
                local_z = (
                    -height / 2 + actual_element_height / 2 + j * actual_element_height
                )

                local_coords = np.array([local_x, local_y, local_z])

                # Transform to global coordinates
                global_coords = rotation_matrix @ local_coords + np.array(center)

                element = DisplacementDiscontinuityElement(
                    element_id=element_id,
                    center=tuple(global_coords),
                    dimensions=(actual_element_length, actual_element_height),
                    orientation=orientation,
                    displacement=(0.0, 0.0, 0.0),
                    stress=initial_stress,
                )

                elements.append(element)
                element_id += 1

        return cls(fracture_id, elements, material)

    @classmethod
    def create_elliptical(
        cls,
        fracture_id: int,
        center: Tuple[float, float, float],
        length: float,
        height: float,
        element_size: Tuple[float, float],
        orientation: Tuple[float, float, float] = (0.0, 90.0, 0.0),
        material: Optional[Material] = None,
        initial_stress: Optional[Tuple[float, float, float]] = None,
    ) -> "Fracture":
        """
        Create an elliptical fracture.

        Parameters
        ----------
        fracture_id : int
            Unique identifier for the fracture
        center : Tuple[float, float, float]
            Center coordinates (x, y, z) in meters
        length : float
            Fracture length (major axis) in meters
        height : float
            Fracture height (minor axis) in meters
        element_size : Tuple[float, float]
            Element size (length, height) in meters
        orientation : Tuple[float, float, float], optional
            Fracture orientation (o1, o2, o3) in degrees, by default (0, 90, 0)
        material : Material, optional
            Material properties, by default None (creates default material)
        initial_stress : Tuple[float, float, float], optional
            Initial stress (Ssl, Ssh, Snn) in Pa, by default (0, 0, 0)

        Returns
        -------
        Fracture
            Elliptical fracture instance
        """
        if material is None:
            material = Material(shear_modulus=10e9, poisson_ratio=0.25)

        if initial_stress is None:
            initial_stress = (0.0, 0.0, 0.0)

        # Calculate number of elements
        n_elements_length = max(1, int(np.ceil(length / element_size[0])))
        n_elements_height = max(1, int(np.ceil(height / element_size[1])))

        # Adjust element size to fit exactly
        actual_element_length = length / n_elements_length
        actual_element_height = height / n_elements_height

        # Create rotation matrix
        strike_rad = np.deg2rad(orientation[0])
        dip_rad = np.deg2rad(orientation[1])
        yaw_rad = np.deg2rad(orientation[2])

        cos_strike = np.cos(strike_rad)
        sin_strike = np.sin(strike_rad)
        cos_dip = np.cos(dip_rad)
        sin_dip = np.sin(dip_rad)
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)

        # Rotation matrices
        strike_matrix = np.array(
            [
                [cos_strike, sin_strike, 0.0],
                [-sin_strike, cos_strike, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        dip_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, cos_dip, sin_dip], [0.0, -sin_dip, cos_dip]]
        )

        yaw_matrix = np.array(
            [[cos_yaw, sin_yaw, 0.0], [-sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]]
        )

        rotation_matrix = np.matmul(np.matmul(strike_matrix, dip_matrix), yaw_matrix)

        # Create elements
        elements = []
        element_id = 1

        for i in range(n_elements_length):
            for j in range(n_elements_height):
                # Local coordinates (before rotation)
                local_x = (
                    -length / 2 + actual_element_length / 2 + i * actual_element_length
                )
                local_y = 0.0
                local_z = (
                    -height / 2 + actual_element_height / 2 + j * actual_element_height
                )

                # Check if point is inside ellipse
                ellipse_test = (local_x / (length / 2)) ** 2 + (
                    local_z / (height / 2)
                ) ** 2

                if ellipse_test <= 1.0:
                    local_coords = np.array([local_x, local_y, local_z])

                    # Transform to global coordinates
                    global_coords = rotation_matrix @ local_coords + np.array(center)

                    element = DisplacementDiscontinuityElement(
                        element_id=element_id,
                        center=tuple(global_coords),
                        dimensions=(actual_element_length, actual_element_height),
                        orientation=orientation,
                        displacement=(0.0, 0.0, 0.0),
                        stress=initial_stress,
                    )

                    elements.append(element)
                    element_id += 1

        return cls(fracture_id, elements, material)

    @property
    def fracture_id(self) -> int:
        """Fracture identifier."""
        return self._fracture_id

    @property
    def elements(self) -> List[DisplacementDiscontinuityElement]:
        """List of elements in the fracture."""
        return self._elements.copy()

    @property
    def n_elements(self) -> int:
        """Number of elements in the fracture."""
        return len(self._elements)

    @property
    def material(self) -> Material:
        """Material properties of the fracture."""
        return self._material

    def get_element(self, element_id: int) -> DisplacementDiscontinuityElement:
        """
        Get a specific element by ID.

        Parameters
        ----------
        element_id : int
            Element identifier

        Returns
        -------
        DisplacementDiscontinuityElement
            The requested element

        Raises
        ------
        ValueError
            If element_id is not found
        """
        for element in self._elements:
            if element.element_id == element_id:
                return element
        raise ValueError(f"Element {element_id} not found")

    def get_element_centers(self) -> List[Tuple[float, float, float]]:
        """
        Get centers of all elements.

        Returns
        -------
        List[Tuple[float, float, float]]
            List of element centers
        """
        return [element.center for element in self._elements]

    def get_total_area(self) -> float:
        """
        Calculate total fracture area.

        Returns
        -------
        float
            Total fracture area in square meters
        """
        return sum(element.area for element in self._elements)

    def get_bounding_box(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Get the bounding box of the fracture.

        Returns
        -------
        Tuple[Tuple[float, float, float], Tuple[float, float, float]]
            ((min_x, min_y, min_z), (max_x, max_y, max_z))
        """
        centers = self.get_element_centers()
        centers_array = np.array(centers)

        min_coords = tuple(np.min(centers_array, axis=0))
        max_coords = tuple(np.max(centers_array, axis=0))

        return (min_coords, max_coords)

    def set_initial_stress(self, stress: Tuple[float, float, float]) -> None:
        """
        Set initial stress for all elements.

        Parameters
        ----------
        stress : Tuple[float, float, float]
            Initial stress (Ssl, Ssh, Snn) in Pa
        """
        for element in self._elements:
            element.set_stress(stress)

    def clear_displacements(self) -> None:
        """Clear displacement discontinuities for all elements."""
        for element in self._elements:
            element.set_displacement((0.0, 0.0, 0.0))

    def __repr__(self) -> str:
        return f"Fracture(id={self._fracture_id}, n_elements={self.n_elements})"

    def __str__(self) -> str:
        min_coords, max_coords = self.get_bounding_box()
        return (
            f"Fracture {self._fracture_id}:\n"
            f"  Elements: {self.n_elements}\n"
            f"  Total Area: {self.get_total_area():.2f} m²\n"
            f"  Bounding Box: {min_coords} to {max_coords}\n"
            f"  Material: {self._material.name}"
        )

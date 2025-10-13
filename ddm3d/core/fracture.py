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
        orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
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
            Fracture orientation (o1, o2, o3) in degrees, by default (0, 0, 0)
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

        # Create rotation matrix
        o1_rad = np.deg2rad(orientation[0])
        o2_rad = np.deg2rad(orientation[1])
        o3_rad = np.deg2rad(orientation[2])

        cos_o1 = np.cos(o1_rad)
        sin_o1 = np.sin(o1_rad)
        cos_o2 = np.cos(o2_rad)
        sin_o2 = np.sin(o2_rad)
        cos_o3 = np.cos(o3_rad)
        sin_o3 = np.sin(o3_rad)

        dl = element_size[0]
        dh = element_size[1]

        c_x = center[0]
        c_y = center[1]
        c_z = center[2]

        # Rotation matrices
        o1_matrix = np.array(
            [
                [cos_o1, sin_o1, 0.0],
                [-sin_o1, cos_o1, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        o2_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, cos_o2, sin_o2], [0.0, -sin_o2, cos_o2]]
        )

        o3_matrix = np.array(
            [[cos_o3, sin_o3, 0.0], [-sin_o3, cos_o3, 0.0], [0.0, 0.0, 1.0]]
        )

        rotation_matrix = np.matmul(np.matmul(o1_matrix, o2_matrix), o3_matrix)

        # Create elements
        elements = []
        element_id = 1

        for i in range(1, round(length/dl)):
            for j in range(1, round(height/dh)):
                # Local coordinates (before rotation)
                local_x = c_x-length/2+dl/2*(2*i-1)
                local_y = c_y
                local_z = c_z-height/2+dh/2*(2*j-1)

                local_coords = np.array([local_x - c_x, local_y - c_y, local_z - c_z])

                # Transform to global coordinates
                global_coords = np.matmul(rotation_matrix, local_coords)

                element = DisplacementDiscontinuityElement(
                    element_id=element_id,
                    center=tuple([global_coords[0]+c_x, global_coords[1]+c_y, global_coords[2]+c_z]),
                    dimensions=(dl, dh),
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
        orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
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
            Fracture orientation (o1, o2, o3) in degrees, by default (0, 0, 0)
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

        dl = element_size[0]
        dh = element_size[1]

        c_x = center[0]
        c_y = center[1]
        c_z = center[2]

        # Create rotation matrix
        o1_rad = np.deg2rad(orientation[0])
        o2_rad = np.deg2rad(orientation[1])
        o3_rad = np.deg2rad(orientation[2])

        cos_o1 = np.cos(o1_rad)
        sin_o1 = np.sin(o1_rad)
        cos_o2 = np.cos(o2_rad)
        sin_o2 = np.sin(o2_rad)
        cos_o3 = np.cos(o3_rad)
        sin_o3 = np.sin(o3_rad)

        # Rotation matrices
        o1_matrix = np.array(
            [
                [cos_o1, sin_o1, 0.0],
                [-sin_o1, cos_o1, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        o2_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, cos_o2, sin_o2], [0.0, -sin_o2, cos_o2]]
        )

        o3_matrix = np.array(
            [[cos_o3, sin_o3, 0.0], [-sin_o3, cos_o3, 0.0], [0.0, 0.0, 1.0]]
        )

        rotation_matrix = np.matmul(np.matmul(o1_matrix, o2_matrix), o3_matrix)

        # Create elements
        elements = []
        element_id = 1

        for i in range(1, round(length/dl)):
            for j in range(1, round(height/dh)):
                # Local coordinates (before rotation)
                local_x = c_x-length/2+dl/2*(2*i-1)
                local_y = c_y
                local_z = c_z-height/2+dh/2*(2*j-1)

                # Check if point is inside ellipse
                ellipse_test = (local_x-c_x)**2/(length/2)**2+(local_z-c_z)**2/(height/2)**2

                if ellipse_test <= 1.0:
                    local_coords = np.array([local_x - c_x, local_y - c_y, local_z - c_z])

                    # Transform to global coordinates
                    global_coords = np.matmul(rotation_matrix, local_coords)

                    element = DisplacementDiscontinuityElement(
                        element_id=element_id,
                        center=tuple([global_coords[0]+c_x, global_coords[1]+c_y, global_coords[2]+c_z]),
                        dimensions=(dl, dh),
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

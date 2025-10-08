"""Plane class for DDM3D calculations."""

from typing import List, Tuple, Dict, Any
import numpy as np


class Plane:
    """
    A plane for monitoring stress and displacement fields.
    
    This class represents a 2D plane in 3D space with a regular grid of
    monitoring points for calculating and storing stress/displacement data.
    """
    
    def __init__(
        self,
        plane_id: int,
        center: Tuple[float, float, float],
        size: Tuple[float, float],
        node_size: Tuple[float, float],
        orientation: str
    ):
        """
        Initialize a monitoring plane.
        
        Parameters
        ----------
        plane_id : int
            Unique identifier for the plane
        center : Tuple[float, float, float]
            Center coordinates (x, y, z) in meters
        size : Tuple[float, float]
            Plane size (length, width) in meters
        node_size : Tuple[float, float]
            Node spacing (dx, dy) in meters
        orientation : str
            Plane orientation ('XY', 'XZ', or 'YZ')
        """
        self._plane_id = int(plane_id)
        
        if len(center) != 3:
            raise ValueError("Center must be a 3-tuple (x, y, z)")
        self._center = tuple(float(c) for c in center)
        
        if len(size) != 2:
            raise ValueError("Size must be a 2-tuple (length, width)")
        if any(s <= 0 for s in size):
            raise ValueError("Size must be positive")
        self._size = tuple(float(s) for s in size)
        
        if len(node_size) != 2:
            raise ValueError("Node size must be a 2-tuple (dx, dy)")
        if any(ns <= 0 for ns in node_size):
            raise ValueError("Node size must be positive")
        self._node_size = tuple(float(ns) for ns in node_size)
        
        if orientation not in ['XY', 'XZ', 'YZ']:
            raise ValueError("Orientation must be 'XY', 'XZ', or 'YZ'")
        self._orientation = orientation
        
        # Create nodes
        self._nodes = self._create_nodes()
    
    def _create_nodes(self) -> List[Dict[str, Any]]:
        """Create the grid of monitoring nodes."""
        n_nodes_length = int(self._size[0] / self._node_size[0])
        n_nodes_width = int(self._size[1] / self._node_size[1])
        
        nodes = []
        node_id = 1
        
        for i in range(n_nodes_length):
            for j in range(n_nodes_width):
                if self._orientation == 'XY':
                    x = self._center[0] - self._size[0]/2 + self._node_size[0]/2 * (2*i + 1)
                    y = self._center[1] - self._size[1]/2 + self._node_size[1]/2 * (2*j + 1)
                    z = self._center[2]
                elif self._orientation == 'XZ':
                    x = self._center[0] - self._size[0]/2 + self._node_size[0]/2 * (2*i + 1)
                    y = self._center[1]
                    z = self._center[2] - self._size[1]/2 + self._node_size[1]/2 * (2*j + 1)
                else:  # YZ
                    x = self._center[0]
                    y = self._center[1] - self._size[0]/2 + self._node_size[0]/2 * (2*i + 1)
                    z = self._center[2] - self._size[1]/2 + self._node_size[1]/2 * (2*j + 1)
                
                node = {
                    'id': node_id,
                    'row': i,
                    'col': j,
                    'position': (x, y, z),
                    'stress': {
                        'SXX': [],
                        'SYY': [],
                        'SZZ': [],
                        'SXY': [],
                        'SXZ': [],
                        'SYZ': []
                    },
                    'strain': {
                        'EXX': [],
                        'EYY': [],
                        'EZZ': []
                    },
                    'displacement': {
                        'UXX': [],
                        'UYY': [],
                        'UZZ': []
                    }
                }
                
                nodes.append(node)
                node_id += 1
        
        return nodes
    
    @classmethod
    def create_xy_plane(
        cls,
        plane_id: int,
        center: Tuple[float, float, float],
        size: Tuple[float, float],
        node_size: Tuple[float, float]
    ) -> "Plane":
        """
        Create a plane parallel to the XY plane.
        
        Parameters
        ----------
        plane_id : int
            Unique identifier for the plane
        center : Tuple[float, float, float]
            Center coordinates (x, y, z) in meters
        size : Tuple[float, float]
            Plane size (length, width) in meters
        node_size : Tuple[float, float]
            Node spacing (dx, dy) in meters
            
        Returns
        -------
        Plane
            XY plane instance
        """
        return cls(plane_id, center, size, node_size, 'XY')
    
    @classmethod
    def create_xz_plane(
        cls,
        plane_id: int,
        center: Tuple[float, float, float],
        size: Tuple[float, float],
        node_size: Tuple[float, float]
    ) -> "Plane":
        """
        Create a plane parallel to the XZ plane.
        
        Parameters
        ----------
        plane_id : int
            Unique identifier for the plane
        center : Tuple[float, float, float]
            Center coordinates (x, y, z) in meters
        size : Tuple[float, float]
            Plane size (length, width) in meters
        node_size : Tuple[float, float]
            Node spacing (dx, dy) in meters
            
        Returns
        -------
        Plane
            XZ plane instance
        """
        return cls(plane_id, center, size, node_size, 'XZ')
    
    @classmethod
    def create_yz_plane(
        cls,
        plane_id: int,
        center: Tuple[float, float, float],
        size: Tuple[float, float],
        node_size: Tuple[float, float]
    ) -> "Plane":
        """
        Create a plane parallel to the YZ plane.
        
        Parameters
        ----------
        plane_id : int
            Unique identifier for the plane
        center : Tuple[float, float, float]
            Center coordinates (x, y, z) in meters
        size : Tuple[float, float]
            Plane size (length, width) in meters
        node_size : Tuple[float, float]
            Node spacing (dx, dy) in meters
            
        Returns
        -------
        Plane
            YZ plane instance
        """
        return cls(plane_id, center, size, node_size, 'YZ')
    
    @property
    def plane_id(self) -> int:
        """Plane identifier."""
        return self._plane_id
    
    @property
    def center(self) -> Tuple[float, float, float]:
        """Center coordinates (x, y, z) in meters."""
        return self._center
    
    @property
    def size(self) -> Tuple[float, float]:
        """Plane size (length, width) in meters."""
        return self._size
    
    @property
    def node_size(self) -> Tuple[float, float]:
        """Node spacing (dx, dy) in meters."""
        return self._node_size
    
    @property
    def orientation(self) -> str:
        """Plane orientation ('XY', 'XZ', or 'YZ')."""
        return self._orientation
    
    @property
    def nodes(self) -> List[Dict[str, Any]]:
        """List of monitoring nodes."""
        return self._nodes.copy()
    
    @property
    def n_nodes(self) -> int:
        """Number of monitoring nodes."""
        return len(self._nodes)
    
    def get_node(self, node_id: int) -> Dict[str, Any]:
        """
        Get a specific node by ID.
        
        Parameters
        ----------
        node_id : int
            Node identifier
            
        Returns
        -------
        Dict[str, Any]
            The requested node
            
        Raises
        ------
        ValueError
            If node_id is not found
        """
        for node in self._nodes:
            if node['id'] == node_id:
                return node
        raise ValueError(f"Node {node_id} not found")
    
    def get_node_positions(self) -> List[Tuple[float, float, float]]:
        """
        Get positions of all nodes.
        
        Returns
        -------
        List[Tuple[float, float, float]]
            List of node positions
        """
        return [node['position'] for node in self._nodes]
    
    def add_stress_data(
        self,
        node_id: int,
        sxx: float,
        syy: float,
        szz: float,
        sxy: float,
        sxz: float,
        syz: float
    ) -> None:
        """
        Add stress data for a specific node.
        
        Parameters
        ----------
        node_id : int
            Node identifier
        sxx, syy, szz : float
            Normal stress components in Pa
        sxy, sxz, syz : float
            Shear stress components in Pa
        """
        node = self.get_node(node_id)
        node['stress']['SXX'].append(float(sxx))
        node['stress']['SYY'].append(float(syy))
        node['stress']['SZZ'].append(float(szz))
        node['stress']['SXY'].append(float(sxy))
        node['stress']['SXZ'].append(float(sxz))
        node['stress']['SYZ'].append(float(syz))
    
    def add_strain_data(self, node_id: int, exx: float, eyy: float, ezz: float) -> None:
        """
        Add strain data for a specific node.
        
        Parameters
        ----------
        node_id : int
            Node identifier
        exx, eyy, ezz : float
            Strain components (dimensionless)
        """
        node = self.get_node(node_id)
        node['strain']['EXX'].append(float(exx))
        node['strain']['EYY'].append(float(eyy))
        node['strain']['EZZ'].append(float(ezz))
    
    def add_displacement_data(self, node_id: int, uxx: float, uyy: float, uzz: float) -> None:
        """
        Add displacement data for a specific node.
        
        Parameters
        ----------
        node_id : int
            Node identifier
        uxx, uyy, uzz : float
            Displacement components in meters
        """
        node = self.get_node(node_id)
        node['displacement']['UXX'].append(float(uxx))
        node['displacement']['UYY'].append(float(uyy))
        node['displacement']['UZZ'].append(float(uzz))
    
    def clear_all_data(self) -> None:
        """Clear data from all nodes."""
        for node in self._nodes:
            for component in node['stress']:
                node['stress'][component].clear()
            for component in node['strain']:
                node['strain'][component].clear()
            for component in node['displacement']:
                node['displacement'][component].clear()
    
    def get_grid_shape(self) -> Tuple[int, int]:
        """
        Get the shape of the monitoring grid.
        
        Returns
        -------
        Tuple[int, int]
            Grid shape (n_rows, n_cols)
        """
        n_rows = int(self._size[0] / self._node_size[0])
        n_cols = int(self._size[1] / self._node_size[1])
        return (n_rows, n_cols)
    
    def get_data_count(self) -> int:
        """Get the number of data points stored."""
        if not self._nodes:
            return 0
        first_node = self._nodes[0]
        if not first_node['stress']['SXX']:
            return 0
        return len(first_node['stress']['SXX'])
    
    def __repr__(self) -> str:
        return f"Plane(id={self._plane_id}, orientation={self._orientation}, n_nodes={self.n_nodes})"
    
    def __str__(self) -> str:
        data_count = self.get_data_count()
        return (
            f"Plane {self._plane_id} ({self._orientation}):\n"
            f"  Center: {self._center} m\n"
            f"  Size: {self._size} m\n"
            f"  Node Size: {self._node_size} m\n"
            f"  Nodes: {self.n_nodes}\n"
            f"  Data Points: {data_count}"
        )

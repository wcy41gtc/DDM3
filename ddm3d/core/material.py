"""Material properties for DDM3D calculations."""

from typing import Dict, Any
import numpy as np


class Material:
    """
    Material properties for displacement discontinuity calculations.
    
    This class encapsulates the elastic properties of the rock material
    used in DDM calculations.
    """
    
    def __init__(
        self,
        shear_modulus: float,
        poisson_ratio: float,
        density: float = 2650.0,
        name: str = "Rock"
    ):
        """
        Initialize material properties.
        
        Parameters
        ----------
        shear_modulus : float
            Shear modulus in Pa
        poisson_ratio : float
            Poisson's ratio (dimensionless, 0 < nu < 0.5)
        density : float, optional
            Material density in kg/m³, by default 2650.0
        name : str, optional
            Material name, by default "Rock"
            
        Raises
        ------
        ValueError
            If shear_modulus <= 0 or poisson_ratio not in (0, 0.5)
        """
        if shear_modulus <= 0:
            raise ValueError("Shear modulus must be positive")
        if not 0 < poisson_ratio < 0.5:
            raise ValueError("Poisson's ratio must be in (0, 0.5)")
        if density <= 0:
            raise ValueError("Density must be positive")
            
        self._shear_modulus = float(shear_modulus)
        self._poisson_ratio = float(poisson_ratio)
        self._density = float(density)
        self._name = str(name)
        
        # Calculate derived properties
        self._youngs_modulus = 2.0 * self._shear_modulus * (1.0 + self._poisson_ratio)
        self._bulk_modulus = self._youngs_modulus / (3.0 * (1.0 - 2.0 * self._poisson_ratio))
        self._lame_lambda = self._youngs_modulus * self._poisson_ratio / (
            (1.0 + self._poisson_ratio) * (1.0 - 2.0 * self._poisson_ratio)
        )
    
    @property
    def shear_modulus(self) -> float:
        """Shear modulus in Pa."""
        return self._shear_modulus
    
    @property
    def poisson_ratio(self) -> float:
        """Poisson's ratio."""
        return self._poisson_ratio
    
    @property
    def density(self) -> float:
        """Material density in kg/m³."""
        return self._density
    
    @property
    def name(self) -> str:
        """Material name."""
        return self._name
    
    @property
    def youngs_modulus(self) -> float:
        """Young's modulus in Pa."""
        return self._youngs_modulus
    
    @property
    def bulk_modulus(self) -> float:
        """Bulk modulus in Pa."""
        return self._bulk_modulus
    
    @property
    def lame_lambda(self) -> float:
        """Lame's first parameter in Pa."""
        return self._lame_lambda
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert material properties to dictionary."""
        return {
            "shear_modulus": self._shear_modulus,
            "poisson_ratio": self._poisson_ratio,
            "density": self._density,
            "name": self._name,
            "youngs_modulus": self._youngs_modulus,
            "bulk_modulus": self._bulk_modulus,
            "lame_lambda": self._lame_lambda,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Material":
        """Create Material from dictionary."""
        return cls(
            shear_modulus=data["shear_modulus"],
            poisson_ratio=data["poisson_ratio"],
            density=data.get("density", 2650.0),
            name=data.get("name", "Rock")
        )
    
    def __repr__(self) -> str:
        return (
            f"Material(name='{self._name}', "
            f"G={self._shear_modulus:.2e} Pa, "
            f"nu={self._poisson_ratio:.3f})"
        )
    
    def __str__(self) -> str:
        return (
            f"Material: {self._name}\n"
            f"  Shear Modulus: {self._shear_modulus:.2e} Pa\n"
            f"  Poisson's Ratio: {self._poisson_ratio:.3f}\n"
            f"  Young's Modulus: {self._youngs_modulus:.2e} Pa\n"
            f"  Density: {self._density:.1f} kg/m³"
        )
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Material):
            return False
        return (
            abs(self._shear_modulus - other._shear_modulus) < 1e-10 and
            abs(self._poisson_ratio - other._poisson_ratio) < 1e-10 and
            abs(self._density - other._density) < 1e-10 and
            self._name == other._name
        )

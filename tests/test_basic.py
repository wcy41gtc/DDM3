"""Basic tests for DDM3D package."""

import pytest
import numpy as np
from ddm3d import Material, Fracture, Fiber, DDMCalculator


class TestMaterial:
    """Test Material class."""
    
    def test_material_creation(self):
        """Test material creation with valid parameters."""
        material = Material(
            shear_modulus=10e9,
            poisson_ratio=0.25,
            density=2650.0,
            name="Test Rock"
        )
        
        assert material.shear_modulus == 10e9
        assert material.poisson_ratio == 0.25
        assert material.density == 2650.0
        assert material.name == "Test Rock"
        assert material.youngs_modulus > 0
        assert material.bulk_modulus > 0
    
    def test_material_validation(self):
        """Test material parameter validation."""
        # Test negative shear modulus
        with pytest.raises(ValueError):
            Material(shear_modulus=-1e9, poisson_ratio=0.25)
        
        # Test invalid Poisson ratio
        with pytest.raises(ValueError):
            Material(shear_modulus=10e9, poisson_ratio=0.6)
        
        # Test negative density
        with pytest.raises(ValueError):
            Material(shear_modulus=10e9, poisson_ratio=0.25, density=-1000)


class TestFracture:
    """Test Fracture class."""
    
    def test_rectangular_fracture_creation(self):
        """Test rectangular fracture creation."""
        material = Material(shear_modulus=10e9, poisson_ratio=0.25)
        
        fracture = Fracture.create_rectangular(
            fracture_id=1,
            center=(0, 0, 0),
            length=100,
            height=50,
            element_size=(10, 10),
            material=material
        )
        
        assert fracture.fracture_id == 1
        assert fracture.n_elements > 0
        assert fracture.material == material
        assert fracture.get_total_area() > 0
    
    def test_elliptical_fracture_creation(self):
        """Test elliptical fracture creation."""
        material = Material(shear_modulus=10e9, poisson_ratio=0.25)
        
        fracture = Fracture.create_elliptical(
            fracture_id=1,
            center=(0, 0, 0),
            length=100,
            height=50,
            element_size=(10, 10),
            material=material
        )
        
        assert fracture.fracture_id == 1
        assert fracture.n_elements > 0
        assert fracture.material == material
        assert fracture.get_total_area() > 0


class TestFiber:
    """Test Fiber class."""
    
    def test_linear_fiber_creation(self):
        """Test linear fiber creation."""
        fiber = Fiber.create_linear(
            fiber_id=1,
            start=(0, 0, 0),
            end=(100, 0, 0),
            n_channels=10
        )
        
        assert fiber.fiber_id == 1
        assert fiber.n_channels == 10
        assert len(fiber.channels) == 10
        
        # Check first and last channel positions
        first_channel = fiber.get_channel(1)
        last_channel = fiber.get_channel(10)
        
        assert first_channel.position[0] > 0
        assert last_channel.position[0] < 100
    
    def test_fiber_validation(self):
        """Test fiber parameter validation."""
        # Test zero channels
        with pytest.raises(ValueError):
            Fiber.create_linear(
                fiber_id=1,
                start=(0, 0, 0),
                end=(100, 0, 0),
                n_channels=0
            )


class TestDDMCalculator:
    """Test DDMCalculator class."""
    
    def test_calculator_creation(self):
        """Test calculator creation."""
        calculator = DDMCalculator()
        assert calculator._tolerance == 1e-10
    
    def test_simple_calculation(self):
        """Test a simple DDM calculation."""
        # Create material and fracture
        material = Material(shear_modulus=10e9, poisson_ratio=0.25)
        
        fracture = Fracture.create_rectangular(
            fracture_id=1,
            center=(0, 0, 0),
            length=20,
            height=10,
            element_size=(5, 5),
            material=material,
            initial_stress=(0, 0, 1e6)
        )
        
        # Create calculator and solve
        calculator = DDMCalculator()
        calculator.solve_displacement_discontinuities([fracture])
        
        # Check that displacements were calculated
        element = fracture.get_element(1)
        assert element.dsl is not None
        assert element.dsh is not None
        assert element.dnn is not None
    
    def test_fiber_response_calculation(self):
        """Test fiber response calculation."""
        # Create material and fracture
        material = Material(shear_modulus=10e9, poisson_ratio=0.25)
        
        fracture = Fracture.create_rectangular(
            fracture_id=1,
            center=(0, 0, 0),
            length=20,
            height=10,
            element_size=(5, 5),
            material=material,
            initial_stress=(0, 0, 1e6)
        )
        
        # Create fiber
        fiber = Fiber.create_linear(
            fiber_id=1,
            start=(10, 5, -50),
            end=(10, 5, 50),
            n_channels=10
        )
        
        # Calculate response
        calculator = DDMCalculator()
        calculator.solve_displacement_discontinuities([fracture])
        calculator.calculate_fiber_response([fracture], [fiber])
        
        # Check that data was stored
        channel = fiber.get_channel(1)
        assert channel.get_data_count() > 0
        assert len(channel.get_stress_data('SXX')) > 0
        assert len(channel.get_strain_data('EXX')) > 0
        assert len(channel.get_displacement_data('UXX')) > 0


if __name__ == "__main__":
    pytest.main([__file__])

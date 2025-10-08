"""DDM calculator for solving displacement discontinuity problems."""

from typing import List, Tuple, Optional
import numpy as np

# Use numpy's lstsq directly
def lstsq(a, b, rcond=None):
    """Least squares solution using numpy."""
    return np.linalg.lstsq(a, b, rcond=rcond)

from ..core.fracture import Fracture
from ..core.fiber import Fiber, Channel
from ..core.plane import Plane
from ..core.material import Material


class DDMCalculator:
    """
    Displacement Discontinuity Method calculator.
    
    This class provides methods for solving displacement discontinuity problems
    and calculating stress/displacement fields at monitoring points.
    """
    
    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize the DDM calculator.
        
        Parameters
        ----------
        tolerance : float, optional
            Numerical tolerance for calculations, by default 1e-10
        """
        self._tolerance = float(tolerance)
    
    def solve_displacement_discontinuities(
        self,
        fractures: List[Fracture],
        use_least_squares: bool = True
    ) -> None:
        """
        Solve for displacement discontinuities in all fractures.
        
        Parameters
        ----------
        fractures : List[Fracture]
            List of fractures to solve
        use_least_squares : bool, optional
            Use least squares solver instead of direct solve, by default True
        """
        if not fractures:
            raise ValueError("At least one fracture must be provided")
        
        for fracture in fractures:
            self._solve_single_fracture(fracture, use_least_squares)
    
    def _solve_single_fracture(
        self,
        fracture: Fracture,
        use_least_squares: bool
    ) -> None:
        """Solve displacement discontinuities for a single fracture."""
        n_elements = fracture.n_elements
        elements = fracture.elements
        material = fracture.material
        
        # Initialize coefficient matrices
        coef_slsl = np.zeros((n_elements, n_elements))
        coef_slsh = np.zeros((n_elements, n_elements))
        coef_slnn = np.zeros((n_elements, n_elements))
        coef_shsl = np.zeros((n_elements, n_elements))
        coef_shsh = np.zeros((n_elements, n_elements))
        coef_shnn = np.zeros((n_elements, n_elements))
        coef_nnsl = np.zeros((n_elements, n_elements))
        coef_nnsh = np.zeros((n_elements, n_elements))
        coef_nnnn = np.zeros((n_elements, n_elements))
        
        # Initialize stress vectors
        Ssl = np.zeros((n_elements, 1))
        Ssh = np.zeros((n_elements, 1))
        Snn = np.zeros((n_elements, 1))
        
        # Fill stress vectors
        for i, element in enumerate(elements):
            Ssl[i] = element.Ssl
            Ssh[i] = element.Ssh
            Snn[i] = element.Snn
        
        # Calculate influence coefficients
        for i, receiver in enumerate(elements):
            for j, source in enumerate(elements):
                # Calculate influence coefficients
                coefs = self._calculate_influence_coefficients(
                    receiver, source, material
                )
                
                # Store coefficients
                coef_slsl[i, j] = coefs[0]
                coef_slsh[i, j] = coefs[1]
                coef_slnn[i, j] = coefs[2]
                coef_shsl[i, j] = coefs[3]
                coef_shsh[i, j] = coefs[4]
                coef_shnn[i, j] = coefs[5]
                coef_nnsl[i, j] = coefs[6]
                coef_nnsh[i, j] = coefs[7]
                coef_nnnn[i, j] = coefs[8]
        
        # Assemble system matrix and right-hand side
        S = np.vstack((Ssl, Ssh, Snn))
        coef = np.vstack((
            np.hstack((coef_slsl, coef_slsh, coef_slnn)),
            np.hstack((coef_shsl, coef_shsh, coef_shnn)),
            np.hstack((coef_nnsl, coef_nnsh, coef_nnnn))
        ))
        
        # Handle numerical issues
        coef[np.isnan(coef)] = 0.0
        coef[np.abs(coef) < self._tolerance] = 0.0
        
        # Solve system
        if use_least_squares:
            DD_solution, _, _, _ = lstsq(coef, S)
        else:
            DD_solution = np.linalg.solve(coef, S)
        
        # Apply solution to elements
        for i, element in enumerate(elements):
            dsl = DD_solution[i][0]
            dsh = DD_solution[i + n_elements][0]
            dnn = DD_solution[i + 2 * n_elements][0]
            element.set_displacement((dsl, dsh, dnn))
    
    def _calculate_influence_coefficients(
        self,
        receiver: "DisplacementDiscontinuityElement",
        source: "DisplacementDiscontinuityElement",
        material: Material
    ) -> Tuple[float, ...]:
        """
        Calculate influence coefficients between two elements.
        
        Returns
        -------
        Tuple[float, ...]
            Nine influence coefficients (slsl, slsh, slnn, shsl, shsh, shnn, nnsl, nnsh, nnnn)
        """
        # Calculate relative position
        gamma = receiver.strike - source.strike
        cos_gamma = np.cos(np.deg2rad(gamma))
        cos_2gamma = np.cos(np.deg2rad(2 * gamma))
        sin_gamma = np.sin(np.deg2rad(gamma))
        sin_2gamma = np.sin(np.deg2rad(2 * gamma))
        
        # Transform coordinates to local system
        cos_beta = np.cos(np.deg2rad(receiver.strike))
        sin_beta = np.sin(np.deg2rad(receiver.strike))
        
        x1 = (receiver.x - source.x) * cos_beta + (receiver.y - source.y) * sin_beta
        x2 = receiver.z - source.z
        x3 = -(receiver.x - source.x) * sin_beta + (receiver.y - source.y) * cos_beta
        
        # Element dimensions
        a = source.length / 2.0
        b = source.height / 2.0
        
        # Calculate coefficient
        Cr = material.shear_modulus / (4.0 * np.pi * (1.0 - material.poisson_ratio))
        
        # Calculate distance terms
        r1 = np.sqrt((x1 - a)**2 + (x2 - b)**2 + x3**2)
        r2 = np.sqrt((x1 - a)**2 + (x2 + b)**2 + x3**2)
        r3 = np.sqrt((x1 + a)**2 + (x2 - b)**2 + x3**2)
        r4 = np.sqrt((x1 + a)**2 + (x2 + b)**2 + x3**2)
        
        # Calculate integral terms
        J1, J2, J3, J4, J5, J6, J7, J8, J9 = self._calculate_integrals(
            x1, x2, x3, a, b, r1, r2, r3, r4
        )
        
        J10, J11, J12, J13, J14, J15, J16, J17, J18, J19 = self._calculate_derivatives(
            x1, x2, x3, a, b, r1, r2, r3, r4
        )
        
        # Calculate influence coefficients
        coef_slsl = Cr * (-sin_gamma * cos_gamma * (2 * J8 - x3 * J10) +
                         cos_2gamma * (J6 + material.poisson_ratio * J5 - x3 * J12) +
                         sin_gamma * cos_gamma * (-x3 * J16))
        
        coef_slsh = Cr * (-sin_gamma * cos_gamma * (2 * material.poisson_ratio * J9 - x3 * J11) +
                         cos_2gamma * (-material.poisson_ratio * J7 - x3 * J19) +
                         sin_gamma * cos_gamma * (-x3 * J17))
        
        coef_slnn = Cr * (-sin_gamma * cos_gamma * (J6 + (1 - 2 * material.poisson_ratio) * J5 - x3 * J12) +
                         cos_2gamma * (-x3 * J16) +
                         sin_gamma * cos_gamma * (J6 - x3 * J18))
        
        coef_shsl = Cr * (-sin_gamma * ((1 - material.poisson_ratio) * J9 - x3 * J11) +
                         cos_gamma * (-material.poisson_ratio * J7 - x3 * J19))
        
        coef_shsh = Cr * (-sin_gamma * ((1 - material.poisson_ratio) * J8 - x3 * J13) +
                         cos_gamma * (J6 + material.poisson_ratio * J4 - x3 * J15))
        
        coef_shnn = Cr * (-sin_gamma * (-(1 - 2 * material.poisson_ratio) * J7 - x3 * J19) +
                         cos_gamma * (-x3 * J17))
        
        coef_nnsl = Cr * (sin_gamma**2 * (2 * J8 - x3 * J10) -
                         2 * sin_gamma * cos_gamma * (J6 + material.poisson_ratio * J5 - x3 * J12) +
                         cos_gamma**2 * (-x3 * J16))
        
        coef_nnsh = Cr * (sin_gamma**2 * (2 * material.poisson_ratio * J9 - x3 * J11) -
                         2 * sin_gamma * cos_gamma * (-material.poisson_ratio * J7 - x3 * J19) +
                         cos_gamma**2 * (-x3 * J17))
        
        coef_nnnn = Cr * (sin_gamma**2 * (J6 + (1 - 2 * material.poisson_ratio) * J5 - x3 * J12) -
                         2 * sin_gamma * cos_gamma * (-x3 * J16) +
                         cos_gamma**2 * (J6 - x3 * J18))
        
        return (coef_slsl, coef_slsh, coef_slnn, coef_shsl, coef_shsh, coef_shnn,
                coef_nnsl, coef_nnsh, coef_nnnn)
    
    def _calculate_integrals(
        self,
        x1: float, x2: float, x3: float,
        a: float, b: float,
        r1: float, r2: float, r3: float, r4: float
    ) -> Tuple[float, ...]:
        """Calculate basic integral terms."""
        # I,1
        J1 = (np.log(r1 + x2 - b) - np.log(r2 + x2 + b) -
              np.log(r3 + x2 - b) + np.log(r4 + x2 + b))
        
        # I,2
        J2 = (np.log(r1 + x1 - a) - np.log(r2 + x1 - a) -
              np.log(r3 + x1 + a) + np.log(r4 + x1 + a))
        
        # I,3
        J3 = (-np.arctan((x1 - a) * (x2 - b) / (x3 * r1)) +
              np.arctan((x1 - a) * (x2 + b) / (x3 * r2)) +
              np.arctan((x1 + a) * (x2 - b) / (x3 * r3)) -
              np.arctan((x1 + a) * (x2 + b) / (x3 * r4)))
        
        # I,11
        J4 = ((x1 - a) / (r1 * (r1 + x2 - b)) - (x1 - a) / (r2 * (r2 + x2 + b)) -
              (x1 + a) / (r3 * (r3 + x2 - b)) + (x1 + a) / (r4 * (r4 + x2 + b)))
        
        # I,22
        J5 = ((x2 - b) / (r1 * (r1 + x1 - a)) - (x2 + b) / (r2 * (r2 + x1 - a)) -
              (x2 - b) / (r3 * (r3 + x1 + a)) + (x2 + b) / (r4 * (r4 + x1 + a)))
        
        # I,33
        J6 = ((x1 - a) * (x2 - b) * (x3**2 + r1**2) / (r1 * (x3**2 + (x1 - a)**2) * (x3**2 + (x2 - b)**2)) -
              (x1 - a) * (x2 + b) * (x3**2 + r2**2) / (r2 * (x3**2 + (x1 - a)**2) * (x3**2 + (x2 + b)**2)) -
              (x1 + a) * (x2 - b) * (x3**2 + r3**2) / (r3 * (x3**2 + (x1 + a)**2) * (x3**2 + (x2 - b)**2)) +
              (x1 + a) * (x2 + b) * (x3**2 + r4**2) / (r4 * (x3**2 + (x1 + a)**2) * (x3**2 + (x2 + b)**2)))
        
        # I,12
        J7 = 1/r1 - 1/r2 - 1/r3 + 1/r4
        
        # I,13
        J8 = (x3 / (r1 * (r1 + x2 - b)) - x3 / (r2 * (r2 + x2 + b)) -
              x3 / (r3 * (r3 + x2 - b)) + x3 / (r4 * (r4 + x2 + b)))
        
        # I,23
        J9 = (x3 / (r1 * (r1 + x1 - a)) - x3 / (r2 * (r2 + x1 - a)) -
              x3 / (r3 * (r3 + x1 + a)) + x3 / (r4 * (r4 + x1 + a)))
        
        return (J1, J2, J3, J4, J5, J6, J7, J8, J9)
    
    def _calculate_derivatives(
        self,
        x1: float, x2: float, x3: float,
        a: float, b: float,
        r1: float, r2: float, r3: float, r4: float
    ) -> Tuple[float, ...]:
        """Calculate derivative terms."""
        # I,111
        J10 = ((-(r1 + x2 - b) * ((x1 - a)**2 - r1**2) - (x1 - a)**2 * r1) / (r1**3 * (r1 + x2 - b)**2) -
               (-(r2 + x2 + b) * ((x1 - a)**2 - r2**2) - (x1 - a)**2 * r2) / (r2**3 * (r2 + x2 + b)**2) -
               (-(r3 + x2 - b) * ((x1 + a)**2 - r3**2) - (x1 + a)**2 * r3) / (r3**3 * (r3 + x2 - b)**2) +
               (-(r4 + x2 + b) * ((x1 + a)**2 - r4**2) - (x1 + a)**2 * r4) / (r4**3 * (r4 + x2 + b)**2))
        
        # I,211
        J11 = (-(x1 - a) / r1**3) - (-(x1 - a) / r2**3) - (-(x1 + a) / r3**3) + (-(x1 + a) / r4**3)
        
        # I,311
        J12 = ((-(x1 - a) * x3 * (2 * r1 + x2 - b)) / (r1**3 * (r1 + x2 - b)**2) -
               (-(x1 - a) * x3 * (2 * r2 + x2 + b)) / (r2**3 * (r2 + x2 + b)**2) -
               (-(x1 + a) * x3 * (2 * r3 + x2 - b)) / (r3**3 * (r3 + x2 - b)**2) +
               (-(x1 + a) * x3 * (2 * r4 + x2 + b)) / (r4**3 * (r4 + x2 + b)**2))
        
        # I,311
        J13 = (-(x2 - b) / r1**3) - (-(x2 + b) / r2**3) - (-(x2 - b) / r3**3) + (-(x2 + b) / r4**3)
        
        # I,122
        J14 = ((-(r1 + x1 - a) * ((x2 - b)**2 - r1**2) - (x2 - b)**2 * r1) / (r1**3 * (r1 + x1 - a)**2) -
               (-(r2 + x1 - a) * ((x2 + b)**2 - r2**2) - (x2 + b)**2 * r2) / (r2**3 * (r2 + x1 - a)**2) -
               (-(r3 + x1 + a) * ((x2 - b)**2 - r3**2) - (x2 - b)**2 * r3) / (r3**3 * (r3 + x1 + a)**2) +
               (-(r4 + x1 + a) * ((x2 + b)**2 - r4**2) - (x2 + b)**2 * r4) / (r4**3 * (r4 + x1 + a)**2))
        
        # I,222
        J15 = ((-(x2 - b) * x3 * (2 * r1 + x1 - a)) / (r1**3 * (r1 + x1 - a)**2) -
               (-(x2 + b) * x3 * (2 * r2 + x1 - a)) / (r2**3 * (r2 + x1 - a)**2) -
               (-(x2 - b) * x3 * (2 * r3 + x1 + a)) / (r3**3 * (r3 + x1 + a)**2) +
               (-(x2 + b) * x3 * (2 * r4 + x1 + a)) / (r4**3 * (r4 + x1 + a)**2))
        
        # I,322
        J16 = ((-(r1 + x2 - b) * (x3**2 - r1**2) - x3**2 * r1) / (r1**3 * (r1 + x2 - b)**2) -
               (-(r2 + x2 + b) * (x3**2 - r2**2) - x3**2 * r2) / (r2**3 * (r2 + x2 + b)**2) -
               (-(r3 + x2 - b) * (x3**2 - r3**2) - x3**2 * r3) / (r3**3 * (r3 + x2 - b)**2) +
               (-(r4 + x2 + b) * (x3**2 - r4**2) - x3**2 * r4) / (r4**3 * (r4 + x2 + b)**2))
        
        # I,233
        J17 = ((-(r1 + x1 - a) * (x3**2 - r1**2) - x3**2 * r1) / (r1**3 * (r1 + x1 - a)**2) -
               (-(r2 + x1 - a) * (x3**2 - r2**2) - x3**2 * r2) / (r2**3 * (r2 + x1 - a)**2) -
               (-(r3 + x1 + a) * (x3**2 - r3**2) - x3**2 * r3) / (r3**3 * (r3 + x1 + a)**2) +
               (-(r4 + x1 + a) * (x3**2 - r4**2) - x3**2 * r4) / (r4**3 * (r4 + x1 + a)**2))
        
        # I,333
        J18 = ((-x3 * (x1 - a) * (x2 - b)) * 
               ((x3**2 + (x1 - a)**2)**2 * (x3**2 + (x2 - b)**2 + 2 * r1**2) +
                (x3**2 + (x2 - b)**2)**2 * (x3**2 + (x1 - a)**2 + 2 * r1**2)) / 
               (r1**3 * (x3**2 + (x1 - a)**2)**2 * (x3**2 + (x2 - b)**2)**2) -
               (-x3 * (x1 - a) * (x2 + b)) * 
               ((x3**2 + (x1 - a)**2)**2 * (x3**2 + (x2 + b)**2 + 2 * r2**2) +
                (x3**2 + (x2 + b)**2)**2 * (x3**2 + (x1 - a)**2 + 2 * r2**2)) / 
               (r2**3 * (x3**2 + (x1 - a)**2)**2 * (x3**2 + (x2 + b)**2)**2) -
               (-x3 * (x1 + a) * (x2 - b)) * 
               ((x3**2 + (x1 + a)**2)**2 * (x3**2 + (x2 - b)**2 + 2 * r3**2) +
                (x3**2 + (x2 - b)**2)**2 * (x3**2 + (x1 + a)**2 + 2 * r3**2)) / 
               (r3**3 * (x3**2 + (x1 + a)**2)**2 * (x3**2 + (x2 - b)**2)**2) +
               (-x3 * (x1 + a) * (x2 + b)) * 
               ((x3**2 + (x1 + a)**2)**2 * (x3**2 + (x2 + b)**2 + 2 * r4**2) +
                (x3**2 + (x2 + b)**2)**2 * (x3**2 + (x1 + a)**2 + 2 * r4**2)) / 
               (r4**3 * (x3**2 + (x1 + a)**2)**2 * (x3**2 + (x2 + b)**2)**2))
        
        # I,123
        J19 = (-x3 / r1**3) - (-x3 / r2**3) - (-x3 / r3**3) + (-x3 / r4**3)
        
        return (J10, J11, J12, J13, J14, J15, J16, J17, J18, J19)
    
    def calculate_fiber_response(
        self,
        fractures: List[Fracture],
        fibers: List[Fiber]
    ) -> None:
        """
        Calculate stress and displacement response at fiber channels.
        
        Parameters
        ----------
        fractures : List[Fracture]
            List of fractures
        fibers : List[Fiber]
            List of DAS fibers
        """
        for fiber in fibers:
            for channel in fiber.channels:
                self._calculate_channel_response(channel, fractures)
    
    def _calculate_channel_response(
        self,
        channel: Channel,
        fractures: List[Fracture]
    ) -> None:
        """Calculate response at a single channel."""
        # Initialize stress and displacement
        sxx = syy = szz = sxy = sxz = syz = 0.0
        uxx = uyy = uzz = 0.0
        
        for fracture in fractures:
            for element in fracture.elements:
                # Calculate contribution from this element
                stress_contrib, disp_contrib = self._calculate_element_contribution(
                    channel, element, fracture.material
                )
                
                # Accumulate contributions
                sxx += stress_contrib[0]
                syy += stress_contrib[1]
                szz += stress_contrib[2]
                sxy += stress_contrib[3]
                sxz += stress_contrib[4]
                syz += stress_contrib[5]
                
                uxx += disp_contrib[0]
                uyy += disp_contrib[1]
                uzz += disp_contrib[2]
        
        # Calculate strains
        youngs_modulus = 2.0 * fractures[0].material.shear_modulus * (1.0 + fractures[0].material.poisson_ratio)
        exx = (sxx - fractures[0].material.poisson_ratio * (syy + szz)) / youngs_modulus
        eyy = (syy - fractures[0].material.poisson_ratio * (sxx + szz)) / youngs_modulus
        ezz = (szz - fractures[0].material.poisson_ratio * (sxx + syy)) / youngs_modulus
        
        # Store results
        channel.add_stress_data(sxx, syy, szz, sxy, sxz, syz)
        channel.add_strain_data(exx, eyy, ezz)
        channel.add_displacement_data(uxx, uyy, uzz)
    
    def _calculate_element_contribution(
        self,
        channel: Channel,
        element: "DisplacementDiscontinuityElement",
        material: Material
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """Calculate stress and displacement contribution from an element."""
        # Transform coordinates
        cos_beta = np.cos(np.deg2rad(element.strike))
        cos_2beta = np.cos(np.deg2rad(2 * element.strike))
        sin_beta = np.sin(np.deg2rad(element.strike))
        sin_2beta = np.sin(np.deg2rad(2 * element.strike))
        
        x1 = (channel.x - element.x) * cos_beta + (channel.y - element.y) * sin_beta
        x2 = channel.z - element.z
        x3 = -(channel.x - element.x) * sin_beta + (channel.y - element.y) * cos_beta
        
        # Element dimensions
        a = element.length / 2.0
        b = element.height / 2.0
        
        # Calculate coefficient
        Cr = material.shear_modulus / (4.0 * np.pi * (1.0 - material.poisson_ratio))
        
        # Calculate distance terms
        r1 = np.sqrt((x1 - a)**2 + (x2 - b)**2 + x3**2)
        r2 = np.sqrt((x1 - a)**2 + (x2 + b)**2 + x3**2)
        r3 = np.sqrt((x1 + a)**2 + (x2 - b)**2 + x3**2)
        r4 = np.sqrt((x1 + a)**2 + (x2 + b)**2 + x3**2)
        
        # Calculate integral terms
        J1, J2, J3, J4, J5, J6, J7, J8, J9 = self._calculate_integrals(
            x1, x2, x3, a, b, r1, r2, r3, r4
        )
        
        J10, J11, J12, J13, J14, J15, J16, J17, J18, J19 = self._calculate_derivatives(
            x1, x2, x3, a, b, r1, r2, r3, r4
        )
        
        # Calculate stress in local coordinates
        SS11 = (Cr * element.dsl * (2 * J8 - x3 * J10) +
                Cr * element.dsh * (2 * material.poisson_ratio * J9 - x3 * J11) +
                Cr * element.dnn * (J6 + (1 - 2 * material.poisson_ratio) * J5 - x3 * J12))
        
        SS22 = (Cr * element.dsl * (2 * material.poisson_ratio * J8 - x3 * J13) +
                Cr * element.dsh * (2 * J9 - x3 * J14) +
                Cr * element.dnn * (J6 + (1 - 2 * material.poisson_ratio) * J4 - x3 * J15))
        
        SS33 = (Cr * element.dsl * (-x3 * J16) +
                Cr * element.dsh * (-x3 * J17) +
                Cr * element.dnn * (J6 - x3 * J18))
        
        SS12 = (Cr * element.dsl * ((1 - material.poisson_ratio) * J9 - x3 * J11) +
                Cr * element.dsh * ((1 - material.poisson_ratio) * J8 - x3 * J13) +
                Cr * element.dnn * (-(1 - 2 * material.poisson_ratio) * J7 - x3 * J19))
        
        SS13 = (Cr * element.dsl * (J6 + material.poisson_ratio * J5 - x3 * J12) +
                Cr * element.dsh * (-material.poisson_ratio * J7 - x3 * J19) +
                Cr * element.dnn * (-x3 * J16))
        
        SS23 = (Cr * element.dsl * (-material.poisson_ratio * J7 - x3 * J19) +
                Cr * element.dsh * (J6 + material.poisson_ratio * J4 - x3 * J15) +
                Cr * element.dnn * (-x3 * J17))
        
        # Transform stress to global coordinates
        sxx = cos_beta**2 * SS11 - sin_2beta * SS13 + sin_beta**2 * SS33
        syy = sin_beta**2 * SS11 + sin_2beta * SS13 + cos_beta**2 * SS33
        szz = SS22
        sxy = sin_beta * cos_beta * SS11 + cos_2beta * SS13 - sin_beta * cos_beta * SS33
        sxz = cos_beta * SS12 - sin_beta * SS23
        syz = sin_beta * SS12 + cos_beta * SS23
        
        # Calculate displacement in local coordinates
        U1 = ((2 * (1 - material.poisson_ratio) * element.dsl * J3 -
               (1 - 2 * material.poisson_ratio) * element.dnn * J1 -
               x3 * (element.dsl * J4 + element.dsh * J7 + element.dnn * J8)) /
              (8 * np.pi * (1 - material.poisson_ratio)))
        
        U2 = ((2 * (1 - material.poisson_ratio) * element.dsh * J3 -
               (1 - 2 * material.poisson_ratio) * element.dnn * J2 -
               x3 * (element.dsl * J7 + element.dsh * J5 + element.dnn * J9)) /
              (8 * np.pi * (1 - material.poisson_ratio)))
        
        U3 = ((2 * (1 - material.poisson_ratio) * element.dnn * J3 +
               (1 - 2 * material.poisson_ratio) * (element.dsl * J1 + element.dsh * J2) -
               x3 * (element.dsl * J8 + element.dsh * J9 + element.dnn * J6)) /
              (8 * np.pi * (1 - material.poisson_ratio)))
        
        # Transform displacement to global coordinates
        uxx = cos_beta * U1 - sin_beta * U3
        uyy = sin_beta * U1 + cos_beta * U3
        uzz = U2
        
        return ((sxx, syy, szz, sxy, sxz, syz), (uxx, uyy, uzz))
    
    def calculate_plane_response(
        self,
        fractures: List[Fracture],
        planes: List[Plane]
    ) -> None:
        """
        Calculate stress and displacement response at plane nodes.
        
        Parameters
        ----------
        fractures : List[Fracture]
            List of fractures
        planes : List[Plane]
            List of monitoring planes
        """
        for plane in planes:
            for node in plane.nodes:
                self._calculate_node_response(node, fractures)
    
    def _calculate_node_response(
        self,
        node: dict,
        fractures: List[Fracture]
    ) -> None:
        """Calculate response at a single node."""
        # Initialize stress and displacement
        sxx = syy = szz = sxy = sxz = syz = 0.0
        uxx = uyy = uzz = 0.0
        
        for fracture in fractures:
            for element in fracture.elements:
                # Calculate contribution from this element
                stress_contrib, disp_contrib = self._calculate_element_contribution(
                    type('Channel', (), {'x': node['position'][0], 'y': node['position'][1], 'z': node['position'][2]})(),
                    element, fracture.material
                )
                
                # Accumulate contributions
                sxx += stress_contrib[0]
                syy += stress_contrib[1]
                szz += stress_contrib[2]
                sxy += stress_contrib[3]
                sxz += stress_contrib[4]
                syz += stress_contrib[5]
                
                uxx += disp_contrib[0]
                uyy += disp_contrib[1]
                uzz += disp_contrib[2]
        
        # Calculate strains
        youngs_modulus = 2.0 * fractures[0].material.shear_modulus * (1.0 + fractures[0].material.poisson_ratio)
        exx = (sxx - fractures[0].material.poisson_ratio * (syy + szz)) / youngs_modulus
        eyy = (syy - fractures[0].material.poisson_ratio * (sxx + szz)) / youngs_modulus
        ezz = (szz - fractures[0].material.poisson_ratio * (sxx + syy)) / youngs_modulus
        
        # Store results
        plane.add_stress_data(node['id'], sxx, syy, szz, sxy, sxz, syz)
        plane.add_strain_data(node['id'], exx, eyy, ezz)
        plane.add_displacement_data(node['id'], uxx, uyy, uzz)

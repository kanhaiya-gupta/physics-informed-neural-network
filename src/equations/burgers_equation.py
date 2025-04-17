import torch
from .base_equation import BaseEquation

class BurgersEquation(BaseEquation):
    """
    Implementation of Burgers' equation:
    u_t + u*u_x - nu*u_xx = 0
    """
    def __init__(self, nu=0.01):
        """
        Initialize Burgers' equation
        Args:
            nu: viscosity coefficient
        """
        super().__init__()
        self.nu = nu

    def pde_residual(self, x, t, u, u_x, u_xx, u_t):
        """
        Compute Burgers' equation residual
        Args:
            x: spatial coordinate
            t: temporal coordinate
            u: solution
            u_x: first spatial derivative
            u_xx: second spatial derivative
            u_t: temporal derivative
        Returns:
            residual: PDE residual
        """
        # Burgers' equation: u_t + u*u_x - nu*u_xx = 0
        return u_t + u*u_x - self.nu*u_xx

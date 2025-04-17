import torch
from .base_equation import BaseEquation

class WaveEquation(BaseEquation):
    """
    Implementation of the wave equation:
    u_tt - c^2*u_xx = 0
    """
    def __init__(self, c=1.0):
        """
        Initialize wave equation
        Args:
            c: wave speed
        """
        super().__init__()
        self.c = c

    def pde_residual(self, x, t, u, u_x, u_xx, u_t):
        """
        Compute wave equation residual
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
        # Wave equation: u_tt - c^2*u_xx = 0
        # Note: u_tt needs to be computed separately as it's not provided in the base method
        u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t),
                                 create_graph=True)[0]
        return u_tt - (self.c**2)*u_xx

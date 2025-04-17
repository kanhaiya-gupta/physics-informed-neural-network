import torch
from .base_equation import BaseEquation

class SHMEquation(BaseEquation):
    """
    Simple Harmonic Motion (SHM) equation:
    d²x/dt² + ω²x = 0
    where:
    - x is the displacement
    - t is time
    - ω is the angular frequency
    """
    
    def __init__(self, omega=1.0):
        """
        Initialize the SHM equation.
        
        Args:
            omega (float): Angular frequency of the oscillation
        """
        super().__init__()
        self.omega = omega
        
    def compute_residual(self, x, t, model):
        """
        Compute the residual of the SHM equation.
        
        Args:
            x (torch.Tensor): Spatial coordinates (not used in SHM)
            t (torch.Tensor): Time coordinates
            model (BasePINN): The neural network model
            
        Returns:
            torch.Tensor: Residual of the SHM equation
        """
        # Get the prediction and its derivatives
        u = model(t)
        
        # Compute first and second time derivatives
        du_dt = torch.autograd.grad(u, t, 
                                  grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]
        d2u_dt2 = torch.autograd.grad(du_dt, t,
                                     grad_outputs=torch.ones_like(du_dt),
                                     create_graph=True)[0]
        
        # Compute the residual: d²x/dt² + ω²x
        residual = d2u_dt2 + (self.omega ** 2) * u
        
        return residual
    
    def exact_solution(self, t):
        """
        Compute the exact solution of the SHM equation.
        For initial conditions x(0) = 1, dx/dt(0) = 0:
        x(t) = cos(ωt)
        
        Args:
            t (torch.Tensor): Time coordinates
            
        Returns:
            torch.Tensor: Exact solution
        """
        return torch.cos(self.omega * t) 
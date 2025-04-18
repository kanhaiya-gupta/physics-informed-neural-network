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

    def compute_pde_loss(self, x, t, u):
        """
        Compute the PDE loss for the Wave equation.
        
        Args:
            x (torch.Tensor): Spatial coordinates
            t (torch.Tensor): Time coordinates
            u (torch.Tensor): Predicted solution
            
        Returns:
            torch.Tensor: PDE loss
        """
        # Ensure x and t require gradients
        x = x.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)
        u = u.clone().detach().requires_grad_(True)
        
        # Compute derivatives with allow_unused=True
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]
        if u_x is None:
            u_x = torch.zeros_like(x)
            
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, allow_unused=True)[0]
        if u_xx is None:
            u_xx = torch.zeros_like(x)
            
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]
        if u_t is None:
            u_t = torch.zeros_like(t)
            
        u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True, allow_unused=True)[0]
        if u_tt is None:
            u_tt = torch.zeros_like(t)
        
        # Wave equation: u_tt = c^2 * u_xx
        pde_residual = u_tt - (self.c ** 2) * u_xx
        
        return torch.mean(pde_residual ** 2)

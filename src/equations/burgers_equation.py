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

    def compute_pde_loss(self, x, t, u):
        """
        Compute the PDE loss for Burgers' equation.
        
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
        
        # Burgers' equation: u_t + u*u_x = ν*u_xx
        pde_residual = u_t + u * u_x - self.nu * u_xx
        
        return torch.mean(pde_residual ** 2)

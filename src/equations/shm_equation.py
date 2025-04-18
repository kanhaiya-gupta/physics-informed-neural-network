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

    def compute_pde_loss(self, t, u):
        """
        Compute the PDE loss for Simple Harmonic Motion.
        
        Args:
            t (torch.Tensor): Time points
            u (torch.Tensor): Predicted solution
            
        Returns:
            torch.Tensor: PDE loss
        """
        # Ensure t requires gradients
        t = t.clone().detach().requires_grad_(True)
        
        # Compute derivatives with allow_unused=True
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True, allow_unused=True)[0]
        if u_t is None:
            u_t = torch.zeros_like(t)
            
        u_tt = torch.autograd.grad(u_t.sum(), t, create_graph=True, allow_unused=True)[0]
        if u_tt is None:
            u_tt = torch.zeros_like(t)
        
        # PDE residual: d^2x/dt^2 + ω^2x = 0
        pde_residual = u_tt + (self.omega ** 2) * u
        
        return torch.mean(pde_residual ** 2)
    
    def compute_ic_loss(self, t, u):
        """
        Compute the initial condition loss.
        
        Args:
            t (torch.Tensor): Time points
            u (torch.Tensor): Predicted solution
            
        Returns:
            torch.Tensor: Initial condition loss
        """
        # Initial conditions: x(0) = 1, dx/dt(0) = 0
        t0_mask = torch.abs(t) < 1e-6
        x0 = u[t0_mask]
        
        if len(x0) == 0:
            return torch.tensor(0.0, device=t.device)
        
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        v0 = u_t[t0_mask]
        
        ic_loss = torch.mean((x0 - 1.0) ** 2) + torch.mean(v0 ** 2)
        return ic_loss 
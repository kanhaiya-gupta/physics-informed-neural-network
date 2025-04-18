import torch

class HeatEquation:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Thermal diffusivity

    def pde_residual(self, x, t, u, u_x, u_xx, u_t):
        # Heat equation: u_t = alpha * u_xx
        return u_t - self.alpha * u_xx
        
    def compute_pde_loss(self, points, u):
        """
        Compute the PDE loss for the Heat equation.
        
        Args:
            points (torch.Tensor): Input points (x, t)
            u (torch.Tensor): Predicted solution
            
        Returns:
            torch.Tensor: PDE loss
        """
        # Ensure points require gradients and are detached from previous computation
        points = points.clone().detach().requires_grad_(True)
        x = points[:, 0:1]
        t = points[:, 1:2]
        
        # Ensure u is part of the computation graph
        u = u.clone().detach().requires_grad_(True)
        
        # Compute derivatives with allow_unused=True
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True, allow_unused=True)[0]
        if u_x is None:
            u_x = torch.zeros_like(x, requires_grad=True)
            
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, allow_unused=True)[0]
        if u_xx is None:
            u_xx = torch.zeros_like(x, requires_grad=True)
            
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True, allow_unused=True)[0]
        if u_t is None:
            u_t = torch.zeros_like(t, requires_grad=True)
        
        # Compute PDE residual
        residual = self.pde_residual(x, t, u, u_x, u_xx, u_t)
        
        return residual.pow(2).mean()

    def compute_bc_loss(self, points, u):
        """
        Compute the boundary condition loss for the Heat equation.
        For this example, we use Dirichlet boundary conditions:
        u(0,t) = u(1,t) = 0
        
        Args:
            points (torch.Tensor): Input points (x, t)
            u (torch.Tensor): Predicted solution
            
        Returns:
            torch.Tensor: Boundary condition loss
        """
        x = points[:, 0:1]
        t = points[:, 1:2]
        
        # Find points at boundaries (x=0 and x=1)
        left_boundary = torch.abs(x) < 1e-6
        right_boundary = torch.abs(x - 1.0) < 1e-6
        
        # Get predictions at boundaries
        u_left = u[left_boundary]
        u_right = u[right_boundary]
        
        # Compute boundary condition loss
        bc_loss = torch.mean(u_left ** 2) + torch.mean(u_right ** 2)
        
        return bc_loss

    def compute_ic_loss(self, points, u):
        """
        Compute the initial condition loss for the Heat equation.
        For this example, we use u(x,0) = sin(πx)
        
        Args:
            points (torch.Tensor): Input points (x, t)
            u (torch.Tensor): Predicted solution
            
        Returns:
            torch.Tensor: Initial condition loss
        """
        x = points[:, 0:1]
        t = points[:, 1:2]
        
        # Find points at initial time (t=0)
        initial_time = torch.abs(t) < 1e-6
        
        if not torch.any(initial_time):
            return torch.tensor(0.0, device=u.device)
        
        # Get predictions at initial time
        u_initial = u[initial_time]
        x_initial = x[initial_time]
        
        # Compute exact initial condition
        u_exact = torch.sin(torch.pi * x_initial)
        
        # Compute initial condition loss
        ic_loss = torch.mean((u_initial - u_exact) ** 2)
        
        return ic_loss
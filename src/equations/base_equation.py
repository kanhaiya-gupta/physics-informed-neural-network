import torch

class BaseEquation:
    """
    Base class for defining partial differential equations (PDEs)
    All specific PDE implementations should inherit from this class
    """
    def __init__(self):
        pass

    def pde_residual(self, x, t, u, u_x, u_xx, u_t):
        """
        Compute the PDE residual
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
        raise NotImplementedError("Subclasses must implement pde_residual")

    def compute_derivatives(self, x, t, u):
        """
        Compute necessary derivatives for the PDE
        Args:
            x: spatial coordinate
            t: temporal coordinate
            u: solution
        Returns:
            u_x: first spatial derivative
            u_xx: second spatial derivative
            u_t: temporal derivative
        """
        # Enable gradient computation
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        # Compute derivatives
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                 create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
        
        return u_x, u_xx, u_t

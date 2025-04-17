from .base_pinn import BasePINN
import torch

class BurgersPINN(BasePINN):
    def __init__(self, nu=0.01):
        """
        Initialize Burgers' equation PINN
        Args:
            nu: viscosity coefficient
        """
        super(BurgersPINN, self).__init__(layers=[2, 20, 20, 1])  # Input: (x, t), Output: u
        self.nu = nu

    def predict(self, x, t):
        """
        Predict solution u(x,t)
        Args:
            x: spatial coordinate
            t: temporal coordinate
        Returns:
            u: predicted solution
        """
        inputs = torch.cat([x, t], dim=1)
        return self.forward(inputs)

    def compute_loss(self, x, t, u_true=None):
        """
        Compute physics-informed loss for Burgers' equation
        Args:
            x: spatial coordinate
            t: temporal coordinate
            u_true: true solution (if available for supervised loss)
        Returns:
            loss: total loss combining PDE loss and supervised loss
        """
        # Enable gradient computation
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        # Get prediction
        u = self.predict(x, t)
        
        # Compute derivatives
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                 create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
        
        # Burgers' equation: u_t + u*u_x - nu*u_xx = 0
        pde_loss = u_t + u*u_x - self.nu*u_xx
        
        # Compute total loss
        loss = torch.mean(pde_loss**2)
        
        # Add supervised loss if true solution is provided
        if u_true is not None:
            supervised_loss = torch.mean((u - u_true)**2)
            loss += supervised_loss
            
        return loss

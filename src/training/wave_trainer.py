import torch
from src.training.trainer import BaseTrainer
from src.models.wave_pinn import WavePINN
from src.equations.wave_equation import WaveEquation
from src.data.generators.wave_data import WaveDataGenerator
from src.data.initial_conditions.ic_wave import wave_initial_condition
from src.data.boundary_conditions.bc_wave import wave_boundary_condition

class WaveTrainer(BaseTrainer):
    """
    Trainer for wave equation PINN
    """
    def __init__(self, c=1.0, lr=0.001):
        """
        Initialize wave equation trainer
        Args:
            c: wave speed
            lr: learning rate
        """
        model = WavePINN(c=c)
        equation = WaveEquation(c=c)
        data_generator = WaveDataGenerator()
        super().__init__(model, equation, data_generator, lr=lr)

    def compute_loss(self, x, t, u_true=None):
        """
        Compute total loss including PDE loss and boundary/initial conditions
        Args:
            x: spatial coordinate
            t: temporal coordinate
            u_true: true solution (if available)
        Returns:
            loss: total loss
        """
        # Enable gradient computation
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        # Get prediction
        u = self.model.predict(x, t)
        
        # Compute derivatives
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                 create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t),
                                 create_graph=True)[0]
        
        # Compute PDE loss
        pde_loss = self.equation.pde_residual(x, t, u, u_x, u_xx, u_t)
        pde_loss = torch.mean(pde_loss**2)
        
        # Compute initial condition loss
        ic_loss = wave_initial_condition(x, t, u)
        
        # Compute boundary condition loss
        bc_loss = wave_boundary_condition(x, t, u)
        
        # Combine losses
        loss = pde_loss + ic_loss + bc_loss
        
        # Add supervised loss if true solution is provided
        if u_true is not None:
            supervised_loss = torch.mean((u - u_true)**2)
            loss += supervised_loss
            
        return loss

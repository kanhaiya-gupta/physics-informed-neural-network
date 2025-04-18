import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from src.training.trainer import BaseTrainer
from src.models.burgers_pinn import BurgersPINN
from src.equations.burgers_equation import BurgersEquation
from src.data.generators.burgers_data import BurgersDataGenerator
from src.data.initial_conditions.ic_burgers import burgers_initial_condition
from src.data.boundary_conditions.bc_burgers import burgers_boundary_condition

class BurgersTrainer(BaseTrainer):
    """
    Trainer for Burgers' equation PINN
    """
    def __init__(self, nu=0.01, lr=0.001):
        """
        Initialize Burgers' equation trainer
        Args:
            nu: viscosity coefficient
            lr: learning rate
        """
        model = BurgersPINN(nu=nu)
        equation = BurgersEquation(nu=nu)
        data_generator = BurgersDataGenerator()
        super().__init__(model, equation, data_generator, lr=lr)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Create necessary directories
        self.results_dir = "results/burgers"
        self.models_dir = os.path.join(self.results_dir, "models")
        self.plots_dir = os.path.join(self.results_dir, "plots")
        self.metrics_dir = os.path.join(self.results_dir, "metrics")
        
        for dir_path in [self.results_dir, self.models_dir, self.plots_dir, self.metrics_dir]:
            os.makedirs(dir_path, exist_ok=True)

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
        
        # Compute PDE loss
        pde_loss = self.equation.pde_residual(x, t, u, u_x, u_xx, u_t)
        pde_loss = torch.mean(pde_loss**2)
        
        # Compute initial condition loss
        ic_loss = burgers_initial_condition(x, t, u)
        
        # Compute boundary condition loss
        bc_loss = burgers_boundary_condition(x, t, u)
        
        # Combine losses
        loss = pde_loss + ic_loss + bc_loss
        
        # Add supervised loss if true solution is provided
        if u_true is not None:
            supervised_loss = torch.mean((u - u_true)**2)
            loss += supervised_loss
            
        return loss

    def train(self, epochs=1000, lr=0.001):
        self.loss_history = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Generate collocation points
            x, t = self.data_generator.generate_collocation_points()
            x = x.to(self.device)
            t = t.to(self.device)
            
            # Compute loss
            loss = self.compute_loss(x, t)
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())
            
            # Log progress
            if self.log_progress:
                self.log_progress(epoch, float(loss))
            else:
                print(f"Epoch {epoch}, Loss: {float(loss)}")
            
            # Save model periodically
            if (epoch + 1) % 100 == 0:
                self.save_model()
        
        # Save final model
        self.save_model()
        return float(loss)

    def save_model(self):
        # Ensure directory exists
        save_dir = os.path.join("results", "burgers", "models")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, "model.pth")
        torch.save(self.model.state_dict(), model_path)
        
        # Save loss history
        loss_path = os.path.join(self.metrics_dir, "loss_history.npy")
        np.save(loss_path, np.array(self.loss_history))
        
        # Generate and save plots
        self._plot_loss_curve()
        self._plot_solution()
        self._plot_comparison()

    def _plot_loss_curve(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.yscale('log')
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'loss_curve.png'))
        plt.close()

    def _plot_solution(self):
        # Generate grid points for plotting
        x = torch.linspace(0, 1, 100).view(-1, 1)
        t = torch.linspace(0, 1, 100).view(-1, 1)
        X, T = torch.meshgrid(x.squeeze(), t.squeeze())
        
        # Reshape for prediction
        x_flat = X.reshape(-1, 1)
        t_flat = T.reshape(-1, 1)
        
        # Get predictions
        with torch.no_grad():
            u_pred = self.model.predict(x_flat, t_flat)
            u_pred = u_pred.reshape(X.shape)
        
        # Calculate exact solution
        u_exact = self._exact_solution(X, T)
        
        # Plot predicted solution
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.pcolormesh(X.numpy(), T.numpy(), u_pred.numpy(), shading='auto')
        plt.colorbar(label='u(x,t)')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Predicted Solution')
        
        # Plot exact solution
        plt.subplot(1, 2, 2)
        plt.pcolormesh(X.numpy(), T.numpy(), u_exact.numpy(), shading='auto')
        plt.colorbar(label='u(x,t)')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Exact Solution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'solution_comparison.png'))
        plt.close()

    def _plot_comparison(self):
        # Generate points for comparison
        t_fixed = 0.5  # Choose a fixed time
        x = torch.linspace(0, 1, 100).view(-1, 1)
        t = torch.ones_like(x) * t_fixed
        
        # Get predictions
        with torch.no_grad():
            u_pred = self.model.predict(x, t)
        
        # Calculate exact solution
        u_exact = self._exact_solution(x, t)
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.plot(x.numpy(), u_pred.numpy(), 'r-', label='Predicted')
        plt.plot(x.numpy(), u_exact.numpy(), 'b--', label='Exact')
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.title(f'Solution Comparison at t={t_fixed}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'solution_slice.png'))
        plt.close()

    def _exact_solution(self, x, t):
        """Calculate the exact solution for the Burgers' equation with initial condition u(x,0) = -sin(πx)"""
        # For the Burgers' equation with initial condition u(x,0) = -sin(πx)
        # and boundary conditions u(0,t) = u(1,t) = 0
        # The exact solution is u(x,t) = -sin(πx) * exp(-π²t)
        # Convert inputs to numpy arrays for calculation
        x_np = x.detach().numpy()
        t_np = t.detach().numpy()
        
        # Calculate exact solution
        u_exact = -np.sin(np.pi * x_np) * np.exp(-np.pi**2 * t_np)
        
        # Convert back to torch tensor
        return torch.tensor(u_exact, dtype=torch.float32)

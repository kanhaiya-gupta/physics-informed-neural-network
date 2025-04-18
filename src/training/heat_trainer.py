import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from src.models.heat_pinn import HeatPINN
from src.equations.heat_equation import HeatEquation
from src.data.generators.heat_data import HeatDataGenerator
from src.data.initial_conditions.ic_heat import heat_initial_condition
from src.data.boundary_conditions.bc_heat import heat_boundary_condition
import time

class HeatTrainer:
    def __init__(self, alpha=0.1):
        """Initialize the Heat equation trainer."""
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HeatPINN().to(self.device)
        self.equation = HeatEquation(alpha=alpha)
        self.data_generator = HeatDataGenerator()
        self.optimizer = None  # Will be initialized in train method
        self.loss_history = []
        self.log_progress = None  # Custom logging function
        
        # Create necessary directories
        self.data_dir = "data/heat"
        self.results_dir = "results/heat"
        self.models_dir = os.path.join(self.results_dir, "models")
        self.plots_dir = os.path.join(self.results_dir, "plots")
        self.metrics_dir = os.path.join(self.results_dir, "metrics")
        
        for dir_path in [self.data_dir, self.results_dir, self.models_dir, self.plots_dir, self.metrics_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def train(self, epochs=1000, lr=0.001, batch_size=32):
        """Train the model."""
        start_time = time.time()
        
        # Initialize optimizer with the provided learning rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Generate training data and ensure it requires gradients
        x = torch.linspace(0, 1, batch_size, requires_grad=True).reshape(-1, 1).to(self.device)
        t = torch.linspace(0, 1, batch_size, requires_grad=True).reshape(-1, 1).to(self.device)
        
        # Save generated data
        torch.save(x, os.path.join(self.data_dir, "x.pt"))
        torch.save(t, os.path.join(self.data_dir, "t.pt"))
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            points = torch.cat([x, t], dim=1)
            u_pred = self.model(points)
            
            # Calculate losses
            pde_loss = self.equation.compute_pde_loss(points, u_pred)
            bc_loss = self.equation.compute_bc_loss(points, u_pred)
            ic_loss = self.equation.compute_ic_loss(points, u_pred)
            
            # Total loss
            loss = pde_loss + bc_loss + ic_loss
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Log progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
                self.loss_history.append(loss.item())
        
        # Save the trained model and data
        self.save_model()
        
        training_time = time.time() - start_time
        final_loss = float(loss.item())
        training_time = float(training_time)
        
        return final_loss, training_time

    def save_model(self):
        # Ensure directory exists
        save_dir = os.path.join("results", "heat", "models")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, "model.pth")
        torch.save(self.model.state_dict(), model_path)
        
        # Save loss history
        loss_path = os.path.join(self.metrics_dir, "loss_history.npy")
        np.save(loss_path, np.array(self.loss_history))
        
        # Save training data
        data_path = os.path.join(self.data_dir, "training_data.pt")
        torch.save({
            'x': torch.load(os.path.join(self.data_dir, "x.pt")),
            't': torch.load(os.path.join(self.data_dir, "t.pt")),
            'loss_history': self.loss_history
        }, data_path)
        
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
        """Calculate the exact solution for the heat equation with initial condition u(x,0) = sin(πx)"""
        # For the heat equation with initial condition u(x,0) = sin(πx)
        # and boundary conditions u(0,t) = u(1,t) = 0
        # The exact solution is u(x,t) = sin(πx) * exp(-π²t)
        # Convert inputs to numpy arrays for calculation
        x_np = x.detach().numpy()
        t_np = t.detach().numpy()
        
        # Calculate exact solution
        u_exact = np.sin(np.pi * x_np) * np.exp(-np.pi**2 * t_np)
        
        # Convert back to torch tensor
        return torch.tensor(u_exact, dtype=torch.float32)
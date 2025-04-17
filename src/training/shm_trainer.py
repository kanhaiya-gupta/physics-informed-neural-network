import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from ..models.shm_pinn import SHMPINN
from ..equations.shm_equation import SHMEquation
from .trainer import Trainer

class SHMTrainer(Trainer):
    """
    Trainer for the Simple Harmonic Motion PINN.
    """
    
    def __init__(self, omega=1.0, layers=None):
        """
        Initialize the SHM trainer.
        
        Args:
            omega (float): Angular frequency of the oscillation
            layers (list): List of layer sizes for the neural network
        """
        super().__init__()
        self.model = SHMPINN(layers)
        self.equation = SHMEquation(omega)
        self.loss_history = []
        
        # Create results directory
        self.results_dir = os.path.join("results", "shm")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "metrics"), exist_ok=True)
        
    def train(self, epochs=1000, lr=0.001):
        """
        Train the SHM PINN model.
        
        Args:
            epochs (int): Number of training epochs
            lr (float): Learning rate
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Generate training points
        t = torch.linspace(0, 2*np.pi, 100, requires_grad=True).view(-1, 1)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Compute physics loss
            residual = self.equation.compute_residual(None, t, self.model)
            physics_loss = torch.mean(residual ** 2)
            
            # Compute initial condition loss
            t_ic = torch.tensor([[0.0]], requires_grad=True)
            u_ic = self.model(t_ic)
            du_dt_ic = torch.autograd.grad(u_ic, t_ic,
                                         grad_outputs=torch.ones_like(u_ic),
                                         create_graph=True)[0]
            ic_loss = (u_ic - 1.0) ** 2 + (du_dt_ic - 0.0) ** 2
            
            # Total loss
            loss = physics_loss + ic_loss
            
            loss.backward()
            optimizer.step()
            
            self.loss_history.append(loss.item())
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        
        # Save results
        self._save_results()
        
    def _save_results(self):
        """Save the trained model and generate plots."""
        # Save model
        torch.save(self.model.state_dict(),
                  os.path.join(self.results_dir, "models", "model.pth"))
        
        # Save loss history
        np.save(os.path.join(self.results_dir, "metrics", "loss_history.npy"),
                np.array(self.loss_history))
        
        # Generate and save plots
        self._plot_loss_curve()
        self._plot_solution()
        
    def _plot_loss_curve(self):
        """Plot and save the loss curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.savefig(os.path.join(self.results_dir, "plots", "loss_curve.png"))
        plt.close()
        
    def _plot_solution(self):
        """Plot and save the predicted and exact solutions."""
        t = torch.linspace(0, 2*np.pi, 100).view(-1, 1)
        with torch.no_grad():
            prediction = self.model(t)
            exact = self.equation.exact_solution(t)
        
        plt.figure(figsize=(10, 6))
        plt.plot(t.numpy(), prediction.numpy(), label="Predicted")
        plt.plot(t.numpy(), exact.numpy(), '--', label="Exact")
        plt.xlabel("Time")
        plt.ylabel("Displacement")
        plt.title("SHM Solution")
        plt.legend()
        plt.savefig(os.path.join(self.results_dir, "plots", "solution_comparison.png"))
        plt.close() 
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """
    Base class for training physics-informed neural networks
    All specific trainers should inherit from this class
    """
    def __init__(self, model, equation, data_generator, lr=0.001):
        """
        Initialize the trainer
        Args:
            model: PINN model
            equation: PDE equation
            data_generator: data generator for collocation points
            lr: learning rate
        """
        self.model = model
        self.equation = equation
        self.data_generator = data_generator
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.loss_history = []

    @abstractmethod
    def compute_loss(self, x, t, u_true=None):
        """
        Compute the total loss including PDE loss and boundary/initial conditions
        Args:
            x: spatial coordinate
            t: temporal coordinate
            u_true: true solution (if available for supervised loss)
        Returns:
            loss: total loss
        """
        pass

    def train(self, epochs=1000):
        """
        Train the model
        Args:
            epochs: number of training epochs
        """
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Generate collocation points
            x, t = self.data_generator.generate_collocation_points()
            x = x.to(self.device)
            t = t.to(self.device)
            
            # Compute loss and update model
            loss = self.compute_loss(x, t)
            loss.backward()
            self.optimizer.step()
            
            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    def evaluate(self, x, t, u_true=None):
        """
        Evaluate the model on given points
        Args:
            x: spatial coordinate
            t: temporal coordinate
            u_true: true solution (if available)
        Returns:
            u_pred: predicted solution
            loss: evaluation loss
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            t = t.to(self.device)
            u_pred = self.model.predict(x, t)
            loss = self.compute_loss(x, t, u_true) if u_true is not None else None
        return u_pred, loss

class Trainer:
    """
    Base trainer class for Physics-Informed Neural Networks.
    This class provides common functionality for all equation trainers.
    """
    
    def __init__(self):
        """Initialize the base trainer."""
        self.model = None
        self.equation = None
        self.loss_history = []
        
    def train(self, epochs=1000, lr=0.001):
        """
        Base training method to be overridden by specific equation trainers.
        
        Args:
            epochs (int): Number of training epochs
            lr (float): Learning rate
        """
        raise NotImplementedError("Train method must be implemented by child classes")
        
    def _save_results(self):
        """Save the trained model and generate plots."""
        raise NotImplementedError("Save results method must be implemented by child classes")
        
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
        raise NotImplementedError("Plot solution method must be implemented by child classes")

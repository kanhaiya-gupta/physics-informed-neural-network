import torch
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

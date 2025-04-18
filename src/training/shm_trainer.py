import torch
import torch.optim as optim
from src.models.shm_pinn import SHMPINN
from src.equations.shm_equation import SHMEquation
from src.data.generators.shm_data import SHMDataGenerator
import os
import numpy as np
import time

class SHMTrainer:
    """Trainer for Simple Harmonic Motion PINN."""
    
    def __init__(self):
        """Initialize the SHM equation trainer."""
        self.model = SHMPINN()
        self.equation = SHMEquation()
        self.data_generator = SHMDataGenerator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = None  # Will be initialized in train method
        self.loss_history = []
        self.log_progress = None  # Custom logging function
        
        # Create necessary directories
        self.data_dir = "data/shm"
        self.results_dir = "results/shm"
        self.models_dir = os.path.join(self.results_dir, "models")
        self.plots_dir = os.path.join(self.results_dir, "plots")
        self.metrics_dir = os.path.join(self.results_dir, "metrics")
        
        for dir_path in [self.data_dir, self.results_dir, self.models_dir, self.plots_dir, self.metrics_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def log_progress(self, epoch: int, loss: float):
        """Callback for logging training progress."""
        print(f"Epoch {epoch}: Loss = {loss:.6f}")
        
    def save_loss_history(self, epoch: int, loss: float):
        """Save loss history to file."""
        if not hasattr(self, '_loss_history'):
            self._loss_history = []
        self._loss_history.append((epoch, loss))
        np.save("results/shm/metrics/loss_history.npy", np.array(self._loss_history))
        
    def train(self, epochs=1000, lr=0.001, batch_size=32):
        """Train the model."""
        start_time = time.time()
        
        # Initialize optimizer with the provided learning rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Generate training data and ensure it requires gradients
        t = torch.linspace(0, 1, batch_size, requires_grad=True).reshape(-1, 1).to(self.device)
        
        # Save generated data
        torch.save(t, os.path.join(self.data_dir, "t.pt"))
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            u_pred = self.model(t)
            
            # Calculate losses
            pde_loss = self.equation.compute_pde_loss(t, u_pred)
            ic_loss = self.equation.compute_ic_loss(t, u_pred)
            
            # Total loss
            loss = pde_loss + ic_loss
            
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
        save_dir = os.path.join("results", "shm", "models")
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
            't': torch.load(os.path.join(self.data_dir, "t.pt")),
            'loss_history': self.loss_history
        }, data_path)
        
        # Generate and save plots
        self._plot_loss_curve()
        self._plot_solution()
        self._plot_comparison()
        
    def generate_plots(self):
        """Generate and save plots for visualization."""
        import matplotlib.pyplot as plt
        
        # Load the best model
        self.model.load_state_dict(torch.load("results/shm/models/model.pth"))
        self.model.eval()
        
        # Generate data for plotting
        t_plot = self.data_generator.generate_time_points(n_points=200)
        t_tensor = torch.tensor(t_plot, dtype=torch.float32).to(self.device).view(-1, 1)
        
        with torch.no_grad():
            x_pred = self.model(t_tensor).cpu().numpy()
        
        x_exact = self.data_generator.generate_exact_solution(t_plot)
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.plot(t_plot, x_pred, 'b-', label='PINN Prediction')
        plt.plot(t_plot, x_exact, 'r--', label='Exact Solution')
        plt.xlabel('Time')
        plt.ylabel('Displacement')
        plt.title('SHM: PINN vs Exact Solution')
        plt.legend()
        plt.grid(True)
        plt.savefig("results/shm/plots/solution_comparison.png")
        plt.close()
        
        # Plot loss history
        loss_history = np.load("results/shm/metrics/loss_history.npy")
        plt.figure(figsize=(10, 6))
        plt.semilogy(loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.grid(True)
        plt.savefig("results/shm/plots/loss_curve.png")
        plt.close() 
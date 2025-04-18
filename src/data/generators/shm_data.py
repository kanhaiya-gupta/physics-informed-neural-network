import numpy as np
import os

class SHMDataGenerator:
    """Data generator for Simple Harmonic Motion."""
    
    def __init__(self, omega=1.0, t_min=0.0, t_max=2*np.pi):
        """
        Initialize the SHM data generator.
        
        Args:
            omega (float): Angular frequency
            t_min (float): Minimum time
            t_max (float): Maximum time
        """
        self.omega = omega
        self.t_min = t_min
        self.t_max = t_max
    
    def generate_time_points(self, n_points=1000):
        """
        Generate time points for training.
        
        Args:
            n_points (int): Number of time points to generate
            
        Returns:
            numpy.ndarray: Array of time points
        """
        return np.linspace(self.t_min, self.t_max, n_points)
    
    def generate_exact_solution(self, t):
        """
        Generate exact solution for validation.
        
        Args:
            t (numpy.ndarray): Time points
            
        Returns:
            numpy.ndarray: Exact solution values
        """
        return np.cos(self.omega * t)  # Solution for x(0) = 1, v(0) = 0
    
    def generate_training_data(self, n_points=1000):
        """
        Generate training data including time points and exact solutions.
        
        Args:
            n_points (int): Number of points to generate
            
        Returns:
            tuple: (time points, exact solutions)
        """
        t = self.generate_time_points(n_points)
        x = self.generate_exact_solution(t)
        return t, x 
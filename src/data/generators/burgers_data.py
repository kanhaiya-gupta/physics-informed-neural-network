import torch
from .data_generator import BaseDataGenerator

class BurgersDataGenerator(BaseDataGenerator):
    """
    Data generator for Burgers' equation
    """
    def __init__(self, x_range=(-1, 1), t_range=(0, 1), num_points=1000):
        """
        Initialize Burgers' equation data generator
        Args:
            x_range: tuple of (min, max) for spatial domain
            t_range: tuple of (min, max) for temporal domain
            num_points: number of collocation points to generate
        """
        super().__init__(x_range, t_range, num_points)

    def generate_collocation_points(self):
        """
        Generate collocation points for Burgers' equation
        Returns:
            x: spatial coordinates
            t: temporal coordinates
        """
        # Generate random points in the domain
        x = torch.rand(self.num_points, 1) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        t = torch.rand(self.num_points, 1) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
        
        # Add more points near the shock region (x=0) for better resolution
        num_shock_points = self.num_points // 4
        x_shock = torch.rand(num_shock_points, 1) * 0.2 - 0.1  # Concentrate around x=0
        t_shock = torch.rand(num_shock_points, 1) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
        
        # Combine points
        x = torch.cat([x, x_shock])
        t = torch.cat([t, t_shock])
        
        return x.requires_grad_(True), t.requires_grad_(True)

    def generate_initial_condition(self, num_points=100):
        """
        Generate points for initial condition
        Args:
            num_points: number of points to generate
        Returns:
            x: spatial coordinates
            t: temporal coordinates
            u: initial condition values
        """
        x = torch.linspace(self.x_range[0], self.x_range[1], num_points).reshape(-1, 1)
        t = torch.ones(num_points, 1) * self.t_range[0]
        
        # Initial condition: u(x,0) = -sin(pi*x)
        u = -torch.sin(torch.pi * x)
        
        return x.requires_grad_(True), t.requires_grad_(True), u

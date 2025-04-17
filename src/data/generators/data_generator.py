import torch
from abc import ABC, abstractmethod

class BaseDataGenerator(ABC):
    """
    Base class for generating data points for PINN training
    All specific data generators should inherit from this class
    """
    def __init__(self, x_range=(0, 1), t_range=(0, 1), num_points=1000):
        """
        Initialize the data generator
        Args:
            x_range: tuple of (min, max) for spatial domain
            t_range: tuple of (min, max) for temporal domain
            num_points: number of collocation points to generate
        """
        self.x_range = x_range
        self.t_range = t_range
        self.num_points = num_points

    @abstractmethod
    def generate_collocation_points(self):
        """
        Generate collocation points for training
        Returns:
            x: spatial coordinates
            t: temporal coordinates
        """
        pass

    def generate_boundary_points(self, num_points=100):
        """
        Generate points on the boundary of the domain
        Args:
            num_points: number of points to generate per boundary
        Returns:
            x: spatial coordinates
            t: temporal coordinates
        """
        # Left boundary (x = x_min)
        x_left = torch.ones(num_points, 1) * self.x_range[0]
        t_left = torch.rand(num_points, 1) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]

        # Right boundary (x = x_max)
        x_right = torch.ones(num_points, 1) * self.x_range[1]
        t_right = torch.rand(num_points, 1) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]

        # Initial condition (t = t_min)
        x_initial = torch.rand(num_points, 1) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        t_initial = torch.ones(num_points, 1) * self.t_range[0]

        # Combine all boundary points
        x = torch.cat([x_left, x_right, x_initial])
        t = torch.cat([t_left, t_right, t_initial])

        return x.requires_grad_(True), t.requires_grad_(True)

import torch

class HeatDataGenerator:
    def __init__(self, x_range=(0, 1), t_range=(0, 1), num_points=1000):
        self.x_range = x_range
        self.t_range = t_range
        self.num_points = num_points

    def generate_collocation_points(self):
        x = torch.rand(self.num_points, 1) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        t = torch.rand(self.num_points, 1) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
        return x.requires_grad_(True), t.requires_grad_(True)
import torch

def heat_boundary_condition(x, t):
    # Example: u(0, t) = u(1, t) = 0
    return torch.zeros_like(t)
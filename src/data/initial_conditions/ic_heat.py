import torch

def heat_initial_condition(x):
    # Example: u(x, 0) = sin(pi * x)
    return torch.sin(torch.pi * x)
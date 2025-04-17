import torch

def burgers_initial_condition(x, t, u):
    """
    Compute initial condition loss for Burgers' equation
    Args:
        x: spatial coordinate
        t: temporal coordinate
        u: predicted solution
    Returns:
        loss: initial condition loss
    """
    # Initial condition: u(x,0) = -sin(pi*x)
    mask = (t == 0)  # Points at t=0
    if mask.any():
        u_initial = -torch.sin(torch.pi * x[mask])
        loss = torch.mean((u[mask] - u_initial)**2)
    else:
        loss = torch.tensor(0.0, device=x.device)
    return loss

import torch

def wave_initial_condition(x, t, u):
    """
    Compute initial condition loss for wave equation
    Args:
        x: spatial coordinate
        t: temporal coordinate
        u: predicted solution
    Returns:
        loss: initial condition loss
    """
    # Initial condition: u(x,0) = sin(pi*x)
    mask = (t == 0)  # Points at t=0
    if mask.any():
        u_initial = torch.sin(torch.pi * x[mask])
        loss = torch.mean((u[mask] - u_initial)**2)
    else:
        loss = torch.tensor(0.0, device=x.device)
    return loss

def wave_initial_velocity(x, t, u_t):
    """
    Compute initial velocity condition loss for wave equation
    Args:
        x: spatial coordinate
        t: temporal coordinate
        u_t: predicted time derivative
    Returns:
        loss: initial velocity condition loss
    """
    # Initial velocity: u_t(x,0) = 0
    mask = (t == 0)  # Points at t=0
    if mask.any():
        loss = torch.mean(u_t[mask]**2)
    else:
        loss = torch.tensor(0.0, device=x.device)
    return loss

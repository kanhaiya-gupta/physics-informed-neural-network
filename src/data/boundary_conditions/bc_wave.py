import torch

def wave_boundary_condition(x, t, u):
    """
    Compute boundary condition loss for wave equation
    Args:
        x: spatial coordinate
        t: temporal coordinate
        u: predicted solution
    Returns:
        loss: boundary condition loss
    """
    # Boundary conditions: u(0,t) = u(1,t) = 0
    mask = (x == 0) | (x == 1)  # Points at boundaries
    if mask.any():
        loss = torch.mean(u[mask]**2)
    else:
        loss = torch.tensor(0.0, device=x.device)
    return loss

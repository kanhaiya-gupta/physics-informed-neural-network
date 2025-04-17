import torch

def burgers_boundary_condition(x, t, u):
    """
    Compute boundary condition loss for Burgers' equation
    Args:
        x: spatial coordinate
        t: temporal coordinate
        u: predicted solution
    Returns:
        loss: boundary condition loss
    """
    # Boundary conditions: u(-1,t) = u(1,t) = 0
    mask = (x == -1) | (x == 1)  # Points at boundaries
    if mask.any():
        loss = torch.mean(u[mask]**2)
    else:
        loss = torch.tensor(0.0, device=x.device)
    return loss

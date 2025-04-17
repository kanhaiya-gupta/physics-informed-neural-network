import torch

class HeatEquation:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Thermal diffusivity

    def pde_residual(self, x, t, u, u_x, u_xx, u_t):
        # Heat equation: u_t = alpha * u_xx
        return u_t - self.alpha * u_xx
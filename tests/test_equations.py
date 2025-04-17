import pytest
import torch
import numpy as np
from src.equations.heat_equation import HeatEquation
from src.equations.burgers_equation import BurgersEquation
from src.equations.wave_equation import WaveEquation
from src.equations.shm_equation import SHMEquation
from src.models.shm_pinn import SHMPINN

def calculate_derivatives(u, x, t):
    """Helper function to calculate derivatives"""
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    return u_x, u_xx, u_t

def test_heat_equation():
    equation = HeatEquation(alpha=0.1)
    x = torch.rand(10, 1, requires_grad=True)
    t = torch.rand(10, 1, requires_grad=True)
    u = torch.sin(x) * torch.exp(-t)  # A simple solution
    u.requires_grad_(True)
    
    # Calculate derivatives
    u_x, u_xx, u_t = calculate_derivatives(u, x, t)
    
    # Test PDE residual
    residual = equation.pde_residual(x, t, u, u_x, u_xx, u_t)
    assert residual.shape == (10, 1)
    assert not torch.isnan(residual).any()
    assert not torch.isinf(residual).any()

def test_burgers_equation():
    equation = BurgersEquation(nu=0.1)
    x = torch.rand(10, 1, requires_grad=True)
    t = torch.rand(10, 1, requires_grad=True)
    u = torch.sin(x) * torch.exp(-t)  # A simple solution
    u.requires_grad_(True)
    
    # Calculate derivatives
    u_x, u_xx, u_t = calculate_derivatives(u, x, t)
    
    # Test PDE residual
    residual = equation.pde_residual(x, t, u, u_x, u_xx, u_t)
    assert residual.shape == (10, 1)
    assert not torch.isnan(residual).any()
    assert not torch.isinf(residual).any()

def test_wave_equation():
    equation = WaveEquation(c=1.0)
    x = torch.rand(10, 1, requires_grad=True)
    t = torch.rand(10, 1, requires_grad=True)
    u = torch.sin(x) * torch.cos(t)  # A simple solution
    u.requires_grad_(True)
    
    # Calculate derivatives
    u_x, u_xx, u_t = calculate_derivatives(u, x, t)
    
    # Test PDE residual
    residual = equation.pde_residual(x, t, u, u_x, u_xx, u_t)
    assert residual.shape == (10, 1)
    assert not torch.isnan(residual).any()
    assert not torch.isinf(residual).any()

def test_shm_equation():
    equation = SHMEquation(omega=1.0)
    t = torch.rand(10, 1, requires_grad=True)
    model = SHMPINN()  # Create a model instance
    
    # Test compute_residual
    residual = equation.compute_residual(None, t, model)
    assert residual.shape == (10, 1)
    assert not torch.isnan(residual).any()
    assert not torch.isinf(residual).any()
    
    # Test exact solution
    exact = equation.exact_solution(t)
    assert exact.shape == (10, 1)
    assert not torch.isnan(exact).any()
    assert not torch.isinf(exact).any()

import pytest
import torch
from src.models.heat_pinn import HeatPINN
from src.models.burgers_pinn import BurgersPINN
from src.models.wave_pinn import WavePINN
from src.models.shm_pinn import SHMPINN

def test_heat_pinn_forward():
    model = HeatPINN()
    x = torch.rand(10, 1, requires_grad=True)
    t = torch.rand(10, 1, requires_grad=True)
    output = model.predict(x, t)
    assert output.shape == (10, 1)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_burgers_pinn_forward():
    model = BurgersPINN()
    x = torch.rand(10, 1, requires_grad=True)
    t = torch.rand(10, 1, requires_grad=True)
    output = model.predict(x, t)
    assert output.shape == (10, 1)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_wave_pinn_forward():
    model = WavePINN()
    x = torch.rand(10, 1, requires_grad=True)
    t = torch.rand(10, 1, requires_grad=True)
    output = model.predict(x, t)
    assert output.shape == (10, 1)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_shm_pinn_forward():
    model = SHMPINN()
    t = torch.rand(10, 1, requires_grad=True)
    output = model.forward(t)
    assert output.shape == (10, 1)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_heat_pinn_gradients():
    model = HeatPINN()
    x = torch.rand(10, 1, requires_grad=True)
    t = torch.rand(10, 1, requires_grad=True)
    output = model.predict(x, t)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert t.grad is not None

def test_burgers_pinn_gradients():
    model = BurgersPINN()
    x = torch.rand(10, 1, requires_grad=True)
    t = torch.rand(10, 1, requires_grad=True)
    output = model.predict(x, t)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert t.grad is not None

def test_wave_pinn_gradients():
    model = WavePINN()
    x = torch.rand(10, 1, requires_grad=True)
    t = torch.rand(10, 1, requires_grad=True)
    output = model.predict(x, t)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert t.grad is not None

def test_shm_pinn_gradients():
    model = SHMPINN()
    t = torch.rand(10, 1, requires_grad=True)
    output = model.forward(t)
    loss = output.sum()
    loss.backward()
    assert t.grad is not None
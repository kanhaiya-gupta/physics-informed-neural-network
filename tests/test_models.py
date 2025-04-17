import pytest
import torch
from src.models.heat_pinn import HeatPINN

def test_heat_pinn_forward():
    model = HeatPINN()
    x = torch.rand(10, 1, requires_grad=True)
    t = torch.rand(10, 1, requires_grad=True)
    output = model.predict(x, t)
    assert output.shape == (10, 1)
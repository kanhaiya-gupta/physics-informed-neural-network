from src.models.heat_pinn import HeatPINN
from fastapi import Depends

def get_heat_model():
    return HeatPINN()
from src.models.heat_pinn import HeatPINN
import torch

class HeatEvaluator:
    def __init__(self, model: HeatPINN):
        self.model = model

    def evaluate(self, x, t):
        with torch.no_grad():
            return self.model.predict(x, t)
from .base_pinn import BasePINN
import torch

class HeatPINN(BasePINN):
    def __init__(self):
        super(HeatPINN, self).__init__(layers=[2, 20, 20, 1])  # Input: (x, t), Output: u

    def predict(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.forward(inputs)
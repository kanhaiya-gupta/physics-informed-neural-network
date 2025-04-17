import torch
import torch.nn as nn
from .base_pinn import BasePINN

class SHMPINN(BasePINN):
    """
    Physics-Informed Neural Network for Simple Harmonic Motion.
    """
    
    def __init__(self, layers=None):
        """
        Initialize the SHM PINN model.
        
        Args:
            layers (list): List of layer sizes. Default is [1, 20, 20, 20, 1]
        """
        if layers is None:
            layers = [1, 20, 20, 20, 1]
        super().__init__(layers)
        
    def forward(self, t):
        """
        Forward pass of the model.
        
        Args:
            t (torch.Tensor): Time coordinates
            
        Returns:
            torch.Tensor: Predicted displacement
        """
        # Ensure t has the correct shape
        if t.dim() == 1:
            t = t.view(-1, 1)
            
        return super().forward(t) 
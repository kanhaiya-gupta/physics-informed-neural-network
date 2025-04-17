import torch
from src.models.heat_pinn import HeatPINN
from src.equations.heat_equation import HeatEquation
from src.data.generators.heat_data import HeatDataGenerator
from src.data.initial_conditions.ic_heat import heat_initial_condition
from src.data.boundary_conditions.bc_heat import heat_boundary_condition

class HeatTrainer:
    def __init__(self):
        self.model = HeatPINN()
        self.equation = HeatEquation()
        self.data_generator = HeatDataGenerator()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, epochs=1000, lr=0.001):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            x, t = self.data_generator.generate_collocation_points()
            u = self.model(x, t)
            u_x = torch.autograd.grad(u, x, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x, create_graph=True)[0]
            u_t = torch.autograd.grad(u, t, create_graph=True)[0]
            pde_loss = self.equation.pde_residual(x, t, u, u_x, u_xx, u_t).pow(2).mean()
            # Placeholder: Add initial and boundary condition losses
            loss = pde_loss
            loss.backward()
            self.optimizer.step()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
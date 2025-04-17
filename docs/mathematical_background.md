# Mathematical Background

## Overview
This document provides a detailed explanation of the mathematical background and equations implemented in the project.

## Partial Differential Equations (PDEs)

### Heat Equation
The heat equation describes the distribution of heat in a given region over time:
```
∂u/∂t = α ∂²u/∂x²
```
where:
- u(x,t) is the temperature at position x and time t
- α is the thermal diffusivity
- ∂u/∂t is the time derivative
- ∂²u/∂x² is the second spatial derivative

#### Physical Interpretation
The heat equation states that the rate of change of temperature at a point is proportional to the curvature of the temperature profile at that point. This means that heat flows from regions of high temperature to regions of low temperature.

#### Initial and Boundary Conditions
- Initial condition: u(x,0) = f(x)
- Boundary conditions: u(0,t) = g₁(t), u(L,t) = g₂(t)

### Wave Equation
The wave equation describes the propagation of waves:
```
∂²u/∂t² = c² ∂²u/∂x²
```
where:
- u(x,t) is the wave amplitude at position x and time t
- c is the wave speed
- ∂²u/∂t² is the second time derivative
- ∂²u/∂x² is the second spatial derivative

#### Physical Interpretation
The wave equation states that the acceleration of the wave at a point is proportional to the curvature of the wave at that point. This means that waves propagate through space and time.

#### Initial and Boundary Conditions
- Initial condition: u(x,0) = f(x), ∂u/∂t(x,0) = g(x)
- Boundary conditions: u(0,t) = h₁(t), u(L,t) = h₂(t)

### Burgers' Equation
Burgers' equation describes the evolution of a viscous fluid:
```
∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
```
where:
- u(x,t) is the fluid velocity at position x and time t
- ν is the viscosity coefficient
- ∂u/∂t is the time derivative
- u ∂u/∂x is the nonlinear advection term
- ν ∂²u/∂x² is the diffusion term

#### Physical Interpretation
Burgers' equation combines nonlinear advection and diffusion. The advection term u ∂u/∂x represents the transport of momentum by the fluid flow, while the diffusion term ν ∂²u/∂x² represents the dissipation of momentum due to viscosity.

#### Initial and Boundary Conditions
- Initial condition: u(x,0) = f(x)
- Boundary conditions: u(0,t) = g₁(t), u(L,t) = g₂(t)

## Physics-Informed Neural Networks (PINNs)

### Overview
PINNs are neural networks that are trained to solve PDEs by incorporating the physical laws described by the equations into the loss function.

### Architecture
A typical PINN consists of:
1. Input layer: Takes spatial and temporal coordinates (x,t)
2. Hidden layers: Multiple fully connected layers with activation functions
3. Output layer: Predicts the solution u(x,t)

### Loss Function
The loss function for a PINN typically includes:
1. PDE loss: Measures how well the predicted solution satisfies the PDE
2. Initial condition loss: Measures how well the predicted solution matches the initial condition
3. Boundary condition loss: Measures how well the predicted solution matches the boundary conditions

### Training
PINNs are trained using gradient descent to minimize the total loss, which is a weighted sum of the PDE loss, initial condition loss, and boundary condition loss.

## Implementation Details

### Heat Equation Implementation
- The heat equation is implemented in `src/equations/heat_equation.py`
- The PINN model for the heat equation is implemented in `src/models/heat_pinn.py`
- The training logic for the heat equation is implemented in `src/training/heat_trainer.py`

### Wave Equation Implementation
- The wave equation is implemented in `src/equations/wave_equation.py`
- The PINN model for the wave equation is implemented in `src/models/wave_pinn.py`
- The training logic for the wave equation is implemented in `src/training/wave_trainer.py`

### Burgers' Equation Implementation
- Burgers' equation is implemented in `src/equations/burgers_equation.py`
- The PINN model for Burgers' equation is implemented in `src/models/burgers_pinn.py`
- The training logic for Burgers' equation is implemented in `src/training/burgers_trainer.py` 
# Physics-Informed Neural Network (PINN) Project

This project implements a modular framework for solving partial differential equations (PDEs) using Physics-Informed Neural Networks (PINNs). It supports multiple PDEs (e.g., heat, wave, Burgers' equations) and includes a FastAPI server for exposing PINN functionality via RESTful APIs. The framework is designed for extensibility, with separate modules for models, equations, data generation, training, evaluation, and hyperparameter tuning.

## Mathematical Background

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

## Features
- **Modular PINN Implementation**: Supports multiple PDEs with reusable base classes.
- **FastAPI Integration**: RESTful API for training and predicting with PINN models.
- **Hyperparameter Tuning**: Scripts for optimizing model parameters using Optuna.
- **Extensible Structure**: Easily add new PDEs by extending existing modules.
- **Comprehensive Testing**: Unit tests for all major components.
- **Data Generation**: Flexible data generation for training and validation.
- **Visualization Tools**: Tools for plotting and analyzing results.
- **Configuration Management**: YAML-based configuration for easy parameter tuning.

## Project Structure 
The project structure is as follows (for reference):
```
physics_informed_neural_network/
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── endpoints/
│   │   │   ├── __init__.py
│   │   │   ├── heat.py
│   │   │   ├── wave.py
│   │   │   └── burgers.py
│   │   └── router.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── dependencies.py
│   └── schemas/
│       ├── __init__.py
│       ├── heat.py
│       ├── wave.py
│       └── burgers.py
├── src/
│   ├── models/
│   │   ├── base_pinn.py
│   │   ├── heat_pinn.py
│   │   ├── wave_pinn.py
│   │   └── burgers_pinn.py
│   ├── equations/
│   │   ├── base_equation.py
│   │   ├── heat_equation.py
│   │   ├── wave_equation.py
│   │   └── burgers_equation.py
│   ├── data/
│   │   ├── generators/
│   │   │   ├── data_generator.py
│   │   │   ├── heat_data.py
│   │   │   ├── wave_data.py
│   │   │   └── burgers_data.py
│   │   ├── initial_conditions/
│   │   │   ├── ic_heat.py
│   │   │   ├── ic_wave.py
│   │   │   └── ic_burgers.py
│   │   └── boundary_conditions/
│   │       ├── bc_heat.py
│   │       ├── bc_wave.py
│   │       └── bc_burgers.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── heat_trainer.py
│   │   ├── wave_trainer.py
│   │   └── burgers_trainer.py
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   ├── heat_evaluator.py
│   │   ├── wave_evaluator.py
│   │   └── burgers_evaluator.py
│   └── utils/
│       ├── logger.py
│       ├── visualization.py
│       ├── metrics.py
│       └── config_parser.py
├── configs/
│   ├── base_config.yaml
│   ├── equations/
│   │   ├── heat_equation.yaml
│   │   ├── wave_equation.yaml
│   │   └── burgers_equation.yaml
├── scripts/
│   ├── hyperparameter_tuning/
│   │   ├── tune_heat.py
│   │   ├── tune_wave.py
│   │   ├── tune_burgers.py
│   │   └── tuning_utils.py
│   ├── preprocess_data.py
│   ├── run_batch_experiments.py
│   └── analyze_results.py
├── results/
│   ├── heat/
│   │   ├── models/
│   │   ├── plots/
│   │   ├── metrics/
│   │   └── tuning/
│   ├── wave/
│   │   ├── models/
│   │   ├── plots/
│   │   ├── metrics/
│   │   └── tuning/
│   └── burgers/
│       ├── models/
│       ├── plots/
│       ├── metrics/
│       └── tuning/
├── tests/
│   ├── test_models/
│   ├── test_equations/
│   ├── test_data/
│   ├── test_training/
│   └── test_evaluation/
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/physics-informed-neural-network.git
cd physics-informed-neural-network
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the FastAPI Server
```bash
uvicorn app.main:app --reload
```

### Training a Model
```python
from src.training.heat_trainer import HeatTrainer

trainer = HeatTrainer()
trainer.train(epochs=1000, lr=0.001)
```

### Making Predictions
```python
from src.models.heat_pinn import HeatPINN

model = HeatPINN()
prediction = model.predict(x, t)
```

### Running Hyperparameter Tuning
```bash
python scripts/hyperparameter_tuning/tune_heat.py
```

### Analyzing Results
```bash
python scripts/analyze_results.py
```

## API Documentation

### Heat Equation
- **POST /heat/train**: Train a heat equation PINN model
  - Parameters: epochs, learning_rate
  - Returns: Training status and epochs

- **POST /heat/predict**: Make predictions with a trained heat equation PINN model
  - Parameters: x (spatial coordinates), t (time coordinates)
  - Returns: Predicted solution values

### Wave Equation
- **POST /wave/train**: Train a wave equation PINN model
  - Parameters: epochs, learning_rate, c (wave speed)
  - Returns: Training status and epochs

- **POST /wave/predict**: Make predictions with a trained wave equation PINN model
  - Parameters: x (spatial coordinates), t (time coordinates)
  - Returns: Predicted solution values

### Burgers' Equation
- **POST /burgers/train**: Train a Burgers' equation PINN model
  - Parameters: epochs, learning_rate, nu (viscosity coefficient)
  - Returns: Training status and epochs

- **POST /burgers/predict**: Make predictions with a trained Burgers' equation PINN model
  - Parameters: x (spatial coordinates), t (time coordinates)
  - Returns: Predicted solution values

## Examples

### Heat Equation Example
```python
from src.training.heat_trainer import HeatTrainer
from src.models.heat_pinn import HeatPINN

# Train the model
trainer = HeatTrainer()
trainer.train(epochs=1000, lr=0.001)

# Make predictions
model = HeatPINN()
x = torch.linspace(0, 1, 100)
t = torch.linspace(0, 1, 100)
prediction = model.predict(x, t)
```

### Wave Equation Example
```python
from src.training.wave_trainer import WaveTrainer
from src.models.wave_pinn import WavePINN

# Train the model
trainer = WaveTrainer(c=1.0)
trainer.train(epochs=1000, lr=0.001)

# Make predictions
model = WavePINN()
x = torch.linspace(0, 1, 100)
t = torch.linspace(0, 1, 100)
prediction = model.predict(x, t)
```

### Burgers' Equation Example
```python
from src.training.burgers_trainer import BurgersTrainer
from src.models.burgers_pinn import BurgersPINN

# Train the model
trainer = BurgersTrainer(nu=0.01)
trainer.train(epochs=1000, lr=0.001)

# Make predictions
model = BurgersPINN()
x = torch.linspace(-1, 1, 100)
t = torch.linspace(0, 1, 100)
prediction = model.predict(x, t)
```

## Contributing

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests: `pytest`
5. Commit your changes: `git commit -m 'Add some feature'`
6. Push to the branch: `git push origin feature/your-feature-name`
7. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

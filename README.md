# Physics-Informed Neural Network (PINN) Framework

A comprehensive framework for solving partial differential equations (PDEs) using Physics-Informed Neural Networks (PINNs). This project implements PINNs for various physical systems including simple harmonic motion, heat transfer, wave propagation, and fluid dynamics (Burgers' equation). The framework provides a modular architecture for training neural networks that respect physical laws, with RESTful APIs for model training and prediction.

## Features

- **Modular PINN Implementation**: Supports multiple equations with reusable base classes
- **FastAPI Integration**: RESTful API for training and predicting with PINN models
- **Hyperparameter Tuning**: Scripts for optimizing model parameters using Optuna
- **Extensible Structure**: Easily add new equations by extending existing modules
- **Comprehensive Testing**: Unit tests for all major components
- **Data Generation**: Flexible data generation for training and validation
- **Visualization Tools**: Tools for plotting and analyzing results
- **Configuration Management**: YAML-based configuration for easy parameter tuning
- **Structured Logging**: Comprehensive logging system for API interactions and training progress

## Project Structure

```
physics_informed_neural_network/
├── app/                    # FastAPI application
│   ├── api/               # API endpoints
│   ├── core/              # Core application logic
│   └── schemas/           # API request/response schemas
├── src/                   # Source code
│   ├── models/           # Neural network models
│   ├── equations/        # PDE implementations
│   ├── data/            # Data generation
│   ├── training/        # Training implementations
│   ├── evaluation/      # Model evaluation
│   └── utils/          # Utility functions
├── data/                # Generated data
├── results/            # Training results
├── logs/              # Application logs
├── configs/           # Configuration files
├── scripts/          # Utility scripts
├── tests/           # Test suite
├── docs/           # Documentation
├── .github/        # GitHub workflows
├── main.py         # Application entry point
├── test_batch.py   # Batch testing script
├── setup.py        # Package setup
├── requirements.txt # Python dependencies
├── setup_project.sh # Project setup script (Linux/Mac)
├── activate_ml_env.sh # Environment activation (Linux/Mac)
└── activate_ml_env.ps1 # Environment activation (Windows)
```

## Data Organization

Each equation has its own data directory structure:

```
data/{equation}/
├── x.pt                    # Spatial coordinates
├── t.pt                    # Temporal coordinates
└── training_data.pt        # Complete training data
```

Results are stored in:

```
results/{equation}/
├── models/                 # Trained model weights
├── plots/                  # Generated plots
└── metrics/                # Training metrics
```

## Mathematical Background

### Simple Harmonic Motion
The simple harmonic motion equation describes the motion of a mass on a spring:
```
d²x/dt² + ω²x = 0
```
where:
- x(t) is the displacement at time t
- ω is the angular frequency
- d²x/dt² is the second time derivative

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

## API Endpoints

The framework provides the following RESTful API endpoints:

### Batch Training
- **Endpoint**: `/batch/train`
- **Method**: POST
- **Request Body**:
  ```json
  {
    "equations": ["shm", "heat", "wave", "burgers"],
    "config": {
      "epochs": 1000,
      "learning_rate": 0.001,
      "batch_size": 32
    }
  }
  ```
- **Response**:
  ```json
  {
    "status": "completed",
    "results": {
      "equation_name": {
        "status": "success",
        "final_loss": float,
        "training_time": float
      }
    }
  }
  ```

### Batch Prediction
- **Endpoint**: `/batch/predict`
- **Method**: POST
- **Request Body**:
  ```json
  {
    "equations": ["shm", "heat", "wave", "burgers"],
    "points": {
      "equation_name": [0.0, 0.1, 0.2, 0.3]
    }
  }
  ```
- **Response**:
  ```json
  {
    "status": "completed",
    "results": {
      "equation_name": {
        "status": "success",
        "predictions": [float]
      }
    }
  }
  ```

## Logging System

The framework includes a comprehensive logging system that tracks various aspects of the application:

### Log Directory Structure
```
logs/
├── api/                    # API-related logs
│   ├── access.log         # API access logs
│   ├── error.log          # API error logs
│   └── batch.log          # Batch operation logs
├── training/              # Training-related logs
│   ├── shm/              # SHM training logs
│   ├── heat/             # Heat equation logs
│   ├── wave/             # Wave equation logs
│   └── burgers/          # Burgers' equation logs
└── system/               # System-level logs
```

### Log Configuration
- Maximum file size: 10MB
- Backup files: 5
- Automatic rotation
- Timestamp format: `%Y-%m-%d %H:%M:%S`
- Log level: INFO for normal operations, ERROR for errors

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the FastAPI server:
   ```bash
   python main.py
   ```
4. Access the API at `http://localhost:8000`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

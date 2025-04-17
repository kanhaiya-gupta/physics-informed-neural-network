# Physics-Informed Neural Network (PINN) Project

A modular PINN project with FastAPI support for solving differential equations (e.g., heat, wave, Burgers' equations).

---

### Project Structure 
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
│   ├── test_data_generation.py
│   ├── test_models.py
│   ├── test_equations.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   ├── test_api.py
│   ├── test_scripts.py
│   └── test_utils.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

### File Contents

#### Top-Level Files

```python
from fastapi import FastAPI
from app.api.router import api_router

app = FastAPI(title="PINN API", description="API for Physics-Informed Neural Networks")
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

```
plain
fastapi
uvicorn
pydantic
torch
numpy
matplotlib
pyyaml
pytest
optuna
```

<xaiArtifact artifact_id="ea90b771-6caf-4407-a5df-d5fc797688ae" artifact_version_id="802e8703-7ddd-4bce-ae6a-40903fa94266" title="README.md" contentType="text/markdown">
# Physics-Informed Neural Network (PINN) Project

This project implements a modular framework for solving partial differential equations (PDEs) using Physics-Informed Neural Networks (PINNs). It supports multiple PDEs (e.g., heat, wave, Burgers' equations) and includes a FastAPI server for exposing PINN functionality via RESTful APIs. The framework is designed for extensibility, with separate modules for models, equations, data generation, training, evaluation, and hyperparameter tuning.

## Features
- **Modular PINN Implementation**: Supports multiple PDEs with reusable base classes.
- **FastAPI Integration**: RESTful API for training and predicting with PINN models.
- **Hyperparameter Tuning**: Scripts for optimizing model parameters using Optuna.
- **Extensible Structure**: Easily add new PDEs by extending existing modules.
- **Comprehensive Testing**: Unit tests for all major components.

## Project Structure

### Top-Level Files
- `main.py`: Entry point for the FastAPI server.
- `requirements.txt`: Lists project dependencies.
- `README.md`: Project documentation (this file).
- `.gitignore`: Ignores unnecessary files (e.g., `__pycache__`, `results/`).

### `app/` Directory
FastAPI application logic for exposing PINN functionality via APIs.
- `app/__init__.py`: Marks `app` as a Python package.
- `app/api/router.py`: Aggregates API endpoints for all PDEs.
- `app/api/endpoints/heat.py`: API endpoints for the heat equation (training, prediction).
- `app/api/endpoints/wave.py`: API endpoints for the wave equation.
- `app/api/endpoints/burgers.py`: API endpoints for the Burgers' equation.
- `app/core/config.py`: Configures FastAPI settings (e.g., CORS).
- `app/core/dependencies.py`: Manages dependency injection for models and trainers.
- `app/schemas/heat.py`: Pydantic schemas for heat equation API requests/responses.
- `app/schemas/wave.py`: Pydantic schemas for wave equation API requests/responses.
- `app/schemas/burgers.py`: Pydantic schemas for Burgers' equation API requests/responses.

### `src/` Directory
Core PINN implementation, including models, equations, data generation, training, and utilities.
- `src/models/base_pinn.py`: Base class for PINN neural networks.
- `src/models/heat_pinn.py`: PINN model for the heat equation.
- `src/models/wave_pinn.py`: PINN model for the wave equation.
- `src/models/burgers_pinn.py`: PINN model for the Burgers' equation.
- `src/equations/base_equation.py`: Abstract base class for PDEs.
- `src/equations/heat_equation.py`: Heat equation PDE implementation.
- `src/equations/wave_equation.py`: Wave equation PDE implementation.
- `src/equations/burgers_equation.py`: Burgers' equation PDE implementation.
- `src/data/generators/data_generator.py`: Base class for data generation.
- `src/data/generators/heat_data.py`: Data generator for the heat equation.
- `src/data/generators/wave_data.py`: Data generator for the wave equation.
- `src/data/generators/burgers_data.py`: Data generator for the Burgers' equation.
- `src/data/initial_conditions/ic_heat.py`: Initial conditions for the heat equation.
- `src/data/initial_conditions/ic_wave.py`: Initial conditions for the wave equation.
- `src/data/initial_conditions/ic_burgers.py`: Initial conditions for the Burgers' equation.
- `src/data/boundary_conditions/bc_heat.py`: Boundary conditions for the heat equation.
- `src/data/boundary_conditions/bc_wave.py`: Boundary conditions for the wave equation.
- `src/data/boundary_conditions/bc_burgers.py`: Boundary conditions for the Burgers' equation.
- `src/training/trainer.py`: Base class for training PINN models.
- `src/training/heat_trainer.py`: Training logic for the heat equation.
- `src/training/wave_trainer.py`: Training logic for the wave equation.
- `src/training/burgers_trainer.py`: Training logic for the Burgers' equation.
- `src/evaluation/evaluator.py`: Base class for evaluating PINN models.
- `src/evaluation/heat_evaluator.py`: Evaluation logic for the heat equation.
- `src/evaluation/wave_evaluator.py`: Evaluation logic for the wave equation.
- `src/evaluation/burgers_evaluator.py`: Evaluation logic for the Burgers' equation.
- `src/utils/logger.py`: Logging utilities.
- `src/utils/visualization.py`: Plotting and visualization functions.
- `src/utils/metrics.py`: Loss and evaluation metrics.
- `src/utils/config_parser.py`: YAML configuration parser.

### `configs/` Directory
Configuration files for PINN hyperparameters and PDE parameters.
- `configs/base_config.yaml`: Base configuration for PINN models and training.
- `configs/equations/heat_equation.yaml`: Heat equation-specific parameters.
- `configs/equations/wave_equation.yaml`: Wave equation-specific parameters.
- `configs/equations/burgers_equation.yaml`: Burgers' equation-specific parameters.

### `scripts/` Directory
Scripts for hyperparameter tuning, data preprocessing, and experiments.
- `scripts/hyperparameter_tuning/tune_heat.py`: Hyperparameter tuning for the heat equation.
- `scripts/hyperparameter_tuning/tune_wave.py`: Hyperparameter tuning for the wave equation.
- `scripts/hyperparameter_tuning/tune_burgers.py`: Hyperparameter tuning for the Burgers' equation.
- `scripts/hyperparameter_tuning/tuning_utils.py`: Shared utilities for tuning.
- `scripts/preprocess_data.py`: Data preprocessing script.
- `scripts/run_batch_experiments.py`: Runs multiple experiments.
- `scripts/analyze_results.py`: Analyzes experiment results.

### `results/` Directory
Stores model checkpoints, plots, metrics, and tuning results.
- `results/heat/`: Results for the heat equation.
  - `models/`: Saved model checkpoints.
  - `plots/`: Visualization outputs.
  - `metrics/`: Evaluation metrics.
  - `tuning/`: Hyperparameter tuning results.
- `results/wave/`: Results for the wave equation (same structure).
- `results/burgers/`: Results for the Burgers' equation (same structure).

### `tests/` Directory
Unit tests for all components.
- `test_data_generation.py`: Tests for data generators.
- `test_models.py`: Tests for PINN models.
- `test_equations.py`: Tests for PDE implementations.
- `test_training.py`: Tests for training logic.
- `test_evaluation.py`: Tests for evaluation logic.
- `test_api.py`: Tests for FastAPI endpoints.
- `test_scripts.py`: Tests for scripts (e.g., hyperparameter tuning).
- `test_utils.py`: Tests for utility functions.

## Setup
1. Clone the repository or create the structure using the provided `setup_project.sh` script.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up a virtual environment (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

## Usage

### Running the FastAPI Server
Start the FastAPI server to expose PINN functionality:
```bash
python main.py
```
Access the API at `http://localhost:8000`. Use the interactive API docs at `http://localhost:8000/docs`.

### Training a PINN Model
Train a model via the API:
```bash
curl -X POST "http://localhost:8000/heat/train" -H "Content-Type: application/json" -d '{"epochs": 1000, "learning_rate": 0.001}'
```
Or use a script for offline training:
```bash
python scripts/run_batch_experiments.py
```

### Predicting with a Trained Model
Make predictions via the API:
```bash
curl -X POST "http://localhost:8000/heat/predict" -H "Content-Type: application/json" -d '{"x": [0.0, 0.5, 1.0], "t": [0.0, 0.1, 0.2]}'
```

### Hyperparameter Tuning
Run hyperparameter tuning for a specific equation:
```bash
python scripts/hyperparameter_tuning/tune_heat.py
```
Results are saved in `results/heat/tuning/`.

### Running Tests
Execute unit tests:
```bash
pytest tests/
```

## Adding a New PDE
To add a new PDE (e.g., advection equation):
1. Create equation-specific files:
   - `src/models/advection_pinn.py`
   - `src/equations/advection_equation.py`
   - `src/data/generators/advection_data.py`
   - `src/data/initial_conditions/ic_advection.py`
   - `src/data/boundary_conditions/bc_advection.py`
   - `src/training/advection_trainer.py`
   - `src/evaluation/advection_evaluator.py`
   - `app/api/endpoints/advection.py`
   - `app/schemas/advection.py`
   - `configs/equations/advection_equ
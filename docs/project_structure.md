# Project Structure Documentation

## Overview
This document provides a detailed description of the project structure and the purpose of each file and directory.

## Directory Structure

### `app/` Directory
The `app` directory contains the FastAPI application code for exposing PINN functionality via RESTful APIs.

#### `api/` Subdirectory
- `endpoints/`: Contains API endpoints for each equation type
  - `heat.py`: API endpoints for the heat equation
  - `wave.py`: API endpoints for the wave equation
  - `burgers.py`: API endpoints for Burgers' equation
- `router.py`: Aggregates API endpoints for all equations

#### `core/` Subdirectory
- `config.py`: Configures FastAPI settings (e.g., CORS)
- `dependencies.py`: Manages dependency injection for models and trainers

#### `schemas/` Subdirectory
- `heat.py`: Pydantic schemas for heat equation API requests/responses
- `wave.py`: Pydantic schemas for wave equation API requests/responses
- `burgers.py`: Pydantic schemas for Burgers' equation API requests/responses

### `src/` Directory
The `src` directory contains the core PINN implementation.

#### `models/` Subdirectory
- `base_pinn.py`: Base class for PINN neural networks
- `heat_pinn.py`: PINN model for the heat equation
- `wave_pinn.py`: PINN model for the wave equation
- `burgers_pinn.py`: PINN model for Burgers' equation

#### `equations/` Subdirectory
- `base_equation.py`: Abstract base class for PDEs
- `heat_equation.py`: Heat equation PDE implementation
- `wave_equation.py`: Wave equation PDE implementation
- `burgers_equation.py`: Burgers' equation PDE implementation

#### `data/` Subdirectory
- `generators/`: Data generation for training and validation
  - `data_generator.py`: Base class for data generation
  - `heat_data.py`: Data generator for the heat equation
  - `wave_data.py`: Data generator for the wave equation
  - `burgers_data.py`: Data generator for Burgers' equation
- `initial_conditions/`: Initial conditions for each equation
  - `ic_heat.py`: Initial conditions for the heat equation
  - `ic_wave.py`: Initial conditions for the wave equation
  - `ic_burgers.py`: Initial conditions for Burgers' equation
- `boundary_conditions/`: Boundary conditions for each equation
  - `bc_heat.py`: Boundary conditions for the heat equation
  - `bc_wave.py`: Boundary conditions for the wave equation
  - `bc_burgers.py`: Boundary conditions for Burgers' equation

#### `training/` Subdirectory
- `trainer.py`: Base class for training PINN models
- `heat_trainer.py`: Training logic for the heat equation
- `wave_trainer.py`: Training logic for the wave equation
- `burgers_trainer.py`: Training logic for Burgers' equation

#### `evaluation/` Subdirectory
- `evaluator.py`: Base class for evaluating PINN models
- `heat_evaluator.py`: Evaluation logic for the heat equation
- `wave_evaluator.py`: Evaluation logic for the wave equation
- `burgers_evaluator.py`: Evaluation logic for Burgers' equation

#### `utils/` Subdirectory
- `logger.py`: Logging utilities
- `visualization.py`: Plotting and visualization functions
- `metrics.py`: Loss and evaluation metrics
- `config_parser.py`: YAML configuration parser

### `configs/` Directory
The `configs` directory contains configuration files for PINN hyperparameters and PDE parameters.

- `base_config.yaml`: Base configuration for PINN models and training
- `equations/`: Equation-specific parameters
  - `heat_equation.yaml`: Heat equation parameters
  - `wave_equation.yaml`: Wave equation parameters
  - `burgers_equation.yaml`: Burgers' equation parameters

### `scripts/` Directory
The `scripts` directory contains utility scripts for various tasks.

- `hyperparameter_tuning/`: Scripts for optimizing model parameters
  - `tune_heat.py`: Hyperparameter tuning for the heat equation
  - `tune_wave.py`: Hyperparameter tuning for the wave equation
  - `tune_burgers.py`: Hyperparameter tuning for Burgers' equation
  - `tuning_utils.py`: Shared utilities for tuning
- `preprocess_data.py`: Data preprocessing script
- `run_batch_experiments.py`: Runs multiple experiments
- `analyze_results.py`: Analyzes experiment results

### `results/` Directory
The `results` directory stores model checkpoints, plots, metrics, and tuning results.

- `heat/`: Results for the heat equation
  - `models/`: Saved model checkpoints
  - `plots/`: Visualization outputs
  - `metrics/`: Evaluation metrics
  - `tuning/`: Hyperparameter tuning results
- `wave/`: Results for the wave equation (same structure)
- `burgers/`: Results for Burgers' equation (same structure)

### `tests/` Directory
The `tests` directory contains unit tests for all components.

- `test_models/`: Tests for PINN models
- `test_equations/`: Tests for PDE implementations
- `test_data/`: Tests for data generators
- `test_training/`: Tests for training logic
- `test_evaluation/`: Tests for evaluation logic

## File Descriptions

### Configuration Files
- `base_config.yaml`: Contains common configuration parameters for all equations
- `heat_equation.yaml`: Contains heat equation-specific parameters
- `wave_equation.yaml`: Contains wave equation-specific parameters
- `burgers_equation.yaml`: Contains Burgers' equation-specific parameters

### API Files
- `router.py`: Sets up the FastAPI router and includes the routers for each equation type
- `heat.py`: Implements the heat equation API endpoints for training and prediction
- `wave.py`: Implements the wave equation API endpoints for training and prediction
- `burgers.py`: Implements the Burgers' equation API endpoints for training and prediction

### Model Files
- `base_pinn.py`: Defines the base class for PINN neural networks
- `heat_pinn.py`: Implements the PINN model for the heat equation
- `wave_pinn.py`: Implements the PINN model for the wave equation
- `burgers_pinn.py`: Implements the PINN model for Burgers' equation

### Training Files
- `trainer.py`: Defines the base class for training PINN models
- `heat_trainer.py`: Implements the training logic for the heat equation
- `wave_trainer.py`: Implements the training logic for the wave equation
- `burgers_trainer.py`: Implements the training logic for Burgers' equation

### Utility Files
- `logger.py`: Provides logging utilities for the project
- `visualization.py`: Contains functions for plotting and visualizing results
- `metrics.py`: Defines loss and evaluation metrics
- `config_parser.py`: Provides utilities for parsing YAML configuration files 
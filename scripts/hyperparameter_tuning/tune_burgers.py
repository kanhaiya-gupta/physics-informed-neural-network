import optuna
import torch
from src.training.burgers_trainer import BurgersTrainer
from src.utils.config_parser import load_config
from .tuning_utils import create_study

def objective(trial):
    """
    Objective function for hyperparameter optimization
    Args:
        trial: Optuna trial object
    Returns:
        loss: validation loss
    """
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    nu = trial.suggest_float("nu", 0.001, 0.1, log=True)
    layers = [2] + [trial.suggest_int(f"layer_{i}", 10, 50) for i in range(3)] + [1]
    
    # Load and update config
    config = load_config("configs/equations/burgers_equation.yaml")
    config["training"]["learning_rate"] = lr
    config["model"]["layers"] = layers
    config["equation"]["nu"] = nu
    
    # Initialize trainer
    trainer = BurgersTrainer(nu=nu, lr=lr)
    
    # Train model
    trainer.train(epochs=1000)
    
    # Generate validation points
    x_val = torch.linspace(-1, 1, 100).reshape(-1, 1)
    t_val = torch.linspace(0, 1, 100).reshape(-1, 1)
    
    # Evaluate model
    _, loss = trainer.evaluate(x_val, t_val)
    
    return loss.item()

def tune_burgers_hyperparameters(n_trials=50):
    """
    Run hyperparameter optimization for Burgers' equation
    Args:
        n_trials: number of optimization trials
    """
    # Create study
    study = create_study("burgers_hyperparameter_tuning")
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials)
    
    # Print results
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print(f"\nBest validation loss: {study.best_value:.6f}")
    
    # Save results
    study.trials_dataframe().to_csv("results/burgers_hyperparameter_tuning.csv")
    
    return study.best_params

if __name__ == "__main__":
    tune_burgers_hyperparameters()

import optuna
import torch
from src.training.wave_trainer import WaveTrainer
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
    c = trial.suggest_float("c", 0.5, 2.0)
    layers = [2] + [trial.suggest_int(f"layer_{i}", 10, 50) for i in range(3)] + [1]
    
    # Load and update config
    config = load_config("configs/equations/wave_equation.yaml")
    config["training"]["learning_rate"] = lr
    config["model"]["layers"] = layers
    config["equation"]["c"] = c
    
    # Initialize trainer
    trainer = WaveTrainer(c=c, lr=lr)
    
    # Train model
    trainer.train(epochs=1000)
    
    # Generate validation points
    x_val = torch.linspace(0, 1, 100).reshape(-1, 1)
    t_val = torch.linspace(0, 1, 100).reshape(-1, 1)
    
    # Evaluate model
    _, loss = trainer.evaluate(x_val, t_val)
    
    return loss.item()

def tune_wave_hyperparameters(n_trials=50):
    """
    Run hyperparameter optimization for wave equation
    Args:
        n_trials: number of optimization trials
    """
    # Create study
    study = create_study("wave_hyperparameter_tuning")
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials)
    
    # Print results
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print(f"\nBest validation loss: {study.best_value:.6f}")
    
    # Save results
    study.trials_dataframe().to_csv("results/wave_hyperparameter_tuning.csv")
    
    return study.best_params

if __name__ == "__main__":
    tune_wave_hyperparameters()

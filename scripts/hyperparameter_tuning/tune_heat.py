import optuna
from src.training.heat_trainer import HeatTrainer
from src.utils.config_parser import load_config

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    layers = [2] + [trial.suggest_int(f"layer_{i}", 10, 50) for i in range(2)] + [1]
    config = load_config("configs/equations/heat_equation.yaml")
    config["training"]["learning_rate"] = lr
    config["model"]["layers"] = layers
    trainer = HeatTrainer()  # Modify to accept config
    trainer.train(epochs=500, lr=lr)  # Simplified for demo
    return trainer.evaluate()  # Placeholder: Return evaluation metric

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)
print("Best hyperparameters:", study.best_params)
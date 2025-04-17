import os
import yaml
import torch
import numpy as np
from datetime import datetime
from src.training.heat_trainer import HeatTrainer
from src.training.burgers_trainer import BurgersTrainer
from src.training.wave_trainer import WaveTrainer
from src.utils.config_parser import load_config, save_config

def run_experiment(equation_type, config, experiment_id):
    """
    Run a single experiment
    Args:
        equation_type: type of equation ('heat', 'burgers', or 'wave')
        config: configuration dictionary
        experiment_id: unique identifier for the experiment
    Returns:
        results: dictionary containing experiment results
    """
    # Create results directory
    results_dir = f"results/experiments/{equation_type}/{experiment_id}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save configuration
    save_config(config, os.path.join(results_dir, "config.yaml"))
    
    # Initialize trainer based on equation type
    if equation_type == 'heat':
        trainer = HeatTrainer()
    elif equation_type == 'burgers':
        trainer = BurgersTrainer(nu=config['equation']['nu'])
    elif equation_type == 'wave':
        trainer = WaveTrainer(c=config['equation']['c'])
    else:
        raise ValueError(f"Unknown equation type: {equation_type}")
    
    # Train model
    trainer.train(epochs=config['training']['epochs'])
    
    # Generate test points
    x_test = torch.linspace(config['domain']['x_min'], 
                           config['domain']['x_max'], 
                           100).reshape(-1, 1)
    t_test = torch.linspace(config['domain']['t_min'], 
                           config['domain']['t_max'], 
                           100).reshape(-1, 1)
    
    # Evaluate model
    u_pred, loss = trainer.evaluate(x_test, t_test)
    
    # Save results
    results = {
        'loss': loss.item(),
        'x_test': x_test.numpy(),
        't_test': t_test.numpy(),
        'u_pred': u_pred.detach().numpy()
    }
    
    np.savez(os.path.join(results_dir, "results.npz"), **results)
    
    return results

def generate_configurations(base_config, variations):
    """
    Generate multiple configurations by varying parameters
    Args:
        base_config: base configuration dictionary
        variations: dictionary of parameter variations
    Returns:
        configs: list of configuration dictionaries
    """
    configs = []
    
    # Generate all combinations of variations
    from itertools import product
    keys = list(variations.keys())
    values = list(variations.values())
    for combination in product(*values):
        config = base_config.copy()
        for key, value in zip(keys, combination):
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                current = current[k]
            current[keys[-1]] = value
        configs.append(config)
    
    return configs

def main():
    """
    Main function to run batch experiments
    """
    # Load base configurations
    heat_config = load_config("configs/equations/heat_equation.yaml")
    burgers_config = load_config("configs/equations/burgers_equation.yaml")
    wave_config = load_config("configs/equations/wave_equation.yaml")
    
    # Define parameter variations
    variations = {
        'training.learning_rate': [1e-3, 1e-4],
        'training.epochs': [1000, 2000],
        'model.layers': [
            [2, 20, 20, 1],
            [2, 30, 30, 1],
            [2, 40, 40, 1]
        ]
    }
    
    # Generate configurations
    heat_configs = generate_configurations(heat_config, variations)
    burgers_configs = generate_configurations(burgers_config, variations)
    wave_configs = generate_configurations(wave_config, variations)
    
    # Run experiments
    for i, config in enumerate(heat_configs):
        experiment_id = f"heat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
        print(f"\nRunning experiment {experiment_id}...")
        run_experiment('heat', config, experiment_id)
    
    for i, config in enumerate(burgers_configs):
        experiment_id = f"burgers_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
        print(f"\nRunning experiment {experiment_id}...")
        run_experiment('burgers', config, experiment_id)
    
    for i, config in enumerate(wave_configs):
        experiment_id = f"wave_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
        print(f"\nRunning experiment {experiment_id}...")
        run_experiment('wave', config, experiment_id)

if __name__ == "__main__":
    main()

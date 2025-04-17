import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.config_parser import load_config

def load_tuning_results(equation_type):
    """
    Load hyperparameter tuning results
    Args:
        equation_type: type of equation ('heat', 'burgers', or 'wave')
    Returns:
        df: DataFrame containing tuning results
    """
    results_path = f"results/{equation_type}_hyperparameter_tuning.csv"
    if os.path.exists(results_path):
        return pd.read_csv(results_path)
    else:
        raise FileNotFoundError(f"Results file not found: {results_path}")

def plot_tuning_results(equation_type):
    """
    Plot hyperparameter tuning results
    Args:
        equation_type: type of equation ('heat', 'burgers', or 'wave')
    """
    # Load results
    df = load_tuning_results(equation_type)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot learning rate vs loss
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=df, x='params_lr', y='value')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate vs Loss')
    
    # Plot layer sizes vs loss
    layer_cols = [col for col in df.columns if col.startswith('params_layer_')]
    for i, col in enumerate(layer_cols):
        plt.subplot(2, 2, i+2)
        sns.scatterplot(data=df, x=col, y='value')
        plt.xlabel(f'Layer {i+1} Size')
        plt.ylabel('Loss')
        plt.title(f'Layer {i+1} Size vs Loss')
    
    plt.tight_layout()
    plt.savefig(f"results/{equation_type}_tuning_plots.png")
    plt.close()

def analyze_model_performance(equation_type):
    """
    Analyze model performance
    Args:
        equation_type: type of equation ('heat', 'burgers', or 'wave')
    """
    # Load config
    config = load_config(f"configs/equations/{equation_type}_equation.yaml")
    
    # Load results
    df = load_tuning_results(equation_type)
    
    # Calculate statistics
    stats = {
        'mean_loss': df['value'].mean(),
        'std_loss': df['value'].std(),
        'min_loss': df['value'].min(),
        'max_loss': df['value'].max(),
        'best_lr': df.loc[df['value'].idxmin(), 'params_lr'],
        'best_layers': [df.loc[df['value'].idxmin(), f'params_layer_{i}'] 
                       for i in range(len([col for col in df.columns if col.startswith('params_layer_')]))]
    }
    
    # Save statistics
    with open(f"results/{equation_type}_performance_stats.txt", 'w') as f:
        f.write(f"Performance Statistics for {equation_type} equation:\n")
        f.write(f"Mean Loss: {stats['mean_loss']:.6f}\n")
        f.write(f"Standard Deviation: {stats['std_loss']:.6f}\n")
        f.write(f"Minimum Loss: {stats['min_loss']:.6f}\n")
        f.write(f"Maximum Loss: {stats['max_loss']:.6f}\n")
        f.write(f"Best Learning Rate: {stats['best_lr']:.6f}\n")
        f.write(f"Best Layer Sizes: {stats['best_layers']}\n")
    
    return stats

def main():
    """
    Main function to analyze results for all equations
    """
    equations = ['heat', 'burgers', 'wave']
    
    for eq in equations:
        print(f"\nAnalyzing results for {eq} equation...")
        try:
            # Plot tuning results
            plot_tuning_results(eq)
            print(f"Generated tuning plots for {eq} equation")
            
            # Analyze performance
            stats = analyze_model_performance(eq)
            print(f"Generated performance statistics for {eq} equation")
            print(f"Best loss: {stats['min_loss']:.6f}")
            
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

if __name__ == "__main__":
    main()

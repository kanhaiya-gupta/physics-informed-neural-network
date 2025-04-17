import os
import numpy as np
import torch
from src.data.generators.heat_data import HeatDataGenerator
from src.data.generators.burgers_data import BurgersDataGenerator
from src.data.generators.wave_data import WaveDataGenerator

def preprocess_heat_data(num_points=1000, save_dir="data/heat"):
    """
    Preprocess data for heat equation
    Args:
        num_points: number of points to generate
        save_dir: directory to save preprocessed data
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize data generator
    generator = HeatDataGenerator(num_points=num_points)
    
    # Generate collocation points
    x, t = generator.generate_collocation_points()
    
    # Generate boundary points
    x_bc, t_bc = generator.generate_boundary_points()
    
    # Save data
    np.savez(os.path.join(save_dir, "collocation_points.npz"),
             x=x.numpy(), t=t.numpy())
    np.savez(os.path.join(save_dir, "boundary_points.npz"),
             x=x_bc.numpy(), t=t_bc.numpy())

def preprocess_burgers_data(num_points=1000, save_dir="data/burgers"):
    """
    Preprocess data for Burgers' equation
    Args:
        num_points: number of points to generate
        save_dir: directory to save preprocessed data
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize data generator
    generator = BurgersDataGenerator(num_points=num_points)
    
    # Generate collocation points
    x, t = generator.generate_collocation_points()
    
    # Generate boundary points
    x_bc, t_bc = generator.generate_boundary_points()
    
    # Generate initial condition points
    x_ic, t_ic, u_ic = generator.generate_initial_condition()
    
    # Save data
    np.savez(os.path.join(save_dir, "collocation_points.npz"),
             x=x.numpy(), t=t.numpy())
    np.savez(os.path.join(save_dir, "boundary_points.npz"),
             x=x_bc.numpy(), t=t_bc.numpy())
    np.savez(os.path.join(save_dir, "initial_condition.npz"),
             x=x_ic.numpy(), t=t_ic.numpy(), u=u_ic.numpy())

def preprocess_wave_data(num_points=1000, save_dir="data/wave"):
    """
    Preprocess data for wave equation
    Args:
        num_points: number of points to generate
        save_dir: directory to save preprocessed data
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize data generator
    generator = WaveDataGenerator(num_points=num_points)
    
    # Generate collocation points
    x, t = generator.generate_collocation_points()
    
    # Generate boundary points
    x_bc, t_bc = generator.generate_boundary_points()
    
    # Generate initial condition points
    x_ic, t_ic, u_ic, u_t_ic = generator.generate_initial_condition()
    
    # Save data
    np.savez(os.path.join(save_dir, "collocation_points.npz"),
             x=x.numpy(), t=t.numpy())
    np.savez(os.path.join(save_dir, "boundary_points.npz"),
             x=x_bc.numpy(), t=t_bc.numpy())
    np.savez(os.path.join(save_dir, "initial_condition.npz"),
             x=x_ic.numpy(), t=t_ic.numpy(), u=u_ic.numpy(), u_t=u_t_ic.numpy())

def main():
    """
    Main function to preprocess data for all equations
    """
    # Preprocess heat equation data
    print("Preprocessing heat equation data...")
    preprocess_heat_data()
    
    # Preprocess Burgers' equation data
    print("Preprocessing Burgers' equation data...")
    preprocess_burgers_data()
    
    # Preprocess wave equation data
    print("Preprocessing wave equation data...")
    preprocess_wave_data()
    
    print("Data preprocessing completed!")

if __name__ == "__main__":
    main()

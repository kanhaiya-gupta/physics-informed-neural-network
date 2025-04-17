import pytest
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from src.training.heat_trainer import HeatTrainer
from src.training.burgers_trainer import BurgersTrainer
from src.training.wave_trainer import WaveTrainer
from src.training.shm_trainer import SHMTrainer

def test_heat_trainer():
    trainer = HeatTrainer()
    trainer.train(epochs=1)
    assert len(trainer.loss_history) == 1
    assert isinstance(trainer.loss_history[0], float)
    assert trainer.loss_history[0] > 0

def test_burgers_trainer():
    trainer = BurgersTrainer()
    trainer.train(epochs=1)
    assert len(trainer.loss_history) == 1
    assert isinstance(trainer.loss_history[0], float)
    assert trainer.loss_history[0] > 0

def test_wave_trainer():
    trainer = WaveTrainer()
    trainer.train(epochs=1)
    assert len(trainer.loss_history) == 1
    assert isinstance(trainer.loss_history[0], float)
    assert trainer.loss_history[0] > 0

def test_shm_trainer():
    trainer = SHMTrainer()
    trainer.train(epochs=1)
    assert len(trainer.loss_history) == 1
    assert isinstance(trainer.loss_history[0], float)
    assert trainer.loss_history[0] > 0

def test_training_convergence():
    # Test that loss decreases over multiple epochs
    trainer = HeatTrainer()
    trainer.train(epochs=1)
    initial_loss = trainer.loss_history[0]
    trainer.train(epochs=10)
    final_loss = trainer.loss_history[-1]
    assert final_loss < initial_loss

def test_model_saving():
    trainer = HeatTrainer()
    trainer.train(epochs=1)
    # Check if model file exists
    import os
    assert os.path.exists('results/heat/models/model.pth')
    assert os.path.exists('results/heat/metrics/loss_history.npy')

def test_plot_generation():
    trainer = HeatTrainer()
    trainer.train(epochs=1)
    # Check if plot files exist
    import os
    assert os.path.exists('results/heat/plots/loss_curve.png')
    assert os.path.exists('results/heat/plots/solution_comparison.png')
    assert os.path.exists('results/heat/plots/solution_slice.png')

def test_training_parameters():
    # Test different learning rates
    trainer1 = HeatTrainer()
    trainer1.optimizer = torch.optim.Adam(trainer1.model.parameters(), lr=0.01)
    trainer1.train(epochs=1)
    loss1 = trainer1.loss_history[0]
    
    trainer2 = HeatTrainer()
    trainer2.optimizer = torch.optim.Adam(trainer2.model.parameters(), lr=0.001)
    trainer2.train(epochs=1)
    loss2 = trainer2.loss_history[0]
    
    assert loss1 != loss2  # Different learning rates should give different losses

import pytest
import yaml
import os
from src.utils.config_parser import load_config, save_config

def test_load_config():
    # Create a temporary config file
    config_data = {
        "model": {
            "layers": [2, 50, 50, 1],
            "activation": "tanh"
        },
        "training": {
            "epochs": 1000,
            "learning_rate": 0.001
        }
    }
    
    # Save the config
    config_path = "temp_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    
    # Test loading
    loaded_config = load_config(config_path)
    assert loaded_config == config_data
    
    # Clean up
    os.remove(config_path)

def test_save_config():
    # Test data
    config_data = {
        "model": {
            "layers": [2, 50, 50, 1],
            "activation": "tanh"
        },
        "training": {
            "epochs": 1000,
            "learning_rate": 0.001
        }
    }
    
    # Save the config
    config_path = "temp_config.yaml"
    save_config(config_data, config_path)
    
    # Verify the file exists and contains correct data
    assert os.path.exists(config_path)
    with open(config_path, "r") as f:
        loaded_data = yaml.safe_load(f)
    assert loaded_data == config_data
    
    # Clean up
    os.remove(config_path)

def test_load_config_nonexistent():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")

def test_save_config_invalid_path():
    with pytest.raises(IOError):
        save_config({}, "/nonexistent/path/config.yaml")

def test_config_validation():
    # Test invalid config data
    invalid_config = {
        "model": {
            "layers": "invalid",  # Should be a list
            "activation": 123  # Should be a string
        }
    }
    
    config_path = "temp_config.yaml"
    with pytest.raises(yaml.YAMLError):
        save_config(invalid_config, config_path)
    
    if os.path.exists(config_path):
        os.remove(config_path)

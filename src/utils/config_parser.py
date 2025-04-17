import yaml

def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_config(config, file_path):
    """Save a configuration dictionary to a YAML file.
    
    Args:
        config (dict): Configuration dictionary to save
        file_path (str): Path where to save the YAML file
    """
    with open(file_path, 'w') as file:
        yaml.safe_dump(config, file, default_flow_style=False)
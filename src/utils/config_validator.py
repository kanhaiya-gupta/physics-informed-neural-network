import yaml
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validates configuration files for PINN equations."""
    
    REQUIRED_SECTIONS = ["equation", "model", "training", "data"]
    OPTIONAL_SECTIONS = ["results"]
    
    # Required fields for each equation type
    EQUATION_REQUIRED_FIELDS = {
        "heat": {
            "equation": ["alpha", "x_min", "x_max", "t_min", "t_max"],
            "model": ["hidden_layers", "activation"],
            "training": ["epochs", "learning_rate", "batch_size", "n_collocation"],
            "data": ["n_test", "noise_level"]
        },
        "wave": {
            "equation": ["c", "x_min", "x_max", "t_min", "t_max"],
            "model": ["hidden_layers", "activation"],
            "training": ["epochs", "learning_rate", "batch_size", "n_collocation"],
            "data": ["n_test", "noise_level"]
        },
        "burgers": {
            "equation": ["nu", "x_min", "x_max", "t_min", "t_max"],
            "model": ["hidden_layers", "activation"],
            "training": ["epochs", "learning_rate", "batch_size", "n_collocation"],
            "data": ["n_test", "noise_level"]
        },
        "shm": {
            "equation": ["omega"],
            "model": ["layers", "activation"],
            "training": ["epochs", "learning_rate", "batch_size"],
            "data": ["n_test", "noise_level"]
        }
    }
    
    # Valid values for certain fields
    VALID_VALUES = {
        "activation": ["tanh", "relu", "sigmoid"],
        "optimizer": ["adam", "sgd", "rmsprop"],
        "scheduler": ["cosine", "step", "plateau", None],
        "ic_type": ["sin", "gaussian", "constant"],
        "bc_type": ["dirichlet", "neumann", "periodic"]
    }
    
    @staticmethod
    def get_equation_type(config_path: str) -> str:
        """Extracts the equation type from the config file path."""
        filename = Path(config_path).stem
        return filename.split('_')[0]  # e.g., 'heat_equation.yaml' -> 'heat'
    
    @staticmethod
    def validate_config(config: Dict[str, Any], config_path: str) -> List[str]:
        """
        Validates a configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            config_path: Path to the configuration file (for logging)
            
        Returns:
            List of validation errors, empty if valid
        """
        errors = []
        equation_type = ConfigValidator.get_equation_type(config_path)
        
        if equation_type not in ConfigValidator.EQUATION_REQUIRED_FIELDS:
            errors.append(f"Unknown equation type: {equation_type}")
            return errors
        
        required_fields = ConfigValidator.EQUATION_REQUIRED_FIELDS[equation_type]
        
        # Check required sections
        for section in ConfigValidator.REQUIRED_SECTIONS:
            if section not in config:
                errors.append(f"Missing required section: {section}")
                continue
                
            # Check required fields in section
            if section in required_fields:
                for field in required_fields[section]:
                    if field not in config[section]:
                        errors.append(f"Missing required field '{field}' in section '{section}'")
        
        # Validate field values
        for section, fields in config.items():
            for field, value in fields.items():
                # Check if field has valid values defined
                if field in ConfigValidator.VALID_VALUES:
                    if value not in ConfigValidator.VALID_VALUES[field]:
                        errors.append(
                            f"Invalid value '{value}' for field '{field}' in section '{section}'. "
                            f"Valid values: {ConfigValidator.VALID_VALUES[field]}"
                        )
                
                # Type-specific validation
                if field in ["hidden_layers", "layers"] and not isinstance(value, list):
                    errors.append(f"Field '{field}' must be a list, got {type(value)}")
                elif field in ["epochs", "batch_size", "n_collocation", "n_test"]:
                    if not isinstance(value, int) or value <= 0:
                        errors.append(f"Field '{field}' must be a positive integer, got {value}")
                elif field in ["learning_rate", "noise_level", "alpha", "c", "nu", "omega"]:
                    if not isinstance(value, (int, float)) or value < 0:
                        errors.append(f"Field '{field}' must be a non-negative number, got {value}")
        
        return errors
    
    @classmethod
    def validate_config_file(cls, config_path: str) -> bool:
        """
        Validates a configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            errors = cls.validate_config(config, config_path)
            
            if errors:
                logger.error(f"Configuration file {config_path} is invalid:")
                for error in errors:
                    logger.error(f"  - {error}")
                return False
            
            logger.info(f"Configuration file {config_path} is valid")
            return True
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file {config_path}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error validating configuration file {config_path}: {str(e)}")
            return False

def validate_all_configs():
    """Validates all configuration files in the configs/equations directory."""
    config_dir = Path("configs/equations")
    if not config_dir.exists():
        logger.error(f"Configuration directory {config_dir} does not exist")
        return False
    
    all_valid = True
    for config_file in config_dir.glob("*.yaml"):
        if not ConfigValidator.validate_config_file(str(config_file)):
            all_valid = False
    
    return all_valid

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    validate_all_configs() 
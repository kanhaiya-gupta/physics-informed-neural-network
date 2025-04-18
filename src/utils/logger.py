import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_dir="logs", log_file=None):
    """Set up a logger with both file and console handlers.
    
    Args:
        name (str): Name of the logger
        log_dir (str): Directory to store log files
        log_file (str, optional): Specific log file name. If None, uses name.log
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid adding handlers if they already exist
    if logger.handlers:
        return logger
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    if log_file is None:
        log_file = f"{name}.log"
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, log_file),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create loggers for different components
api_logger = setup_logger('api', log_file='api.log')

# Create separate loggers for each equation type
shm_logger = setup_logger('shm', log_file='shm_training.log')
heat_logger = setup_logger('heat', log_file='heat_training.log')
wave_logger = setup_logger('wave', log_file='wave_training.log')
burgers_logger = setup_logger('burgers', log_file='burgers_training.log')

def log_api_request(endpoint: str, request_data: dict):
    """Log API request details."""
    api_logger.info(f"Request to {endpoint} - Data: {request_data}")

def log_api_response(endpoint: str, response_data: dict, status_code: int):
    """Log API response details."""
    api_logger.info(f"Response from {endpoint} - Status: {status_code} - Data: {response_data}")

def log_api_error(endpoint: str, error: Exception, status_code: int):
    """Log API errors."""
    api_logger.error(f"Error in {endpoint} - Status: {status_code} - Error: {str(error)}")

def get_equation_logger(equation_type: str) -> logging.Logger:
    """Get the appropriate logger for the given equation type."""
    loggers = {
        'shm': shm_logger,
        'heat': heat_logger,
        'wave': wave_logger,
        'burgers': burgers_logger
    }
    return loggers.get(equation_type.lower(), logging.getLogger(equation_type))

def log_training_progress(model_type: str, epoch: int, loss: float, additional_metrics: dict = None):
    """Log training progress."""
    logger = get_equation_logger(model_type)
    msg = f"Epoch {epoch} - Loss: {loss:.6f}"
    if additional_metrics:
        msg += f" - Metrics: {additional_metrics}"
    logger.info(msg)

def log_training_start(model_type: str, config: dict):
    """Log start of training with configuration."""
    logger = get_equation_logger(model_type)
    logger.info(f"Starting training with config: {config}")

def log_training_complete(model_type: str, final_loss: float, training_time: float):
    """Log training completion with metrics."""
    logger = get_equation_logger(model_type)
    logger.info(
        f"Training completed - Final Loss: {final_loss:.6f} - "
        f"Training Time: {training_time:.2f}s"
    )

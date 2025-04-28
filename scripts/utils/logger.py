import logging
import logging.handlers
import os
import yaml
from pathlib import Path
from datetime import datetime

def setup_logger(name, log_dir='logs'):
    """Set up a logger with file and console handlers."""
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Create file handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'{name}_{timestamp}.log')
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_error(logger, error, context=None):
    """Log an error with context information."""
    if context:
        logger.error(f"Error in {context}: {str(error)}")
    else:
        logger.error(f"Error: {str(error)}")
    logger.exception(error)

def log_info(logger, message, context=None):
    """Log an info message with optional context."""
    if context:
        logger.info(f"{context}: {message}")
    else:
        logger.info(message)

def log_warning(logger, message, context=None):
    """Log a warning message with optional context."""
    if context:
        logger.warning(f"{context}: {message}")
    else:
        logger.warning(message)

def setup_logger_from_config(config_path: str = "config/config.yaml") -> logging.Logger:
    """
    Set up logging configuration based on YAML config file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        logging.Logger: Configured logger instance
    """
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create logs directory if it doesn't exist
        log_dir = Path(config['paths']['logs'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger('rain_prediction')
        logger.setLevel(config['logging']['level'])
        
        # Create formatter
        formatter = logging.Formatter(config['logging']['format'])
        
        # Create file handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            config['logging']['file'],
            maxBytes=config['logging']['max_size'],
            backupCount=config['logging']['backup_count']
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        logger.info("Logger successfully configured")
        return logger
        
    except Exception as e:
        # Fallback to basic logging if configuration fails
        logging.basicConfig(level=logging.INFO)
        logging.error(f"Failed to configure logger: {str(e)}")
        return logging.getLogger('rain_prediction') 
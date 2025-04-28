import pytest
import logging
import os
from pathlib import Path
from scripts.utils.logger import setup_logger, log_error, log_warning, log_info
from scripts.utils.config_manager import ConfigManager
from scripts.utils.error_handler import (
    RainPredictionError,
    DataLoadingError,
    ModelTrainingError,
    ConfigurationError,
    retry_on_error,
    handle_errors,
    validate_input
)

@pytest.fixture
def test_config():
    return {
        'data': {
            'raw_data_path': 'test_data/raw',
            'processed_data_path': 'test_data/processed',
            'cache_path': 'test_data/cache'
        },
        'model': {
            'architecture': {
                'cnn_layers': [],
                'lstm_layers': [],
                'dense_layers': []
            }
        },
        'logging': {
            'level': 'INFO',
            'format': '%(message)s',
            'file': 'test.log',
            'max_size': 1000,
            'backup_count': 1
        },
        'error_handling': {
            'max_retries': 3,
            'retry_delay': 1
        },
        'paths': {
            'models': 'test_models',
            'results': 'test_results',
            'visualizations': 'test_visualizations',
            'logs': 'test_logs'
        }
    }

@pytest.fixture
def logger(tmp_path, test_config):
    # Create test config file
    config_path = tmp_path / "test_config.yaml"
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)
    
    return setup_logger(str(config_path))

def test_logger_setup(tmp_path, test_config):
    # Create test config file
    config_path = tmp_path / "test_config.yaml"
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)
    
    # Test logger setup
    logger = setup_logger(str(config_path))
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.INFO
    
    # Test logger handlers
    assert len(logger.handlers) == 2  # File and console handlers
    assert any(isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers)
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

def test_logging_functions(logger):
    # Test log_error
    with pytest.raises(Exception):
        log_error(logger, Exception("Test error"), "Test context")
    
    # Test log_warning
    log_warning(logger, "Test warning", "Test context")
    
    # Test log_info
    log_info(logger, "Test info", "Test context")

def test_config_manager(tmp_path, test_config):
    # Create test config file
    config_path = tmp_path / "test_config.yaml"
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)
    
    # Test config manager initialization
    manager = ConfigManager(str(config_path))
    config = manager.get_config()
    assert config['data']['raw_data_path'] == 'test_data/raw'
    
    # Test config validation
    assert manager.validate_config() is True
    
    # Test config update
    updates = {'data': {'raw_data_path': 'new_path'}}
    manager.update_config(updates)
    assert manager.get_config()['data']['raw_data_path'] == 'new_path'
    
    # Test invalid config
    with pytest.raises(ValueError):
        manager.config = None
        manager.validate_config()
    
    # Test missing section
    with pytest.raises(ValueError):
        manager.config = {'data': {}}
        manager.validate_config()
    
    # Test environment variables
    os.environ['RAIN_PREDICTION_DATA__RAW_DATA_PATH'] = 'env_path'
    manager = ConfigManager(str(config_path))
    assert manager.get_config()['data']['raw_data_path'] == 'env_path'

def test_error_handler():
    # Test custom exceptions
    with pytest.raises(DataLoadingError):
        raise DataLoadingError("Test error")
    
    with pytest.raises(ModelTrainingError):
        raise ModelTrainingError("Test error")
    
    with pytest.raises(ConfigurationError):
        raise ConfigurationError("Test error")
    
    # Test retry decorator
    @retry_on_error(max_retries=2, delay=0.1)
    def failing_function():
        raise Exception("Test error")
    
    with pytest.raises(Exception):
        failing_function()
    
    # Test error handling decorator
    @handle_errors()
    def error_function():
        raise Exception("Test error")
    
    with pytest.raises(Exception):
        error_function()
    
    # Test input validation decorator
    @validate_input(lambda x: x > 0, "Value must be positive")
    def positive_function(x):
        return x
    
    with pytest.raises(ValueError):
        positive_function(-1)
    
    assert positive_function(1) == 1

def test_config_manager_save(tmp_path, test_config):
    # Test config saving
    config_path = tmp_path / "test_config.yaml"
    save_path = tmp_path / "saved_config.yaml"
    
    manager = ConfigManager(str(config_path))
    manager.config = test_config
    manager.save_config(str(save_path))
    
    assert save_path.exists()
    with open(save_path, 'r') as f:
        saved_config = yaml.safe_load(f)
    assert saved_config == test_config

def test_cleanup(tmp_path):
    # Clean up test files
    for path in tmp_path.glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    
    # Verify cleanup
    assert not any(tmp_path.glob("**/*")) 
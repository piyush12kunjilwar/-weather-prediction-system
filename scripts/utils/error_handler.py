import time
import logging
from functools import wraps
from typing import Callable, Any, Type, Tuple
from .logger import log_error, log_warning

class RainPredictionError(Exception):
    """Base exception for rain prediction errors."""
    pass

class DataLoadingError(RainPredictionError):
    """Exception raised for errors in data loading."""
    pass

class ModelTrainingError(RainPredictionError):
    """Exception raised for errors in model training."""
    pass

class ValidationError(RainPredictionError):
    """Exception raised for validation errors."""
    pass

class ConfigurationError(RainPredictionError):
    """Exception raised for configuration errors."""
    pass

def retry_on_error(
    max_retries: int = 3,
    delay: int = 5,
    exceptions: Tuple[Type[Exception]] = (Exception,),
    logger: logging.Logger = None
) -> Callable:
    """
    Decorator for retrying a function on error.
    
    Args:
        max_retries (int): Maximum number of retries
        delay (int): Delay between retries in seconds
        exceptions (Tuple[Type[Exception]]): Exceptions to catch
        logger (logging.Logger): Logger instance
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (attempt + 1)
                        if logger:
                            log_warning(
                                logger,
                                f"Attempt {attempt + 1} failed. Retrying in {wait_time} seconds...",
                                f"Function: {func.__name__}"
                            )
                        time.sleep(wait_time)
                    else:
                        if logger:
                            log_error(
                                logger,
                                last_exception,
                                f"Function: {func.__name__}"
                            )
                        raise last_exception
            return None
        return wrapper
    return decorator

def handle_errors(
    exceptions: Tuple[Type[Exception]] = (Exception,),
    logger: logging.Logger = None
) -> Callable:
    """
    Decorator for handling errors in a function.
    
    Args:
        exceptions (Tuple[Type[Exception]]): Exceptions to catch
        logger (logging.Logger): Logger instance
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if logger:
                    log_error(
                        logger,
                        e,
                        f"Function: {func.__name__}"
                    )
                raise
        return wrapper
    return decorator

def validate_input(
    validation_func: Callable,
    error_message: str = "Invalid input",
    logger: logging.Logger = None
) -> Callable:
    """
    Decorator for validating function input.
    
    Args:
        validation_func (Callable): Function to validate input
        error_message (str): Error message if validation fails
        logger (logging.Logger): Logger instance
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not validation_func(*args, **kwargs):
                if logger:
                    log_error(
                        logger,
                        ValueError(error_message),
                        f"Function: {func.__name__}"
                    )
                raise ValueError(error_message)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def handle_error(error, context=None, logger=None):
    """
    Handle errors with appropriate logging and error type.
    
    Args:
        error: The error that occurred
        context: Additional context about where the error occurred
        logger: Logger instance for logging the error
    """
    error_mapping = {
        'data_loading': DataLoadingError,
        'model_training': ModelTrainingError,
        'validation': ValidationError,
        'configuration': ConfigurationError
    }
    
    # Determine error type based on context
    error_type = error_mapping.get(context, Exception)
    
    # Create error message
    error_msg = str(error)
    if context:
        error_msg = f"{context} error: {error_msg}"
    
    # Log error if logger is provided
    if logger:
        logger.error(error_msg)
        logger.exception(error)
    
    # Raise appropriate error
    raise error_type(error_msg)

def validate_data(data, required_fields, context=None, logger=None):
    """
    Validate data against required fields.
    
    Args:
        data: Data to validate
        required_fields: List of required fields
        context: Context for error messages
        logger: Logger instance
    """
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        error_msg = f"Missing required fields: {', '.join(missing_fields)}"
        handle_error(error_msg, context, logger)

def validate_config(config, required_keys, context=None, logger=None):
    """
    Validate configuration against required keys.
    
    Args:
        config: Configuration to validate
        required_keys: List of required keys
        context: Context for error messages
        logger: Logger instance
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        error_msg = f"Missing required configuration keys: {', '.join(missing_keys)}"
        handle_error(error_msg, context, logger) 
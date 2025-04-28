import argparse
import logging
from pathlib import Path
from typing import Dict, Any

from utils.logger import setup_logger
from utils.config_manager import ConfigManager
from utils.error_handler import (
    retry_on_error,
    handle_errors,
    validate_input,
    DataLoadingError,
    ModelTrainingError
)
from data_preparation import prepare_data
from cnn_rnn_model import create_cnn_rnn_model, train_model
from model_evaluation import evaluate_model

def parse_args() -> Dict[str, Any]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Rain Prediction Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'predict'],
                      default='train', help='Pipeline mode')
    parser.add_argument('--data_path', type=str, help='Path to data directory')
    parser.add_argument('--model_path', type=str, help='Path to save/load model')
    return vars(parser.parse_args())

@handle_errors(exceptions=(DataLoadingError,))
@retry_on_error(max_retries=3, delay=5)
def load_and_preprocess_data(config: Dict[str, Any], logger: logging.Logger) -> Any:
    """Load and preprocess data."""
    logger.info("Starting data loading and preprocessing")
    try:
        data = prepare_data(config['data'])
        logger.info("Data loading and preprocessing completed successfully")
        return data
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise DataLoadingError(f"Failed to load data: {str(e)}")

@handle_errors(exceptions=(ModelTrainingError,))
@retry_on_error(max_retries=3, delay=5)
def train_pipeline(config: Dict[str, Any], logger: logging.Logger) -> None:
    """Train the model."""
    logger.info("Starting model training pipeline")
    try:
        # Load and preprocess data
        data = load_and_preprocess_data(config, logger)
        
        # Create model
        model = create_cnn_rnn_model(config['model']['architecture'])
        logger.info("Model created successfully")
        
        # Train model
        history = train_model(
            model,
            data,
            config['model']['training'],
            logger
        )
        logger.info("Model training completed successfully")
        
        # Save model
        model_path = Path(config['paths']['models']) / 'rain_prediction_model.h5'
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise ModelTrainingError(f"Failed to train model: {str(e)}")

@handle_errors()
@retry_on_error(max_retries=3, delay=5)
def evaluate_pipeline(config: Dict[str, Any], logger: logging.Logger) -> None:
    """Evaluate the model."""
    logger.info("Starting model evaluation pipeline")
    try:
        # Load data
        data = load_and_preprocess_data(config, logger)
        
        # Load model
        model_path = Path(config['paths']['models']) / 'rain_prediction_model.h5'
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Evaluate model
        metrics = evaluate_model(model_path, data, logger)
        logger.info("Model evaluation completed successfully")
        
        # Save results
        results_path = Path(config['paths']['results']) / 'evaluation_results.json'
        metrics.save(results_path)
        logger.info(f"Results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        raise

def main() -> None:
    """Main pipeline function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logger
    logger = setup_logger(args['config'])
    
    # Load configuration
    config_manager = ConfigManager(args['config'])
    config = config_manager.get_config()
    
    # Update config with command line arguments
    if args['data_path']:
        config['data']['raw_data_path'] = args['data_path']
    if args['model_path']:
        config['paths']['models'] = args['model_path']
    
    try:
        # Run pipeline based on mode
        if args['mode'] == 'train':
            train_pipeline(config, logger)
        elif args['mode'] == 'evaluate':
            evaluate_pipeline(config, logger)
        else:
            logger.error(f"Invalid mode: {args['mode']}")
            raise ValueError(f"Invalid mode: {args['mode']}")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
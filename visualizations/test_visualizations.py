import numpy as np
from model_visualizer import visualize_predictions
import logging
import os
import sys

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def generate_sample_predictions():
    """Generate sample data for testing visualizations"""
    try:
        logger.info("Generating sample data...")
        
        # Generate sample timestamps (30 days)
        timestamps = np.arange('2024-01-01', '2024-01-31', dtype='datetime64[D]')
        logger.info(f"Generated {len(timestamps)} timestamps")
        
        # Generate sample predictions with correct shape (batch_size, 24, 18)
        np.random.seed(42)
        # Create 30 samples, each with shape (24, 18)
        predictions = np.random.normal(0.5, 0.1, size=(30, 24, 18))
        logger.info(f"Generated predictions with shape: {predictions.shape}")
        
        return timestamps, predictions
    except Exception as e:
        logger.error(f"Error generating sample data: {str(e)}")
        raise

def main():
    try:
        logger.info("Starting visualization test...")
        
        # Generate sample data
        timestamps, predictions = generate_sample_predictions()
        
        # Create a dummy model path (we'll use sample data instead of actual model)
        model_path = "rain_prediction_model_final.h5"
        logger.info(f"Using model path: {model_path}")
        
        # Check if visualizations directory exists
        vis_dir = 'visualizations'
        if not os.path.exists(vis_dir):
            logger.info(f"Creating visualizations directory: {vis_dir}")
            os.makedirs(vis_dir)
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        # For visualization purposes, we'll use the mean prediction across all features
        mean_predictions = np.mean(predictions, axis=(1, 2))
        visualize_predictions(model_path, mean_predictions.reshape(-1, 1), timestamps)
        
        # Verify files were created
        expected_files = [
            'predictions_time_series.jpg',
            'prediction_distribution.jpg',
            'predictions_scatter.jpg'
        ]
        
        for file in expected_files:
            file_path = os.path.join(vis_dir, file)
            if os.path.exists(file_path):
                logger.info(f"Successfully created: {file_path}")
            else:
                logger.error(f"Failed to create: {file_path}")
        
        logger.info("Sample visualizations generation completed!")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
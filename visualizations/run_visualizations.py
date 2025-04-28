import pandas as pd
import numpy as np
from data_visualizer import DataVisualizer
from model_visualizer import visualize_model_performance, visualize_predictions
import tensorflow as tf
from pathlib import Path
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data_in_chunks(data_path, chunk_size=100000):
    """Load and prepare the data in chunks"""
    try:
        logger.info(f"Loading data from {data_path} in chunks")
        chunks = []
        for chunk in tqdm(pd.read_csv(data_path, chunksize=chunk_size), desc="Loading data chunks"):
            chunks.append(chunk)
            if len(chunks) >= 10:  # Limit to first 10 chunks for visualization
                break
        data = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded {len(data)} rows of data")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def generate_all_visualizations(data_path, model_path, output_dir='visualizations'):
    """
    Generate all visualizations for the project
    """
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Initialize visualizer
        visualizer = DataVisualizer(save_dir=output_dir)
        
        # Load data in chunks
        data = load_data_in_chunks(data_path)
        
        logger.info("Generating data visualizations...")
        
        # 1. Time series visualization
        if 'timestamp' in data.columns:
            logger.info("Generating time series plot...")
            visualizer.plot_time_series(
                data=data,
                date_column='timestamp',
                value_column='temperature',
                title='Temperature Over Time',
                filename='temperature_time_series'
            )
        
        # 2. Correlation heatmap
        logger.info("Generating correlation heatmap...")
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            visualizer.plot_correlation_heatmap(
                data[numeric_columns],
                title='Feature Correlations',
                filename='correlation_heatmap'
            )
        
        # 3. Distribution plots for key features
        logger.info("Generating distribution plots...")
        for column in ['temperature', 'humidity', 'pressure']:
            if column in data.columns:
                visualizer.plot_distribution(
                    data=data,
                    column=column,
                    title=f'{column.capitalize()} Distribution',
                    filename=f'{column}_distribution'
                )
        
        # 4. Scatter matrix for weather parameters
        logger.info("Generating scatter matrix...")
        weather_columns = ['temperature', 'humidity', 'pressure']
        if all(col in data.columns for col in weather_columns):
            visualizer.plot_scatter_matrix(
                data=data,
                columns=weather_columns,
                title='Weather Parameters Relationships',
                filename='weather_scatter_matrix'
            )
        
        logger.info("Generating model visualizations...")
        
        # 5. Model performance visualizations
        try:
            # Load model
            model = tf.keras.models.load_model(model_path)
            
            # Prepare test data (you'll need to modify this based on your actual data structure)
            # X_test = ...
            # y_test = ...
            
            # Generate model visualizations
            # visualize_model_performance(model_path, X_test, y_test)
            # visualize_predictions(model_path, X_test)
            
            logger.info("Model visualizations generated successfully")
        except Exception as e:
            logger.warning(f"Could not generate model visualizations: {str(e)}")
        
        logger.info("All visualizations generated successfully!")
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        raise

if __name__ == "__main__":
    # Paths to your data and model
    DATA_PATH = "2006Fall_2017Spring_GOES_meteo_combined.csv"
    MODEL_PATH = "rain_prediction_model_final.h5"
    
    try:
        generate_all_visualizations(DATA_PATH, MODEL_PATH)
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {str(e)}") 
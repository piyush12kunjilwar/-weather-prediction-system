import os
import yaml
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import json
import cv2
from datetime import datetime

from utils.logger import setup_logger
from utils.error_handler import handle_error

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        handle_error(e, 'configuration')

def load_model(config, logger):
    """Load trained model."""
    try:
        model_path = os.path.join(
            config['paths']['models'],
            'best_model.h5'
        )
        model = tf.keras.models.load_model(model_path)
        logger.info("Loaded trained model")
        return model
    except Exception as e:
        handle_error(e, 'model_loading', logger)

def load_normalization_params(config, logger):
    """Load normalization parameters."""
    try:
        params_path = os.path.join(
            config['data']['processed_data_path'],
            'meteorological',
            'normalization_params.json'
        )
        with open(params_path, 'r') as f:
            params = json.load(f)
        logger.info("Loaded normalization parameters")
        return params
    except Exception as e:
        handle_error(e, 'params_loading', logger)

def preprocess_satellite_image(image_path, config, logger):
    """Preprocess satellite image."""
    try:
        # Load image
        img = cv2.imread(image_path)
        
        # Resize image
        img = cv2.resize(img, config['data']['satellite_images']['image_size'])
        
        # Normalize image
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        logger.info(f"Preprocessed satellite image: {image_path}")
        
        return img
    except Exception as e:
        handle_error(e, 'image_preprocessing', logger)

def preprocess_meteorological_data(data, params, config, logger):
    """Preprocess meteorological data."""
    try:
        # Normalize features
        for feature in config['data']['meteorological']['features']:
            mean = params['means'][feature]
            std = params['stds'][feature]
            data[feature] = (data[feature] - mean) / std
        
        # Convert to numpy array
        data = data[config['data']['meteorological']['features']].values
        
        # Add batch dimension
        data = np.expand_dims(data, axis=0)
        
        logger.info("Preprocessed meteorological data")
        
        return data
    except Exception as e:
        handle_error(e, 'meteorological_preprocessing', logger)

def make_prediction(model, satellite_data, meteo_data, config, logger):
    """Make prediction using the model."""
    try:
        # Make prediction
        prediction = model.predict([satellite_data, meteo_data])
        
        # Convert to probability
        probability = float(prediction[0][0])
        
        # Convert to binary prediction
        threshold = config['prediction']['threshold']
        binary_prediction = 1 if probability >= threshold else 0
        
        logger.info(f"Prediction probability: {probability:.4f}")
        logger.info(f"Binary prediction: {binary_prediction}")
        
        return probability, binary_prediction
    except Exception as e:
        handle_error(e, 'prediction', logger)

def save_prediction(image_path, meteo_data, probability, binary_prediction, config, logger):
    """Save prediction results."""
    try:
        # Create results directory
        results_dir = os.path.join(
            config['paths']['predictions'],
            datetime.now().strftime('%Y%m%d')
        )
        os.makedirs(results_dir, exist_ok=True)
        
        # Create results dictionary
        results = {
            'image_path': image_path,
            'meteorological_data': meteo_data.to_dict(),
            'probability': probability,
            'binary_prediction': binary_prediction,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        results_path = os.path.join(
            results_dir,
            f"prediction_{datetime.now().strftime('%H%M%S')}.json"
        )
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Saved prediction results to {results_path}")
    except Exception as e:
        handle_error(e, 'results_saving', logger)

def visualize_prediction(image_path, probability, binary_prediction, config, logger):
    """Visualize prediction results."""
    try:
        # Create visualizations directory
        viz_dir = os.path.join(
            config['paths']['visualizations'],
            'predictions',
            datetime.now().strftime('%Y%m%d')
        )
        os.makedirs(viz_dir, exist_ok=True)
        
        # Load and resize image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (800, 600))
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot image
        plt.imshow(img)
        
        # Add prediction text
        prediction_text = f"Flood Probability: {probability:.4f}\nPrediction: {'Flood' if binary_prediction else 'No Flood'}"
        plt.text(
            10, 30,
            prediction_text,
            color='white',
            fontsize=12,
            bbox=dict(
                facecolor='black',
                alpha=0.5,
                edgecolor='none'
            )
        )
        
        # Save visualization
        viz_path = os.path.join(
            viz_dir,
            f"prediction_{datetime.now().strftime('%H%M%S')}.png"
        )
        plt.savefig(viz_path)
        plt.close()
        
        logger.info(f"Saved prediction visualization to {viz_path}")
    except Exception as e:
        handle_error(e, 'visualization', logger)

def main():
    """Main function to make predictions."""
    try:
        # Setup
        config = load_config()
        logger = setup_logger('prediction')
        
        # Load model
        model = load_model(config, logger)
        
        # Load normalization parameters
        params = load_normalization_params(config, logger)
        
        # Example satellite image path
        image_path = os.path.join(
            config['data']['raw_data_path'],
            'satellite_images',
            'example.png'
        )
        
        # Example meteorological data
        meteo_data = pd.DataFrame({
            'temperature': [25.0],
            'humidity': [80.0],
            'precipitation': [50.0],
            'wind_speed': [15.0]
        })
        
        # Preprocess data
        satellite_data = preprocess_satellite_image(image_path, config, logger)
        meteo_data = preprocess_meteorological_data(meteo_data, params, config, logger)
        
        # Make prediction
        probability, binary_prediction = make_prediction(
            model,
            satellite_data,
            meteo_data,
            config,
            logger
        )
        
        # Save results
        save_prediction(
            image_path,
            meteo_data,
            probability,
            binary_prediction,
            config,
            logger
        )
        
        # Visualize results
        visualize_prediction(
            image_path,
            probability,
            binary_prediction,
            config,
            logger
        )
        
        logger.info("Prediction completed successfully")
        
    except Exception as e:
        handle_error(e, 'main', logger)

if __name__ == "__main__":
    main() 
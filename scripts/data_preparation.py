import os
import yaml
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import glob
from datetime import datetime, timedelta

from utils.logger import setup_logger
from utils.error_handler import handle_error, validate_data

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        handle_error(e, 'configuration')

def create_directories(config):
    """Create necessary directories."""
    try:
        for path in [config['data']['raw_data_path'],
                    config['data']['processed_data_path'],
                    config['data']['cache_path']]:
            os.makedirs(path, exist_ok=True)
    except Exception as e:
        handle_error(e, 'directory_creation')

def load_satellite_images(config, logger):
    """Load and preprocess satellite images."""
    try:
        # Get image paths
        image_paths = sorted(glob.glob(os.path.join(
            config['data']['raw_data_path'],
            'satellite_images',
            '*.png'
        )))
        
        if not image_paths:
            raise ValueError("No satellite images found in the specified directory")
        
        logger.info(f"Found {len(image_paths)} satellite images")
        
        # Load and preprocess images
        images = []
        for path in tqdm(image_paths, desc="Loading satellite images"):
            img = Image.open(path).convert('L')  # Convert to grayscale
            img = img.resize(config['data']['satellite_images']['image_size'])
            img = np.array(img) / 255.0  # Normalize to [0, 1]
            images.append(img)
        
        images = np.array(images)
        logger.info(f"Loaded satellite images with shape: {images.shape}")
        
        return images
    except Exception as e:
        handle_error(e, 'satellite_image_loading', logger)

def load_meteorological_data(config, logger):
    """Load and preprocess meteorological data."""
    try:
        # Load data
        meteo_path = os.path.join(
            config['data']['raw_data_path'],
            'meteorological_data.csv'
        )
        df = pd.read_csv(meteo_path)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Select features
        features = config['data']['meteorological']['features']
        if not all(f in df.columns for f in features):
            raise ValueError("Not all specified meteorological features found in data")
        
        # Normalize features
        for feature in features:
            df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()
        
        logger.info(f"Loaded meteorological data with shape: {df[features].shape}")
        
        return df[['timestamp'] + features]
    except Exception as e:
        handle_error(e, 'meteorological_data_loading', logger)

def create_sequences(images, meteo_data, config, logger):
    """Create sequences of satellite images and meteorological data."""
    try:
        sequence_length = config['data']['satellite_images']['sequence_length']
        
        # Create satellite sequences
        satellite_sequences = []
        for i in range(len(images) - sequence_length + 1):
            sequence = images[i:i + sequence_length]
            satellite_sequences.append(sequence)
        
        satellite_sequences = np.array(satellite_sequences)
        logger.info(f"Created satellite sequences with shape: {satellite_sequences.shape}")
        
        # Create meteorological sequences
        meteo_sequences = []
        for i in range(len(meteo_data) - sequence_length + 1):
            sequence = meteo_data.iloc[i:i + sequence_length, 1:].values
            meteo_sequences.append(sequence)
        
        meteo_sequences = np.array(meteo_sequences)
        logger.info(f"Created meteorological sequences with shape: {meteo_sequences.shape}")
        
        return satellite_sequences, meteo_sequences
    except Exception as e:
        handle_error(e, 'sequence_creation', logger)

def save_processed_data(satellite_sequences, meteo_sequences, config, logger):
    """Save processed data to disk."""
    try:
        # Save satellite sequences
        satellite_path = os.path.join(
            config['data']['processed_data_path'],
            'satellite_images',
            f"sequences_{config['data']['satellite_images']['sequence_length']}.npy"
        )
        os.makedirs(os.path.dirname(satellite_path), exist_ok=True)
        np.save(satellite_path, satellite_sequences)
        logger.info(f"Saved satellite sequences to {satellite_path}")
        
        # Save meteorological data
        meteo_path = os.path.join(
            config['data']['processed_data_path'],
            'meteorological_data.csv'
        )
        pd.DataFrame(meteo_sequences.reshape(
            meteo_sequences.shape[0],
            meteo_sequences.shape[1] * meteo_sequences.shape[2]
        )).to_csv(meteo_path, index=False)
        logger.info(f"Saved meteorological data to {meteo_path}")
    except Exception as e:
        handle_error(e, 'data_saving', logger)

def main():
    """Main function to prepare data for training."""
    try:
        # Setup
        config = load_config()
        logger = setup_logger('data_preparation')
        create_directories(config)
        
        # Load data
        images = load_satellite_images(config, logger)
        meteo_data = load_meteorological_data(config, logger)
        
        # Create sequences
        satellite_sequences, meteo_sequences = create_sequences(
            images, meteo_data, config, logger
        )
        
        # Save processed data
        save_processed_data(satellite_sequences, meteo_sequences, config, logger)
        
        logger.info("Data preparation completed successfully")
        
    except Exception as e:
        handle_error(e, 'main', logger)

if __name__ == "__main__":
    main() 
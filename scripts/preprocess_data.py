import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import cv2
from tqdm import tqdm

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

def create_sequences(data, sequence_length, step=1):
    """Create sequences from time series data."""
    sequences = []
    for i in range(0, len(data) - sequence_length + 1, step):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

def preprocess_satellite_images(config, logger):
    """Preprocess satellite images."""
    try:
        # Get configuration parameters
        input_dir = config['data']['raw_data_path']
        output_dir = os.path.join(
            config['data']['processed_data_path'],
            'satellite_images'
        )
        sequence_length = config['data']['satellite_images']['sequence_length']
        image_size = config['data']['satellite_images']['image_size']
        channels = config['data']['satellite_images']['channels']
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of image files
        image_files = sorted([
            f for f in os.listdir(input_dir)
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        # Load and preprocess images
        images = []
        for image_file in tqdm(image_files, desc="Processing satellite images"):
            # Load image
            image_path = os.path.join(input_dir, image_file)
            image = cv2.imread(image_path)
            
            # Resize image
            image = cv2.resize(image, image_size)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            images.append(image)
        
        # Convert to numpy array
        images = np.array(images)
        
        # Create sequences
        sequences = create_sequences(images, sequence_length)
        
        # Save sequences
        output_path = os.path.join(
            output_dir,
            f"sequences_{sequence_length}.npy"
        )
        np.save(output_path, sequences)
        
        logger.info(f"Saved {len(sequences)} satellite sequences to {output_path}")
        logger.info(f"Sequence shape: {sequences.shape}")
        
    except Exception as e:
        handle_error(e, 'satellite_image_preprocessing', logger)

def preprocess_meteorological_data(config, logger):
    """Preprocess meteorological data."""
    try:
        # Get configuration parameters
        input_path = os.path.join(
            config['data']['raw_data_path'],
            'meteorological_data.csv'
        )
        output_dir = os.path.join(
            config['data']['processed_data_path'],
            'meteorological'
        )
        sequence_length = config['data']['satellite_images']['sequence_length']
        features = config['data']['meteorological']['features']
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        df = pd.read_csv(input_path)
        
        # Select features
        df = df[features]
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Normalize data
        df = (df - df.mean()) / df.std()
        
        # Convert to numpy array
        data = df.values
        
        # Create sequences
        sequences = create_sequences(data, sequence_length)
        
        # Save sequences
        output_path = os.path.join(output_dir, 'sequences.npy')
        np.save(output_path, sequences)
        
        logger.info(f"Saved {len(sequences)} meteorological sequences to {output_path}")
        logger.info(f"Sequence shape: {sequences.shape}")
        
    except Exception as e:
        handle_error(e, 'meteorological_data_preprocessing', logger)

def main():
    """Main function to preprocess data."""
    try:
        # Setup
        config = load_config()
        logger = setup_logger('preprocessing')
        
        # Preprocess satellite images
        preprocess_satellite_images(config, logger)
        
        # Preprocess meteorological data
        preprocess_meteorological_data(config, logger)
        
        logger.info("Data preprocessing completed successfully")
        
    except Exception as e:
        handle_error(e, 'main', logger)

if __name__ == "__main__":
    main() 
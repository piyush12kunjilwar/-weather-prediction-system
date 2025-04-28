import os
import yaml
import numpy as np
import pandas as pd
import cv2
import json
from datetime import datetime
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

def preprocess_satellite_images(config, logger):
    """Preprocess satellite images."""
    try:
        # Get parameters
        raw_dir = os.path.join(
            config['data']['raw_data_path'],
            'satellite_images'
        )
        processed_dir = os.path.join(
            config['data']['processed_data_path'],
            'satellite_images'
        )
        os.makedirs(processed_dir, exist_ok=True)
        
        image_size = config['data']['satellite_images']['image_size']
        sequence_length = config['data']['satellite_images']['sequence_length']
        
        # Get image files
        image_files = sorted([
            f for f in os.listdir(raw_dir)
            if f.endswith('.png')
        ])
        
        # Create sequences
        sequences = []
        for i in range(len(image_files) - sequence_length + 1):
            sequence = []
            for j in range(sequence_length):
                img_path = os.path.join(raw_dir, image_files[i + j])
                img = cv2.imread(img_path)
                img = cv2.resize(img, image_size)
                img = img.astype(np.float32) / 255.0
                sequence.append(img)
            sequences.append(sequence)
        
        sequences = np.array(sequences)
        
        # Save sequences
        sequences_path = os.path.join(
            processed_dir,
            f"sequences_{sequence_length}.npy"
        )
        np.save(sequences_path, sequences)
        
        logger.info(f"Preprocessed {len(sequences)} satellite image sequences")
        logger.info(f"Sequence shape: {sequences.shape}")
        logger.info(f"Saved sequences to {sequences_path}")
        
    except Exception as e:
        handle_error(e, 'satellite_preprocessing', logger)

def preprocess_meteorological_data(config, logger):
    """Preprocess meteorological data."""
    try:
        # Get parameters
        raw_path = os.path.join(
            config['data']['raw_data_path'],
            'meteorological',
            'data.csv'
        )
        processed_dir = os.path.join(
            config['data']['processed_data_path'],
            'meteorological'
        )
        os.makedirs(processed_dir, exist_ok=True)
        
        sequence_length = config['data']['meteorological']['sequence_length']
        features = config['data']['meteorological']['features']
        
        # Load data
        df = pd.read_csv(raw_path)
        
        # Normalize features
        for feature in features:
            mean = df[feature].mean()
            std = df[feature].std()
            df[feature] = (df[feature] - mean) / std
        
        # Create sequences
        sequences = []
        for i in range(len(df) - sequence_length + 1):
            sequence = df[features].iloc[i:i + sequence_length].values
            sequences.append(sequence)
        
        sequences = np.array(sequences)
        
        # Save sequences
        sequences_path = os.path.join(processed_dir, 'sequences.npy')
        np.save(sequences_path, sequences)
        
        # Save normalization parameters
        norm_params = {
            'means': {f: df[f].mean() for f in features},
            'stds': {f: df[f].std() for f in features}
        }
        norm_params_path = os.path.join(processed_dir, 'normalization_params.json')
        with open(norm_params_path, 'w') as f:
            json.dump(norm_params, f, indent=4)
        
        logger.info(f"Preprocessed {len(sequences)} meteorological sequences")
        logger.info(f"Sequence shape: {sequences.shape}")
        logger.info(f"Saved sequences to {sequences_path}")
        logger.info(f"Saved normalization parameters to {norm_params_path}")
        
    except Exception as e:
        handle_error(e, 'meteorological_preprocessing', logger)

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
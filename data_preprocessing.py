import os
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def load_meteorological_data(file_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and preprocess meteorological data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing meteorological data
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: Preprocessed data and list of feature names
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Convert timestamp column to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Get feature columns (excluding timestamp)
        feature_columns = [col for col in df.columns if col != 'timestamp']
        
        # Handle missing values using ffill and bfill
        df = df.ffill().bfill()
        
        # Normalize numerical features
        scaler = MinMaxScaler()
        df[feature_columns] = scaler.fit_transform(df[feature_columns])
        
        return df, feature_columns
        
    except Exception as e:
        raise Exception(f"Error loading meteorological data: {str(e)}")

def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int]=(128, 128)) -> np.ndarray:
    """
    Load and preprocess a single satellite image.
    
    Args:
        image_path (str): Path to the image file
        target_size (Tuple[int, int]): Target size for image resizing
        
    Returns:
        np.ndarray: Preprocessed image array
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, target_size)
        
        # Normalize pixel values to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
        
    except Exception as e:
        raise Exception(f"Error processing image {image_path}: {str(e)}")

def extract_timestamp_from_filename(filename: str) -> datetime:
    """
    Extract timestamp from image filename.
    Expected format: YYYY-MM-DD_HH-MM.png
    
    Args:
        filename (str): Name of the image file
        
    Returns:
        datetime: Extracted timestamp
    """
    try:
        # Remove file extension
        filename = os.path.splitext(filename)[0]
        # Parse datetime from filename
        return datetime.strptime(filename, '%Y-%m-%d_%H-%M')
    except Exception as e:
        raise Exception(f"Error extracting timestamp from filename {filename}: {str(e)}")

def create_sequences(images: Dict[datetime, np.ndarray],
                    meteo_data: pd.DataFrame,
                    seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for training from aligned image and meteorological data.
    
    Args:
        images (Dict[datetime, np.ndarray]): Dictionary of timestamps to preprocessed images
        meteo_data (pd.DataFrame): Preprocessed meteorological data
        seq_length (int): Length of sequences to create
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Image sequences and corresponding meteorological sequences
    """
    try:
        # Get common timestamps between images and meteorological data
        common_timestamps = sorted(list(set(images.keys()) & set(meteo_data.index)))
        
        if len(common_timestamps) < seq_length:
            raise ValueError(f"Not enough common timestamps to create sequences of length {seq_length}")
        
        image_sequences = []
        meteo_sequences = []
        
        # Create sequences
        for i in range(len(common_timestamps) - seq_length + 1):
            seq_timestamps = common_timestamps[i:i + seq_length]
            
            # Create image sequence
            img_seq = np.stack([images[ts] for ts in seq_timestamps])
            image_sequences.append(img_seq)
            
            # Create meteorological sequence
            meteo_seq = meteo_data.loc[seq_timestamps].values
            meteo_sequences.append(meteo_seq)
        
        return np.array(image_sequences), np.array(meteo_sequences)
        
    except Exception as e:
        raise Exception(f"Error creating sequences: {str(e)}")

def prepare_data(meteo_file: str,
                image_dir: str,
                seq_length: int,
                target_size: Tuple[int, int]=(128, 128),
                val_split: float=0.2) -> Tuple[Tuple[Dict, np.ndarray], Tuple[Dict, np.ndarray], List[str]]:
    """
    Main function to prepare data for the ConvLSTM model.
    
    Args:
        meteo_file (str): Path to meteorological data CSV file
        image_dir (str): Directory containing satellite images
        seq_length (int): Length of sequences to create
        target_size (Tuple[int, int]): Target size for image resizing
        val_split (float): Fraction of data to use for validation
        
    Returns:
        Tuple[Tuple[Dict, np.ndarray], Tuple[Dict, np.ndarray], List[str]]:
            Training data (images, meteo), validation data (images, meteo), and feature names
    """
    try:
        # Load meteorological data
        meteo_data, feature_names = load_meteorological_data(meteo_file)
        print(f"Loaded meteorological data with {len(feature_names)} features")
        
        # Load and preprocess images
        images = {}
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        print("Processing images...")
        for img_file in tqdm(image_files):
            img_path = os.path.join(image_dir, img_file)
            timestamp = extract_timestamp_from_filename(img_file)
            images[timestamp] = load_and_preprocess_image(img_path, target_size)
        
        # Create sequences
        image_sequences, meteo_sequences = create_sequences(images, meteo_data, seq_length)
        
        # Split into training and validation sets
        n_samples = len(image_sequences)
        n_val = int(n_samples * val_split)
        indices = np.random.permutation(n_samples)
        
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        train_data = (image_sequences[train_idx], meteo_sequences[train_idx])
        val_data = (image_sequences[val_idx], meteo_sequences[val_idx])
        
        print(f"Training set size: {len(train_idx)}")
        print(f"Validation set size: {len(val_idx)}")
        
        return train_data, val_data, feature_names
        
    except Exception as e:
        raise Exception(f"Error preparing data: {str(e)}")

if __name__ == "__main__":
    try:
        # Example usage
        meteo_file = "path/to/meteorological_data.csv"
        image_dir = "path/to/satellite_images"
        seq_length = 10
        
        train_data, val_data, features = prepare_data(
            meteo_file=meteo_file,
            image_dir=image_dir,
            seq_length=seq_length
        )
        
        print("Data preparation completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}") 
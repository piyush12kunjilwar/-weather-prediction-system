import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
from typing import Tuple, List, Dict
import glob

class DataLoader:
    def __init__(
        self,
        data_dir: str,
        image_dir: str,
        meteo_file: str,
        image_size: Tuple[int, int] = (128, 128),
        image_channels: int = 1
    ):
        """
        Initialize the DataLoader
        
        Args:
            data_dir: Base directory containing all data
            image_dir: Directory containing satellite images
            meteo_file: Path to meteorological data CSV
            image_size: Target size for satellite images
            image_channels: Number of image channels
        """
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, image_dir)
        self.meteo_file = os.path.join(data_dir, meteo_file)
        self.image_size = image_size
        self.image_channels = image_channels
        
        # Initialize scalers
        self.meteo_scaler = MinMaxScaler()
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess all data"""
        print("Loading meteorological data...")
        self.meteo_data = pd.read_csv(self.meteo_file)
        
        # Convert timestamp to datetime if it's not already
        self.meteo_data['timestamp'] = pd.to_datetime(self.meteo_data['timestamp'])
        
        # Sort by timestamp
        self.meteo_data = self.meteo_data.sort_values('timestamp')
        
        # Load rain labels
        self.rain_labels = np.load(os.path.join(self.data_dir, 'processed/rain_labels.npy'))
        
        print("Loading satellite images...")
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))
        
        # Verify data alignment
        self._verify_data_alignment()
    
    def _verify_data_alignment(self):
        """Verify that meteorological data and images are properly aligned"""
        print("\nVerifying data alignment...")
        
        # Get timestamps from image filenames
        image_timestamps = []
        for f in self.image_files:
            # Extract timestamp from filename (YYYY-MM-DD_HH-MM.png)
            filename = os.path.basename(f)
            # Convert YYYY-MM-DD_HH-MM to YYYY-MM-DD HH:MM:00
            date_part, time_part = filename.replace('.png', '').split('_')
            timestamp_str = f"{date_part} {time_part.replace('-', ':')}:00"
            image_timestamps.append(pd.to_datetime(timestamp_str))
        
        # Get timestamps from meteorological data and convert to same format
        meteo_timestamps = []
        for ts in self.meteo_data['timestamp'].values:
            # Convert to datetime and format to match image timestamps
            dt = pd.to_datetime(ts)
            meteo_timestamps.append(pd.to_datetime(f"{dt.strftime('%Y-%m-%d %H:%M')}:00"))
        
        # Print detailed information about timestamps
        print("\nTimestamp ranges:")
        print(f"Meteorological data: {meteo_timestamps[0]} to {meteo_timestamps[-1]}")
        print(f"Image files: {image_timestamps[0]} to {image_timestamps[-1]}")
        
        print("\nNumber of timestamps:")
        print(f"Meteorological data: {len(meteo_timestamps)}")
        print(f"Image files: {len(image_timestamps)}")
        
        # Convert to sets for comparison
        image_timestamps_set = set(image_timestamps)
        meteo_timestamps_set = set(meteo_timestamps)
        
        # Find common timestamps
        common_timestamps = image_timestamps_set & meteo_timestamps_set
        
        if len(common_timestamps) == 0:
            print("\nNo matching timestamps found. Sample timestamps:")
            print("\nMeteorological data (first 5):")
            for ts in meteo_timestamps[:5]:
                print(f"  {ts}")
            print("\nImage files (first 5):")
            for ts in image_timestamps[:5]:
                print(f"  {ts}")
            raise ValueError("No matching timestamps found between images and meteorological data")
        
        print(f"\nFound {len(common_timestamps)} matching timestamps")
        
        # Print some sample matching timestamps
        print("\nSample matching timestamps:")
        for ts in sorted(list(common_timestamps))[:5]:
            print(f"  {ts}")
        
        # Store the common timestamps for later use
        self.common_timestamps = sorted(list(common_timestamps))
        
        # Create a mapping from timestamp to index in meteorological data
        self.timestamp_to_meteo_idx = {
            ts: self.meteo_data[self.meteo_data['timestamp'] == ts].index[0]
            for ts in self.common_timestamps
        }
    
    def create_sequences(
        self,
        meteo_window: int = 24,
        image_window: int = 8,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences for training
        
        Args:
            meteo_window: Number of timesteps for meteorological data
            image_window: Number of timesteps for satellite images
            stride: Step size for sequence creation
        
        Returns:
            Tuple of (image_sequences, meteo_sequences, labels)
        """
        print(f"Creating sequences with meteo_window={meteo_window}, image_window={image_window}")
        
        # Use only the common timestamps
        timestamps = self.common_timestamps
        
        # Initialize lists for sequences
        image_sequences = []
        meteo_sequences = []
        labels = []
        
        # Get feature columns (excluding timestamp)
        feature_columns = [col for col in self.meteo_data.columns if col != 'timestamp']
        
        # Create sequences
        for i in tqdm(range(0, len(timestamps) - max(meteo_window, image_window), stride)):
            # Get timestamps for this sequence
            seq_timestamps = timestamps[i:i + max(meteo_window, image_window)]
            
            # Get corresponding meteorological data
            meteo_indices = [self.timestamp_to_meteo_idx[ts] for ts in seq_timestamps]
            meteo_seq = self.meteo_data.iloc[meteo_indices][feature_columns].values
            
            # Get corresponding images
            image_seq = []
            for ts in seq_timestamps:
                img_path = os.path.join(self.image_dir, f"{ts.strftime('%Y-%m-%d_%H-%M')}.png")
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, self.image_size)
                image_seq.append(img)
            
            # Get label (rain or no rain)
            label = self.rain_labels[i + max(meteo_window, image_window) - 1]
            
            # Append sequences
            image_sequences.append(image_seq)
            meteo_sequences.append(meteo_seq)
            labels.append(label)
        
        # Convert to numpy arrays
        image_sequences = np.array(image_sequences)
        meteo_sequences = np.array(meteo_sequences)
        labels = np.array(labels)
        
        # Normalize meteorological data (excluding timestamp)
        meteo_sequences = self.meteo_scaler.fit_transform(
            meteo_sequences.reshape(-1, meteo_sequences.shape[-1])
        ).reshape(meteo_sequences.shape)
        
        # Normalize images
        image_sequences = image_sequences / 255.0
        
        return image_sequences, meteo_sequences, labels
    
    def split_data(
        self,
        image_sequences: np.ndarray,
        meteo_sequences: np.ndarray,
        labels: np.ndarray,
        val_split: float = 0.2,
        test_split: float = 0.1,
        random_state: int = 42
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Split data into train, validation, and test sets
        
        Args:
            image_sequences: Image sequences
            meteo_sequences: Meteorological sequences
            labels: Labels
            val_split: Validation set size
            test_split: Test set size
            random_state: Random seed
        
        Returns:
            Dictionary containing train, validation, and test sets
        """
        # First split into train+val and test
        train_val_idx, test_idx = train_test_split(
            range(len(labels)),
            test_size=test_split,
            random_state=random_state,
            stratify=labels
        )
        
        # Then split train+val into train and val
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_split/(1-test_split),
            random_state=random_state,
            stratify=labels[train_val_idx]
        )
        
        # Create splits
        splits = {
            'train': (
                image_sequences[train_idx],
                meteo_sequences[train_idx],
                labels[train_idx]
            ),
            'val': (
                image_sequences[val_idx],
                meteo_sequences[val_idx],
                labels[val_idx]
            ),
            'test': (
                image_sequences[test_idx],
                meteo_sequences[test_idx],
                labels[test_idx]
            )
        }
        
        # Print split sizes
        for split_name, (_, _, split_labels) in splits.items():
            print(f"{split_name} set size: {len(split_labels)}")
            print(f"{split_name} set positive samples: {np.sum(split_labels)}")
        
        return splits

def main():
    # Example usage
    data_loader = DataLoader(
        data_dir='data',
        image_dir='processed/satellite_images',
        meteo_file='processed/meteorological_data.csv'
    )
    
    # Create sequences with different window sizes
    image_sequences, meteo_sequences, labels = data_loader.create_sequences(
        meteo_window=24,
        image_window=8
    )
    
    # Split data
    splits = data_loader.split_data(
        image_sequences,
        meteo_sequences,
        labels
    )
    
    # Print shapes
    for split_name, (images, meteo, labels) in splits.items():
        print(f"\n{split_name} set shapes:")
        print(f"Images: {images.shape}")
        print(f"Meteorological: {meteo.shape}")
        print(f"Labels: {labels.shape}")

if __name__ == '__main__':
    main() 
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
import pandas as pd
from pathlib import Path

def create_directories():
    """Create necessary directories for data and cache."""
    os.makedirs('data/cache', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

def load_meteo_data():
    """Load and preprocess meteorological data."""
    print("Loading meteorological data...")
    df = pd.read_csv("../data/2006Fall_2017Spring_GOES_meteo_combined.csv")
    
    # Convert numeric columns
    numeric_cols = ["Temp (F)", "RH (%)", "Wind Spd (mph)", "Precip (in)"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Create timestamp
    df["timestamp"] = pd.to_datetime(df["Date_UTC"] + " " + df["Time_UTC"])
    
    # Create binary target for precipitation
    df["has_precip"] = (df["Precip (in)"] > 0).astype(int)
    
    return df

def process_lake_data(df):
    """Process lake data into proper tensors."""
    print("Processing lake data...")
    
    # Initialize empty list for lake tensors
    lake_tensors = []
    
    # Process each lake data point
    for idx, row in df.iterrows():
        # Extract lake data
        lake_data = np.array([row["Temp (F)"], row["RH (%)"], row["Wind Spd (mph)"]])
        
        # Handle NaN values
        lake_data = np.nan_to_num(lake_data, nan=0.0)
        
        # Validate dimensions
        if lake_data.shape != (3,):
            print(f"Warning: Invalid dimensions at index {idx}")
            continue
            
        lake_tensors.append(lake_data)
    
    # Convert to numpy array
    lake_tensors = np.array(lake_tensors)
    
    # Reshape for CNN input (samples, height, width, channels)
    lake_tensors = lake_tensors.reshape(-1, 1, 1, 3)
    
    # Normalize the data
    mean = np.mean(lake_tensors, axis=0)
    std = np.std(lake_tensors, axis=0)
    lake_tensors = (lake_tensors - mean) / (std + 1e-8)  # Add small epsilon to avoid division by zero
    
    return lake_tensors, mean, std

def create_sequences(lake_tensors, targets, sequence_length=24):
    """Create time series sequences for training."""
    print("Creating sequences...")
    
    # Calculate the number of sequences
    num_sequences = len(lake_tensors) - sequence_length + 1
    
    # Create sequences using timeseries_dataset_from_array
    dataset = timeseries_dataset_from_array(
        data=lake_tensors,
        targets=targets[sequence_length-1:],  # Align targets with the last element of each sequence
        sequence_length=sequence_length,
        batch_size=32,
        shuffle=True
    )
    
    return dataset, num_sequences

def cache_data(dataset, metadata):
    """Cache processed data for faster loading."""
    print("Caching data...")
    
    # Save metadata
    np.save('data/cache/metadata.npy', metadata)
    
    # Save tensors
    for i, (x, y) in enumerate(dataset):
        np.save(f'data/cache/batch_{i}_x.npy', x.numpy())
        np.save(f'data/cache/batch_{i}_y.npy', y.numpy())

def load_cached_data():
    """Load cached data if it exists."""
    if not os.path.exists('data/cache/metadata.npy'):
        return None, None
    
    print("Loading cached data...")
    
    # Load metadata
    metadata = np.load('data/cache/metadata.npy', allow_pickle=True).item()
    
    # Load tensors
    x_batches = []
    y_batches = []
    
    i = 0
    while os.path.exists(f'data/cache/batch_{i}_x.npy'):
        x_batches.append(np.load(f'data/cache/batch_{i}_x.npy'))
        y_batches.append(np.load(f'data/cache/batch_{i}_y.npy'))
        i += 1
    
    x = np.concatenate(x_batches)
    y = np.concatenate(y_batches)
    
    return x, y, metadata

def main():
    """Main function to process satellite images and implement caching."""
    # Create directories
    create_directories()
    
    # Try to load cached data first
    cached_data = load_cached_data()
    if cached_data[0] is not None:
        print("Using cached data")
        return cached_data
    
    # Load and preprocess data
    df = load_meteo_data()
    lake_tensors, mean, std = process_lake_data(df)
    targets = df["has_precip"].values
    
    # Create sequences
    dataset, num_sequences = create_sequences(lake_tensors, targets)
    
    # Cache the data
    metadata = {
        "num_samples": num_sequences,
        "sequence_length": 24,
        "input_shape": lake_tensors.shape[1:],
        "target_shape": (1,),
        "normalization_mean": mean,
        "normalization_std": std
    }
    cache_data(dataset, metadata)
    
    return dataset, metadata

if __name__ == "__main__":
    main() 
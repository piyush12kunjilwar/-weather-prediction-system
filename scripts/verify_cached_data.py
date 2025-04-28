import numpy as np
import os
import pandas as pd
import sys
sys.path.append('.')  # Add current directory to Python path
from scripts.process_satellite_images import load_meteo_data, process_lake_data

def verify_cached_data():
    """Verify the cached data is correct."""
    print("Verifying cached data...")
    
    # Check if cache exists
    if not os.path.exists('data/cache/metadata.npy'):
        print("Error: Cache directory or metadata not found")
        return False
    
    # Load metadata
    metadata = np.load('data/cache/metadata.npy', allow_pickle=True).item()
    print("\nMetadata:")
    for key, value in metadata.items():
        if key not in ['normalization_mean', 'normalization_std']:  # Skip printing large arrays
            print(f"{key}: {value}")
    
    # Load and verify tensors
    x_batches = []
    y_batches = []
    
    i = 0
    while os.path.exists(f'data/cache/batch_{i}_x.npy'):
        x_batch = np.load(f'data/cache/batch_{i}_x.npy')
        y_batch = np.load(f'data/cache/batch_{i}_y.npy')
        
        print(f"\nBatch {i}:")
        print(f"X shape: {x_batch.shape}")
        print(f"Y shape: {y_batch.shape}")
        print(f"X min/max: {x_batch.min()}/{x_batch.max()}")
        print(f"Y unique values: {np.unique(y_batch)}")
        
        x_batches.append(x_batch)
        y_batches.append(y_batch)
        i += 1
    
    if not x_batches:
        print("Error: No cached batches found")
        return False
    
    # Concatenate all batches
    x = np.concatenate(x_batches)
    y = np.concatenate(y_batches)
    
    print("\nTotal dataset:")
    print(f"X shape: {x.shape}")
    print(f"Y shape: {y.shape}")
    print(f"X min/max: {x.min()}/{x.max()}")
    print(f"Y distribution: {np.bincount(y.astype(int))}")
    
    # Verify against original data
    print("\nVerifying against original data...")
    df = load_meteo_data()
    lake_tensors, mean, std = process_lake_data(df)
    targets = df["has_precip"].values
    
    # Calculate expected number of sequences
    sequence_length = metadata["sequence_length"]
    expected_sequences = len(lake_tensors) - sequence_length + 1
    
    print(f"Original data shape: {lake_tensors.shape}")
    print(f"Expected number of sequences: {expected_sequences}")
    print(f"Cached number of sequences: {x.shape[0]}")
    
    # Check if shapes match
    if x.shape[0] != expected_sequences:
        print("Error: Number of sequences doesn't match")
        return False
    
    # Check if normalization parameters match
    if not np.allclose(mean, metadata["normalization_mean"]) or not np.allclose(std, metadata["normalization_std"]):
        print("Warning: Normalization parameters differ")
    
    print("\nVerification complete!")
    return True

if __name__ == "__main__":
    verify_cached_data() 
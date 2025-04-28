import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

def create_directories():
    """Create necessary directories for data and cache"""
    os.makedirs('data', exist_ok=True)
    os.makedirs('cache', exist_ok=True)

def load_meteo_data(file_path):
    """Load and preprocess meteorological data"""
    print("Loading meteorological data...")
    df = pd.read_csv(file_path)
    
    # Convert datetime
    df['datetime'] = pd.to_datetime(df['Date_UTC'] + ' ' + df['Time_UTC'])
    df.set_index('datetime', inplace=True)
    
    # Convert numeric columns
    numeric_columns = ['Precip (in)', 'Temp (F)', 'Dew Point (F)', 'Humidity (%)', 
                      'Wind Speed (mph)', 'Wind Gust (mph)', 'Pressure (in)']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create temporal features
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['hour'] = df.index.hour
    df['season'] = (df['month'] % 12 + 3) // 3
    
    # Create binary target
    df['precipitation_category'] = (df['Precip (in)'] > 0).astype(int)
    
    return df

def process_lake_data(df):
    """Process Lake_data_1D into proper tensors"""
    print("Processing lake data...")
    
    def parse_array(x):
        try:
            if isinstance(x, str):
                values = x.strip('[]').split(',')
                arr = np.array([float(v.strip()) for v in values])
            else:
                arr = np.array(x)
            
            # Handle NaN values
            arr = np.nan_to_num(arr)
            
            # Validate dimensions
            if arr.size != 59 * 61:
                return None
            
            # Reshape and normalize
            arr = arr.reshape((59, 61))
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            
            return arr
        except Exception as e:
            print(f"Error processing array: {str(e)}")
            return None
    
    # Process arrays
    lake_arrays = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        arr = parse_array(row['Lake_data_1D'])
        if arr is not None:
            lake_arrays.append(arr)
            valid_indices.append(idx)
    
    # Convert to tensor
    lake_tensor = np.array(lake_arrays)
    lake_tensor = lake_tensor[..., np.newaxis]  # Add channel dimension
    
    # Filter dataframe
    df_filtered = df.loc[valid_indices].copy()
    
    return df_filtered, lake_tensor

def create_sequences(X, y, sequence_length=5):
    """Create time series sequences"""
    print("Creating sequences...")
    
    # Create dataset
    dataset = timeseries_dataset_from_array(
        data=X,
        targets=y,
        sequence_length=sequence_length,
        sequence_stride=1,
        batch_size=32
    )
    
    return dataset

def cache_data(df, lake_tensor, cache_dir='cache'):
    """Cache processed data"""
    print("Caching data...")
    
    # Cache metadata
    metadata = {
        'shape': lake_tensor.shape,
        'sequence_length': 5,
        'num_samples': len(df),
        'class_distribution': df['precipitation_category'].value_counts().to_dict()
    }
    
    with open(f'{cache_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    # Cache tensors
    np.save(f'{cache_dir}/lake_tensor.npy', lake_tensor)
    df.to_pickle(f'{cache_dir}/meteo_data.pkl')
    
    print("Data cached successfully!")

def load_cached_data(cache_dir='cache'):
    """Load cached data"""
    print("Loading cached data...")
    
    # Load metadata
    with open(f'{cache_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Load tensors
    lake_tensor = np.load(f'{cache_dir}/lake_tensor.npy')
    df = pd.read_pickle(f'{cache_dir}/meteo_data.pkl')
    
    return df, lake_tensor, metadata

def main():
    """Main function to load and process data"""
    create_directories()
    
    # Check if cached data exists
    if os.path.exists('cache/metadata.json'):
        print("Loading cached data...")
        df, lake_tensor, metadata = load_cached_data()
    else:
        print("Processing data from scratch...")
        # Load data
        df = load_meteo_data('data/2006Fall_2017Spring_GOES_meteo_combined.csv')
        
        # Process lake data
        df, lake_tensor = process_lake_data(df)
        
        # Cache data
        cache_data(df, lake_tensor)
    
    # Create sequences
    X = lake_tensor
    y = df['precipitation_category'].values
    dataset = create_sequences(X, y)
    
    print("\nData processing complete!")
    print(f"Number of samples: {len(df)}")
    print(f"Lake tensor shape: {lake_tensor.shape}")
    print(f"Class distribution: {df['precipitation_category'].value_counts().to_dict()}")
    
    return df, lake_tensor, dataset

if __name__ == "__main__":
    main() 
import ast
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import os

def process_lake_data_2d(data_str):
    """
    Process the Lake_data_2D string into a numpy array.
    
    Args:
        data_str (str): String representation of the numpy array
        
    Returns:
        numpy.ndarray: Processed array with NaN values replaced by 0
    """
    try:
        # First try to parse as a literal Python expression
        try:
            import ast
            data = ast.literal_eval(data_str)
            if isinstance(data, list):
                return np.array(data, dtype=np.float32).reshape(64, 64)
        except:
            pass

        # If that fails, try to parse the string directly
        # Remove any whitespace and split by commas
        values = []
        for val in data_str.replace(' ', '').split(','):
            val = val.strip('[]')
            if val == 'nan':
                values.append(0.0)
            else:
                try:
                    values.append(float(val))
                except ValueError:
                    values.append(0.0)
        
        # Convert to numpy array and reshape to 2D
        arr = np.array(values, dtype=np.float32)
        if len(arr) != 4096:  # 64x64 = 4096
            print(f"Warning: Expected 4096 values, got {len(arr)}")
            # Pad or truncate to get 4096 values
            if len(arr) < 4096:
                arr = np.pad(arr, (0, 4096 - len(arr)), 'constant')
            else:
                arr = arr[:4096]
        return arr.reshape(64, 64)
        
    except Exception as e:
        print(f"Error processing Lake_data_2D: {e}")
        return np.zeros((64, 64), dtype=np.float32)

def prepare_data():
    """
    Prepare the data for training by processing the Lake_data_2D column and rain labels.
    
    Returns:
        tuple: (lake_data_2d, rain_labels) where lake_data_2d is a numpy array of shape (n_samples, 64, 64)
              and rain_labels is a numpy array of shape (n_samples,)
    """
    try:
        # Read the meteorological data
        df = pd.read_csv('data/meteorological_data.csv')
        print(f"Total rows in dataset: {len(df)}")
        
        # Process Lake_data_2D column
        lake_data_2d = []
        processed = 0
        failed = 0
        
        for idx, row in df.iterrows():
            try:
                processed_data = process_lake_data_2d(row['Lake_data_2D'])
                lake_data_2d.append(processed_data)
                processed += 1
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                failed += 1
                lake_data_2d.append(np.zeros((64, 64), dtype=np.float32))
            
            if idx % 1000 == 0:
                print(f"Processed {idx} rows...")
        
        print(f"\nProcessing complete:")
        print(f"Successfully processed: {processed}")
        print(f"Failed to process: {failed}")
        
        # Convert to numpy array
        lake_data_2d = np.array(lake_data_2d)
        
        # Load rain labels
        rain_labels = np.load('data/rain_labels.npy')
        
        # Validate shapes
        if len(lake_data_2d) != len(rain_labels):
            raise ValueError(f"Shape mismatch: lake_data_2d has {len(lake_data_2d)} samples, but rain_labels has {len(rain_labels)} samples")
        
        # Save processed data
        os.makedirs('data', exist_ok=True)
        np.save('data/satellite_sequences.npy', lake_data_2d)
        
        print("\nData preparation completed!")
        print(f"Satellite data shape: {lake_data_2d.shape}")
        print(f"Rain labels shape: {rain_labels.shape}")
        
        return lake_data_2d, rain_labels
        
    except Exception as e:
        print(f"Error in prepare_data: {e}")
        raise

if __name__ == "__main__":
    satellite_data, rain_labels = prepare_data()
    print(f"Satellite data shape: {satellite_data.shape}")
    print(f"Rain labels shape: {rain_labels.shape}")
    print("Data preparation script completed.") 
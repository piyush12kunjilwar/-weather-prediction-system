"""
Run advanced time series analysis on the meteorological dataset
"""

import pandas as pd
import numpy as np
import os
from step1_time_series_analysis import generate_time_series_report

def create_directories():
    """Create necessary directories for saving visualizations."""
    dirs = [
        'visualizations',
        'visualizations/time_series',
        'visualizations/eda'
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {os.path.abspath(dir_path)}")

def load_and_prepare_data(file_path, nrows=None):
    """Load and prepare the meteorological dataset."""
    print(f"Reading data from: {file_path}")
    print(f"Loading {'all rows' if nrows is None else f'{nrows} rows'}")
    
    try:
        # Read CSV file
        df = pd.read_csv(file_path, nrows=nrows)
        print("Successfully read the CSV file")
        
        # Convert numeric columns
        numeric_columns = [
            'Temp (F)',
            'RH (%)',
            'Wind Spd (mph)',
            'Wind Direction (deg)',
            'Precip (in)'
        ]
        
        for col in numeric_columns:
            print(f"Converting {col} to numeric...")
            df[col] = pd.to_numeric(df[col].replace('m', np.nan), errors='coerce')
        
        # Convert units
        print("\nConverting units...")
        df['Temperature (C)'] = (df['Temp (F)'] - 32) * 5/9
        df['Wind Speed (m/s)'] = df['Wind Spd (mph)'] * 0.44704
        df['Relative Humidity (%)'] = df['RH (%)']
        df['Precip (in)'] = df['Precip (in)'].fillna(0)
        
        print("\nData preparation completed. Shape:", df.shape)
        return df
        
    except Exception as e:
        print(f"Error in load_and_prepare_data: {str(e)}")
        raise

def main():
    # File path
    data_path = 'data/2006Fall_2017Spring_GOES_meteo_combined.csv'
    
    try:
        # Create necessary directories
        print("=== Creating Directory Structure ===")
        create_directories()
        
        # Load and prepare data
        print("\n=== Starting Data Loading and Preparation ===")
        df = load_and_prepare_data(data_path, nrows=10000)  # Using 10000 rows for testing
        
        # Generate time series analysis
        print("\n=== Starting Time Series Analysis ===")
        df_ts = generate_time_series_report(df)
        
        print("\nAnalysis completed successfully!")
        print("Visualizations have been saved in the 'visualizations/time_series' directory")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import ast

def create_directories():
    """Create necessary directories for processed data and visualizations."""
    os.makedirs('processed_data', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)

def load_and_sample_data(file_path, sample_size=10000, random_state=42):
    """Load a random sample of data from the CSV file."""
    print("Loading sample data...")
    # Get total number of rows
    n_rows = sum(1 for _ in open(file_path)) - 1  # Subtract 1 for header
    # Generate random indices to skip
    skip_idx = np.random.choice(np.arange(1, n_rows + 1), 
                               size=n_rows - sample_size, 
                               replace=False)
    # Load data, skipping the random indices
    df = pd.read_csv(file_path, skiprows=skip_idx)
    return df

def process_lake_data(value):
    """Process lake data arrays to extract summary statistics."""
    try:
        if isinstance(value, str):
            # Convert string representation of array to actual array
            arr = np.array(ast.literal_eval(value))
            # Calculate summary statistics
            return {
                'mean': np.nanmean(arr),
                'std': np.nanstd(arr),
                'min': np.nanmin(arr),
                'max': np.nanmax(arr),
                'count': np.count_nonzero(~np.isnan(arr))
            }
        return None
    except:
        return None

def process_data(df):
    """Process the meteorological data."""
    print("Processing data...")
    
    # Convert datetime columns
    df['datetime_utc'] = pd.to_datetime(df['Date_UTC'] + ' ' + df['Time_UTC'])
    df['datetime_cst'] = pd.to_datetime(df['Date_CST'] + ' ' + df['Time_CST'])
    
    # Convert temperature from Fahrenheit to Celsius
    df['Temp_C'] = (df['Temp (F)'] - 32) * 5/9
    
    # Convert wind speed from mph to km/h
    df['Wind_Speed_kmh'] = df['Wind Spd (mph)'] * 1.60934
    
    # Process lake data
    print("Processing lake data...")
    df['Lake_data_1D_stats'] = df['Lake_data_1D'].apply(process_lake_data)
    df['Lake_data_2D_stats'] = df['Lake_data_2D'].apply(process_lake_data)
    
    # Drop original lake data columns to reduce file size
    df = df.drop(['Lake_data_1D', 'Lake_data_2D'], axis=1)
    
    # Handle missing values
    print("Handling missing values...")
    # Define numeric columns for mean imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Forward fill missing values for time series data
    time_series_cols = ['Temp (F)', 'RH (%)', 'Wind Spd (mph)', 'Wind Direction (deg)']
    df[time_series_cols] = df[time_series_cols].ffill()
    
    # Fill remaining missing values in numeric columns with column means
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    return df

def save_processed_data(df, output_path, summary_path):
    """Save processed data and summary to files."""
    # Save processed data
    df.to_csv(output_path, index=False)
    
    # Create summary
    summary = {
        'sample_size': int(len(df)),
        'time_range': {
            'start': str(df['datetime_utc'].min()),
            'end': str(df['datetime_utc'].max())
        },
        'columns': list(df.columns),
        'missing_values': {
            col: int(df[col].isna().sum()) for col in df.columns
        },
        'lake_data_summary': {
            'valid_arrays': int(df['Lake_data_1D_stats'].notna().sum()),
            'array_length': int(df['Lake_data_1D_stats'].apply(lambda x: x['count'] if x else 0).mean()),
            'mean_values': {
                'min': float(df['Lake_data_1D_stats'].apply(lambda x: x['min'] if x else np.nan).mean()),
                'max': float(df['Lake_data_1D_stats'].apply(lambda x: x['max'] if x else np.nan).mean()),
                'mean': float(df['Lake_data_1D_stats'].apply(lambda x: x['mean'] if x else np.nan).mean()),
                'std': float(df['Lake_data_1D_stats'].apply(lambda x: x['std'] if x else np.nan).mean())
            }
        }
    }
    
    # Save summary
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

def main():
    # Setup
    create_directories()
    
    # Define paths
    input_path = 'data/2006Fall_2017Spring_GOES_meteo_combined.csv'
    output_path = 'processed_data/processed_sample.csv'
    summary_path = 'processed_data/summary.json'
    
    # Load and process data
    df = load_and_sample_data(input_path)
    df_processed = process_data(df)
    
    # Save results
    save_processed_data(df_processed, output_path, summary_path)
    print("Processing complete! Check the 'processed_data' directory for results.")

if __name__ == "__main__":
    main() 
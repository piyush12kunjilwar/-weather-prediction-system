"""
Run Exploratory Data Analysis on the meteorological dataset
"""

import pandas as pd
import os
import sys
import numpy as np
from step1_eda import generate_eda_report

def load_data(file_path, nrows=None):
    """Load and prepare the meteorological dataset."""
    print(f"Reading data from: {file_path}")
    print(f"Loading {'all rows' if nrows is None else f'{nrows} rows'}")
    
    try:
        # Read CSV file with specified number of rows
        df = pd.read_csv(file_path, nrows=nrows)
        print("Successfully read the CSV file")
        
        # Print initial data info
        print("\nInitial Data Info:")
        print(df.info())
        
        return df
        
    except Exception as e:
        print(f"Error in load_data: {str(e)}")
        raise

def prepare_data(df):
    """Prepare and transform the data for analysis."""
    try:
        print("\nPreparing data for analysis...")
        
        # Convert numeric columns from strings to float
        print("Converting columns to numeric types...")
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
        
        # Convert temperature from Fahrenheit to Celsius
        print("Converting temperature from Fahrenheit to Celsius...")
        df['Temperature (C)'] = (df['Temp (F)'] - 32) * 5/9
        
        # Convert wind speed from mph to m/s
        print("Converting wind speed from mph to m/s...")
        df['Wind Speed (m/s)'] = df['Wind Spd (mph)'] * 0.44704
        
        # Convert relative humidity column name
        print("Processing relative humidity...")
        df['Relative Humidity (%)'] = df['RH (%)']
        
        # Process precipitation
        print("Processing precipitation...")
        df['Precip (in)'] = df['Precip (in)'].fillna(0)
        
        # Create precipitation category
        print("Creating precipitation category...")
        df['precipitation_category'] = (df['Precip (in)'] > 0).astype(int)
        
        # Print summary statistics of processed columns
        print("\nSummary statistics of processed columns:")
        processed_columns = [
            'Temperature (C)',
            'Relative Humidity (%)',
            'Wind Speed (m/s)',
            'Wind Direction (deg)',
            'Precip (in)',
            'precipitation_category'
        ]
        print(df[processed_columns].describe())
        
        print("\nData preparation completed successfully!")
        return df
        
    except Exception as e:
        print(f"Error in prepare_data: {str(e)}")
        print("Current columns in dataframe:", df.columns.tolist())
        raise

def main():
    # Load the dataset
    data_path = 'data/2006Fall_2017Spring_GOES_meteo_combined.csv'
    
    try:
        # Create visualizations directory if it doesn't exist
        viz_dir = 'visualizations/eda'
        os.makedirs(viz_dir, exist_ok=True)
        print(f"Created visualizations directory at: {os.path.abspath(viz_dir)}")
        
        # Load data (using only first 10000 rows for testing)
        print("\nStep 1: Loading dataset...")
        df = load_data(data_path, nrows=10000)
        
        # Prepare data
        print("\nStep 2: Preparing data...")
        df = prepare_data(df)
        
        # Define features to analyze
        meteorological_features = [
            'Precip (in)',
            'Temperature (C)',
            'Relative Humidity (%)',
            'Wind Speed (m/s)',
            'Wind Direction (deg)'
        ]
        
        print("\nStep 3: Verifying features...")
        for feature in meteorological_features:
            if feature not in df.columns:
                print(f"Warning: {feature} not found in dataset!")
            else:
                print(f"Found feature: {feature}")
        
        print("\nStep 4: Starting EDA report generation...")
        # Generate EDA report
        generate_eda_report(df, meteorological_features)
        
        print("\nStep 5: Checking generated files...")
        # List generated files
        print("Generated visualization files:")
        for file in os.listdir(viz_dir):
            print(f"- {file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the data file at {data_path}")
        print("Please ensure the meteorological dataset is in the data directory.")
    except Exception as e:
        print(f"An error occurred in main: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 
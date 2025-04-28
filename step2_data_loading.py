"""
Step 2: Data Loading and Preprocessing
"""

import pandas as pd
import numpy as np
import ast
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path='2006Fall_2017Spring_GOES_meteo_combined.csv'):
    """
    Advanced data loading and preprocessing with feature engineering.
    
    Features:
    - Temporal features (month, season, day of week)
    - Meteorological features
    - Statistical features (rolling means, std)
    - Weather pattern indicators
    - Satellite image processing
    """
    print("Loading and preprocessing data...")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Print information about the dataset
    print("\nDataset Info:")
    print("Columns in the dataset:", df.columns.tolist())
    print("\nFirst few rows of the dataset:")
    print(df.head())
    print("\nDataset shape:", df.shape)
    
    # Convert precipitation to numeric, handling 'NC' values
    df['Precip (in)'] = pd.to_numeric(df['Precip (in)'].replace('NC', '0'), errors='coerce')
    
    # Process date columns
    df['Date_UTC'] = pd.to_datetime(df['Date_UTC'])
    df['Month'] = df['Date_UTC'].dt.month
    df['DayOfWeek'] = df['Date_UTC'].dt.dayofweek
    df['Season'] = (df['Month'] % 12 + 3) // 3
    df['Hour'] = df['Date_UTC'].dt.hour
    
    # Create binary target with proper thresholding
    df['precipitation_category'] = (df['Precip (in)'] > 0).astype(int)
    
    # Handle missing values with advanced imputation
    df = handle_missing_values(df)
    
    # Process satellite images
    df = process_satellite_images(df)
    
    # Add statistical features
    df = add_statistical_features(df)
    
    # Add meteorological features
    df = add_meteorological_features(df)
    
    # Scale numerical features
    df = scale_numerical_features(df)
    
    return df

def handle_missing_values(df):
    """Advanced missing value handling with multiple strategies."""
    print("\nHandling missing values...")
    
    # Identify columns with missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    print(f"Columns with missing values: {missing_cols}")
    
    # Convert 'NC' to NaN in all columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace('NC', np.nan)
            df[col] = df[col].replace('M', np.nan)
    
    # For temporal data, convert to numeric and use forward fill
    temporal_cols = ['Temp (F)', 'RH (%)', 'Precip (in)', 'Wind Chill (F)', 'Heat Index (F)']
    for col in temporal_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[temporal_cols] = df[temporal_cols].fillna(method='ffill')
    
    # For cloud height data, handle non-numeric values and missing values
    cloud_cols = ['Low Cloud Ht (ft)', 'Med Cloud Ht (ft)', 'High Cloud Ht (ft)']
    for col in cloud_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Use KNN imputation for cloud height data
    imputer = KNNImputer(n_neighbors=5)
    df[cloud_cols] = imputer.fit_transform(df[cloud_cols])
    
    # For pressure data, convert to numeric and use linear interpolation
    pressure_cols = ['Atm Press (hPa)', 'Sea Lev Press (hPa)', 'Altimeter (hPa)']
    for col in pressure_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[pressure_cols] = df[pressure_cols].interpolate(method='linear')
    
    # For remaining numerical columns, convert to numeric and use median imputation
    remaining_numerical = ['Wind Spd (mph)', 'Wind Direction (deg)', 'Peak Wind Gust(mph)', 'Visibility (mi)', 'Dewpt (F)']
    for col in remaining_numerical:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[remaining_numerical] = df[remaining_numerical].fillna(df[remaining_numerical].median())
    
    # For categorical columns, use mode imputation
    categorical_cols = ['File_name_for_1D_lake', 'File_name_for_2D_lake']
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    print("Missing values handled successfully.")
    return df

def process_satellite_images(df):
    """Process satellite image data."""
    print("\nProcessing satellite images...")
    
    try:
        # Convert string representation of arrays to actual arrays
        df['Lake_data_1D'] = df['Lake_data_1D'].apply(lambda x: np.array(ast.literal_eval(str(x))))
        
        # Reshape 1D arrays to 2D images (assuming 64x64 images)
        df['Lake_data_2D'] = df['Lake_data_1D'].apply(lambda x: x.reshape(64, 64) if len(x) == 4096 else np.zeros((64, 64)))
        
        # Normalize images
        df['Lake_data_2D'] = df['Lake_data_2D'].apply(normalize_image)
        
        print("Satellite images processed successfully.")
    except Exception as e:
        print(f"Error processing satellite images: {str(e)}")
        # If there's an error, create dummy data for testing
        print("Creating dummy satellite data for testing...")
        n_samples = len(df)
        df['Lake_data_2D'] = [np.random.rand(64, 64) for _ in range(n_samples)]
    
    return df

def normalize_image(image):
    """Normalize image data to [0,1] range."""
    # Convert to float32
    image = image.astype(np.float32)
    
    # Replace NaN values with 0
    image = np.nan_to_num(image, 0)
    
    # Normalize to [0,1] range
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    else:
        image = np.zeros_like(image)
    
    return image

def add_statistical_features(df):
    """Add statistical features for better pattern recognition."""
    print("\nAdding statistical features...")
    
    # Rolling statistics for precipitation
    window_sizes = [3, 6, 12, 24]  # Different window sizes for different patterns
    for window in window_sizes:
        df[f'precip_rolling_mean_{window}h'] = df['Precip (in)'].rolling(window=window).mean()
        df[f'precip_rolling_std_{window}h'] = df['Precip (in)'].rolling(window=window).std()
        df[f'precip_rolling_max_{window}h'] = df['Precip (in)'].rolling(window=window).max()
    
    # Weather pattern indicators
    df['precip_trend'] = df['Precip (in)'].diff()
    df['precip_acceleration'] = df['precip_trend'].diff()
    
    # Cumulative precipitation
    df['precip_cumulative_24h'] = df['Precip (in)'].rolling(window=24).sum()
    
    print("Statistical features added successfully.")
    return df

def add_meteorological_features(df):
    """Add derived meteorological features."""
    print("\nAdding meteorological features...")
    
    # Temperature features
    df['temp_change_1h'] = df['Temp (F)'].diff()
    df['temp_change_3h'] = df['Temp (F)'].diff(periods=3)
    
    # Humidity features
    df['humidity_change_1h'] = df['RH (%)'].diff()
    
    # Pressure features
    df['pressure_change_1h'] = df['Atm Press (hPa)'].diff()
    df['pressure_change_3h'] = df['Atm Press (hPa)'].diff(periods=3)
    
    # Cloud cover features
    df['total_cloud_cover'] = df[['Low Cloud Ht (ft)', 'Med Cloud Ht (ft)', 'High Cloud Ht (ft)']].mean(axis=1)
    
    print("Meteorological features added successfully.")
    return df

def scale_numerical_features(df):
    """Scale numerical features using StandardScaler."""
    print("\nScaling numerical features...")
    
    # Identify numerical columns to scale
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude binary target and image data
    numerical_cols = [col for col in numerical_cols if col not in ['precipitation_category', 'Lake_data_1D', 'Lake_data_2D']]
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Scale features
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    print("Numerical features scaled successfully.")
    return df 
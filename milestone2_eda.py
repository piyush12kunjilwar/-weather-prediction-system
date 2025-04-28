import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Create directories for visualizations
os.makedirs('visualizations/distributions', exist_ok=True)
os.makedirs('visualizations/correlations', exist_ok=True)
os.makedirs('visualizations/time_series', exist_ok=True)
os.makedirs('visualizations/features', exist_ok=True)

def load_and_preprocess_data():
    """Load and preprocess the meteorological dataset."""
    print('Loading data...')
    data_path = 'data/2006Fall_2017Spring_GOES_meteo_combined.csv'
    df = pd.read_csv(data_path)
    
    # Convert datetime
    print('Converting datetime...')
    df['datetime'] = pd.to_datetime(df['Date_UTC'] + ' ' + df['Time_UTC'])
    df.set_index('datetime', inplace=True)
    
    return df

def handle_missing_data(df):
    """Handle missing values in the dataset."""
    print('\nMissing values before cleaning:')
    print(df.isnull().sum())
    
    # Replace 'm' with NaN in numeric columns
    numeric_columns = ['Temp (F)', 'Wind Spd (mph)', 'RH (%)', 'Precip (in)']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col].replace('m', np.nan), errors='coerce')
    
    # Fill missing values
    df['Precip (in)'] = df['Precip (in)'].fillna(0)  # Assume no precipitation if missing
    df['RH (%)'] = df['RH (%)'].fillna(df['RH (%)'].mean())  # Use mean for humidity
    df['Wind Spd (mph)'] = df['Wind Spd (mph)'].fillna(df['Wind Spd (mph)'].mean())
    df['Temp (F)'] = df['Temp (F)'].interpolate(method='time')  # Interpolate temperature
    
    print('\nMissing values after cleaning:')
    print(df.isnull().sum())
    
    return df

def create_feature_distributions(df):
    """Create distribution plots for key features."""
    features = ['Temp (F)', 'Wind Spd (mph)', 'RH (%)', 'Precip (in)']
    
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=feature, kde=True)
        plt.title(f'Distribution of {feature}')
        plt.savefig(f'visualizations/distributions/{feature.replace(" ", "_").replace("(", "").replace(")", "")}_dist.png')
        plt.close()
    
    print('Distribution plots saved in visualizations/distributions/')

def create_correlation_heatmap(df):
    """Create correlation heatmap for numerical features."""
    numeric_cols = ['Temp (F)', 'Wind Spd (mph)', 'RH (%)', 'Precip (in)']
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Weather Variables')
    plt.tight_layout()
    plt.savefig('visualizations/correlations/correlation_matrix.png')
    plt.close()
    
    print('Correlation heatmap saved in visualizations/correlations/')

def analyze_seasonal_patterns(df):
    """Analyze and visualize seasonal patterns."""
    df['Month'] = df.index.month
    df['Hour'] = df.index.hour
    
    # Monthly patterns
    monthly_avg = df.groupby('Month')[['Temp (F)', 'Wind Spd (mph)', 'RH (%)', 'Precip (in)']].mean()
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(monthly_avg.columns, 1):
        plt.subplot(2, 2, i)
        monthly_avg[col].plot(marker='o')
        plt.title(f'Monthly Average {col}')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/time_series/monthly_patterns.png')
    plt.close()
    
    print('Seasonal patterns saved in visualizations/time_series/')

def create_precipitation_features(df):
    """Create binary precipitation classification and other features."""
    # Create binary precipitation category
    df['precipitation_category'] = (df['Precip (in)'] > 0).astype(int)
    
    # Calculate class distribution
    class_dist = df['precipitation_category'].value_counts(normalize=True)
    
    plt.figure(figsize=(8, 6))
    class_dist.plot(kind='bar')
    plt.title('Distribution of Rain vs No Rain')
    plt.xlabel('Class (0: No Rain, 1: Rain)')
    plt.ylabel('Proportion')
    plt.savefig('visualizations/features/rain_distribution.png')
    plt.close()
    
    print('\nClass distribution:')
    print(class_dist)
    
    return df

def main():
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Handle missing data
    df = handle_missing_data(df)
    
    # Create visualizations
    create_feature_distributions(df)
    create_correlation_heatmap(df)
    analyze_seasonal_patterns(df)
    
    # Create precipitation features
    df = create_precipitation_features(df)
    
    # Save processed dataset
    df.to_csv('data/processed_meteorological_data.csv')
    print('\nProcessed data saved to data/processed_meteorological_data.csv')

if __name__ == "__main__":
    main() 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def setup_directories():
    """Create visualization directories if they don't exist."""
    Path("visualizations/eda").mkdir(parents=True, exist_ok=True)
    Path("visualizations/time_series").mkdir(parents=True, exist_ok=True)
    Path("visualizations/correlations").mkdir(parents=True, exist_ok=True)

def plot_feature_distributions(df):
    """Plot distributions of key meteorological features."""
    print("Plotting feature distributions...")
    
    features = ['Precip (in)', 'Temp (F)', 'RH (%)', 'Wind Spd (mph)']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(features):
        sns.histplot(data=df, x=feature, ax=axes[idx])
        axes[idx].set_title(f'Distribution of {feature}')
    
    plt.tight_layout()
    plt.savefig('visualizations/eda/feature_distributions.png')
    plt.close()

def load_and_prepare_data():
    """Load and prepare data for analysis."""
    print("Loading data...")
    df = pd.read_csv('data/2006Fall_2017Spring_GOES_meteo_combined.csv')
    
    # Convert numeric columns to float
    numeric_features = ['Precip (in)', 'Temp (F)', 'RH (%)', 'Wind Spd (mph)']
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def plot_time_series(df):
    """Visualize time-series trends."""
    print("Plotting time series trends...")
    
    # Create datetime column
    df['datetime'] = pd.to_datetime(df['Date_UTC'] + ' ' + df['Time_UTC'])
    
    # Create daily aggregations for numeric features
    features = ['Precip (in)', 'Temp (F)', 'RH (%)', 'Wind Spd (mph)']
    
    for feature in features:
        # Group by date and calculate mean
        daily_data = df.groupby(df['datetime'].dt.date)[feature].mean()
        
        plt.figure(figsize=(15, 5))
        plt.plot(daily_data.index, daily_data.values)
        plt.title(f'Daily Average {feature} Over Time')
        plt.xlabel('Date')
        plt.ylabel(feature)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'visualizations/time_series/{feature.replace(" ", "_")}_trend.png')
        plt.close()

def create_correlation_heatmap(df):
    """Generate correlation heatmap for numerical features."""
    print("Creating correlation heatmap...")
    
    # Select only numeric columns, excluding Lake data and datetime
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if not col.startswith('Lake_data')]
    
    correlations = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Meteorological Features')
    plt.tight_layout()
    plt.savefig('visualizations/correlations/correlation_heatmap.png')
    plt.close()

def add_precipitation_category(df):
    """Add binary precipitation classification."""
    print("Adding precipitation category...")
    
    df['precipitation_category'] = (df['Precip (in)'] > 0).astype(int)
    
    # Print class distribution
    class_dist = df['precipitation_category'].value_counts(normalize=True)
    print("\nPrecipitation Class Distribution:")
    print(f"No Rain (0): {class_dist[0]:.2%}")
    print(f"Rain (1): {class_dist[1]:.2%}")
    
    return df

def analyze_data():
    """Main function to perform EDA and feature engineering."""
    # Setup
    setup_directories()
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Generate visualizations
    plot_feature_distributions(df)
    plot_time_series(df)
    create_correlation_heatmap(df)
    
    # Add precipitation category
    df = add_precipitation_category(df)
    
    # Save processed data
    print("Saving processed data...")
    df.to_csv('processed_data/processed_with_categories.csv', index=False)
    
    print("Analysis complete! Check the 'visualizations' directory for results.")

if __name__ == "__main__":
    analyze_data() 
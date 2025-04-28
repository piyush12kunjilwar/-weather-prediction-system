"""
Meteorological Data Analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Create visualizations directory
os.makedirs('visualizations', exist_ok=True)

# Set plot style
plt.style.use('default')  # Using default style instead of seaborn
sns.set_theme()  # This is the correct way to set seaborn style

def load_and_prepare_data():
    """Load and prepare the meteorological data."""
    print("Loading data...")
    data_path = 'data/2006Fall_2017Spring_GOES_meteo_combined.csv'
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # Convert datetime
    print("Converting datetime...")
    df['datetime'] = pd.to_datetime(df['Date_UTC'] + ' ' + df['Time_UTC'])
    df.set_index('datetime', inplace=True)
    
    # Convert numeric columns
    print("Converting numeric columns...")
    numeric_columns = ['Temp (F)', 'Wind Spd (mph)', 'RH (%)', 'Precip (in)']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col].replace('m', np.nan), errors='coerce')
    
    # Convert units
    print("Converting units...")
    df['Temperature (C)'] = (df['Temp (F)'] - 32) * 5/9
    df['Wind Speed (m/s)'] = df['Wind Spd (mph)'] * 0.44704
    df['Relative Humidity (%)'] = df['RH (%)']
    df['Precip (in)'] = df['Precip (in)'].fillna(0)
    
    return df

def plot_temperature_analysis(df):
    """Create temperature analysis plots."""
    print("\nCreating temperature plots...")
    
    # Temperature over time
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['Temperature (C)'])
    plt.title('Temperature Over Time')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    plt.savefig('visualizations/temperature_over_time.png')
    plt.close()
    
    # Daily temperature variation
    daily_temp = df.groupby(df.index.hour)['Temperature (C)'].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(daily_temp.index, daily_temp.values)
    plt.title('Average Daily Temperature Variation')
    plt.xlabel('Hour of Day')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    plt.savefig('visualizations/daily_temperature_variation.png')
    plt.close()

def plot_humidity_analysis(df):
    """Create humidity analysis plots."""
    print("Creating humidity plots...")
    
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['Relative Humidity (%)'])
    plt.title('Relative Humidity Over Time')
    plt.xlabel('Date')
    plt.ylabel('Relative Humidity (%)')
    plt.grid(True)
    plt.savefig('visualizations/humidity_over_time.png')
    plt.close()

def plot_wind_analysis(df):
    """Create wind analysis plots."""
    print("Creating wind plots...")
    
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['Wind Speed (m/s)'])
    plt.title('Wind Speed Over Time')
    plt.xlabel('Date')
    plt.ylabel('Wind Speed (m/s)')
    plt.grid(True)
    plt.savefig('visualizations/wind_speed_over_time.png')
    plt.close()

def plot_precipitation_analysis(df):
    """Create precipitation analysis plots."""
    print("Creating precipitation plots...")
    
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['Precip (in)'])
    plt.title('Precipitation Over Time')
    plt.xlabel('Date')
    plt.ylabel('Precipitation (in)')
    plt.grid(True)
    plt.savefig('visualizations/precipitation_over_time.png')
    plt.close()

def main():
    """Main analysis function."""
    print("=== Starting Meteorological Data Analysis ===\n")
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Create visualizations
    plot_temperature_analysis(df)
    plot_humidity_analysis(df)
    plot_wind_analysis(df)
    plot_precipitation_analysis(df)
    
    print("\n=== Analysis Complete ===")
    print("Visualizations have been saved in the 'visualizations' directory")

if __name__ == "__main__":
    main() 
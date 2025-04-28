"""
Advanced Time Series Analysis for Meteorological Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

def prepare_time_series_data(df):
    """
    Prepare data for time series analysis by setting up proper datetime index
    and handling missing values.
    """
    print("Preparing time series data...")
    
    try:
        # Convert datetime columns
        print("Converting datetime columns...")
        df['datetime'] = pd.to_datetime(df['Date_UTC'] + ' ' + df['Time_UTC'])
        df.set_index('datetime', inplace=True)
        
        # Sort by datetime
        print("Sorting by datetime...")
        df.sort_index(inplace=True)
        
        # Resample to hourly frequency and forward fill missing values
        print("Resampling to hourly frequency...")
        df = df.resample('h').ffill(limit=6)  # using 'h' instead of 'H'
        
        print("Time series data preparation completed successfully.")
        return df
        
    except Exception as e:
        print(f"Error in prepare_time_series_data: {str(e)}")
        raise

def create_safe_filename(feature_name):
    """Create a safe filename from a feature name."""
    # Replace unsafe characters
    safe_name = feature_name.lower()
    unsafe_chars = [' ', '(', ')', '/', '\\', '%', '*', ':', '?', '"', '<', '>', '|']
    for char in unsafe_chars:
        safe_name = safe_name.replace(char, '_')
    # Remove multiple underscores
    while '__' in safe_name:
        safe_name = safe_name.replace('__', '_')
    # Remove leading/trailing underscores
    safe_name = safe_name.strip('_')
    return safe_name

def ensure_directory(directory):
    """Ensure directory exists and return absolute path."""
    os.makedirs(directory, exist_ok=True)
    return os.path.abspath(directory)

def plot_time_series_basic(df, features, save_dir='visualizations/time_series'):
    """
    Create basic time series plots with trend lines and confidence intervals.
    """
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.abspath(save_dir)
    print(f"\nGenerating basic time series plots in: {save_dir}")
    
    for feature in features:
        try:
            print(f"Processing feature: {feature}")
            plt.figure(figsize=(15, 7))
            
            # Plot actual data
            plt.plot(df.index, df[feature], label='Actual', alpha=0.6)
            
            # Calculate and plot trend
            z = np.polyfit(range(len(df.index)), df[feature].values, 1)
            p = np.poly1d(z)
            plt.plot(df.index, p(range(len(df.index))), 
                    'r--', label=f'Trend (slope: {z[0]:.4f})')
            
            # Calculate confidence intervals
            confidence_interval = 0.95
            z_score = stats.norm.ppf((1 + confidence_interval) / 2)
            std_err = np.std(df[feature]) / np.sqrt(len(df))
            margin_of_error = z_score * std_err
            
            plt.fill_between(df.index, 
                            df[feature] - margin_of_error,
                            df[feature] + margin_of_error,
                            alpha=0.2, label=f'{confidence_interval*100}% Confidence Interval')
            
            plt.title(f'Time Series of {feature} with Trend and Confidence Interval')
            plt.xlabel('Date')
            plt.ylabel(feature)
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Create safe filename
            safe_name = feature.lower()
            for char in ['(', ')', '/', '\\', ' ', '%', '*', ':', '?', '"', '<', '>', '|']:
                safe_name = safe_name.replace(char, '_')
            while '__' in safe_name:
                safe_name = safe_name.replace('__', '_')
            safe_name = safe_name.strip('_')
            
            save_path = os.path.join(save_dir, f'{safe_name}_basic.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Successfully saved plot to: {save_path}")
            
        except Exception as e:
            print(f"Error plotting {feature}: {str(e)}")
            plt.close()
            continue

def analyze_seasonal_decomposition(df, features, save_dir='visualizations/time_series'):
    """
    Perform and plot seasonal decomposition of time series data.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.abspath(save_dir)
    print(f"\nPerforming seasonal decomposition analysis in: {save_dir}")
    
    for feature in features:
        try:
            print(f"Processing seasonal decomposition for: {feature}")
            decomposition = seasonal_decompose(df[feature], period=24)
            
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
            
            decomposition.observed.plot(ax=ax1)
            ax1.set_title(f'Original {feature} Time Series')
            ax1.grid(True)
            
            decomposition.trend.plot(ax=ax2)
            ax2.set_title('Trend')
            ax2.grid(True)
            
            decomposition.seasonal.plot(ax=ax3)
            ax3.set_title('Seasonal')
            ax3.grid(True)
            
            decomposition.resid.plot(ax=ax4)
            ax4.set_title('Residual')
            ax4.grid(True)
            
            plt.tight_layout()
            
            # Create safe filename
            safe_name = feature.lower()
            for char in ['(', ')', '/', '\\', ' ', '%', '*', ':', '?', '"', '<', '>', '|']:
                safe_name = safe_name.replace(char, '_')
            while '__' in safe_name:
                safe_name = safe_name.replace('__', '_')
            safe_name = safe_name.strip('_')
            
            save_path = os.path.join(save_dir, f'{safe_name}_decomposition.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Successfully saved decomposition plot to: {save_path}")
            
        except Exception as e:
            print(f"Could not perform seasonal decomposition for {feature}: {str(e)}")
            plt.close()

def analyze_hourly_patterns(df, features, save_dir='visualizations/time_series'):
    """
    Analyze and visualize hourly patterns in the data.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.abspath(save_dir)
    print(f"\nAnalyzing hourly patterns in: {save_dir}")
    
    for feature in features:
        try:
            print(f"Processing hourly patterns for: {feature}")
            plt.figure(figsize=(12, 6))
            
            df['hour'] = df.index.hour
            hourly_means = df.groupby('hour')[feature].mean()
            hourly_std = df.groupby('hour')[feature].std()
            
            plt.plot(hourly_means.index, hourly_means.values, 'b-', label='Mean')
            plt.fill_between(hourly_means.index,
                            hourly_means - hourly_std,
                            hourly_means + hourly_std,
                            alpha=0.2, label='±1 std')
            
            plt.title(f'Average {feature} by Hour of Day')
            plt.xlabel('Hour of Day')
            plt.ylabel(feature)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            # Create safe filename
            safe_name = feature.lower()
            for char in ['(', ')', '/', '\\', ' ', '%', '*', ':', '?', '"', '<', '>', '|']:
                safe_name = safe_name.replace(char, '_')
            while '__' in safe_name:
                safe_name = safe_name.replace('__', '_')
            safe_name = safe_name.strip('_')
            
            save_path = os.path.join(save_dir, f'{safe_name}_hourly_pattern.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Successfully saved hourly pattern plot to: {save_path}")
            
        except Exception as e:
            print(f"Error analyzing hourly patterns for {feature}: {str(e)}")
            plt.close()

def analyze_monthly_patterns(df, features, save_dir='visualizations/time_series'):
    """
    Analyze and visualize monthly patterns in the data.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.abspath(save_dir)
    print(f"\nAnalyzing monthly patterns in: {save_dir}")
    
    for feature in features:
        try:
            print(f"Processing monthly patterns for: {feature}")
            plt.figure(figsize=(12, 6))
            
            df['month'] = df.index.month
            monthly_stats = df.groupby('month')[feature].agg(['mean', 'std', 'count'])
            
            plt.errorbar(monthly_stats.index, monthly_stats['mean'],
                        yerr=monthly_stats['std'] / np.sqrt(monthly_stats['count']),
                        fmt='o-', capsize=5, label='Mean with 95% CI')
            
            plt.title(f'Monthly Pattern of {feature}')
            plt.xlabel('Month')
            plt.ylabel(feature)
            plt.grid(True)
            plt.legend()
            
            plt.xticks(range(1, 13), 
                      ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            
            plt.tight_layout()
            
            # Create safe filename
            safe_name = feature.lower()
            for char in ['(', ')', '/', '\\', ' ', '%', '*', ':', '?', '"', '<', '>', '|']:
                safe_name = safe_name.replace(char, '_')
            while '__' in safe_name:
                safe_name = safe_name.replace('__', '_')
            safe_name = safe_name.strip('_')
            
            save_path = os.path.join(save_dir, f'{safe_name}_monthly_pattern.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Successfully saved monthly pattern plot to: {save_path}")
            
        except Exception as e:
            print(f"Error analyzing monthly patterns for {feature}: {str(e)}")
            plt.close()

def generate_time_series_report(df):
    """
    Generate comprehensive time series analysis report.
    """
    print("=== Starting Time Series Analysis ===")
    
    # Prepare data
    df_ts = prepare_time_series_data(df)
    
    # Define features to analyze
    features = [
        'Precip (in)',
        'Temperature (C)',
        'Relative Humidity (%)',
        'Wind Speed (m/s)',
        'Wind Direction (deg)'
    ]
    
    # Run analyses
    plot_time_series_basic(df_ts, features)
    analyze_seasonal_decomposition(df_ts, features)
    analyze_hourly_patterns(df_ts, features)
    analyze_monthly_patterns(df_ts, features)
    
    print("\n=== Time Series Analysis Completed ===")
    return df_ts 
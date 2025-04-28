"""
Step 1: Exploratory Data Analysis (EDA)
This module contains functions for analyzing and visualizing the meteorological dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def plot_feature_distributions(df, features, save_dir='visualizations/eda'):
    """
    Plot distributions of specified meteorological features.
    
    Args:
        df: DataFrame containing the meteorological data
        features: List of feature names to plot
        save_dir: Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=feature, kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{feature.replace(" ", "_").lower()}_distribution.png'))
        plt.close()
        
        # Print basic statistics
        print(f"\nStatistics for {feature}:")
        print(df[feature].describe())

def plot_time_series(df, features, date_column='date', save_dir='visualizations/eda'):
    """
    Plot time series trends for specified features.
    
    Args:
        df: DataFrame containing the meteorological data
        features: List of feature names to plot
        date_column: Name of the column containing dates
        save_dir: Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    for feature in features:
        plt.figure(figsize=(15, 6))
        plt.plot(df[date_column], df[feature])
        plt.title(f'Time Series of {feature}')
        plt.xlabel('Date')
        plt.ylabel(feature)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{feature.replace(" ", "_").lower()}_timeseries.png'))
        plt.close()

def plot_correlation_heatmap(df, features, save_dir='visualizations/eda'):
    """
    Generate correlation heatmap for numerical features.
    
    Args:
        df: DataFrame containing the meteorological data
        features: List of numerical features to include
        save_dir: Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate correlation matrix
    corr_matrix = df[features].corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Heatmap of Meteorological Features')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # Print strong correlations
    print("\nStrong correlations (|correlation| > 0.5):")
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.5:
                print(f"{features[i]} vs {features[j]}: {corr:.3f}")

def analyze_class_imbalance(df, target_column='precipitation_category', save_dir='visualizations/eda'):
    """
    Analyze and visualize class imbalance in the target variable.
    
    Args:
        df: DataFrame containing the meteorological data
        target_column: Name of the binary classification column
        save_dir: Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate class distribution
    class_dist = df[target_column].value_counts()
    total_samples = len(df)
    
    # Create bar plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_dist.index, y=class_dist.values)
    plt.title('Class Distribution in Target Variable')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Add percentage labels
    for i, v in enumerate(class_dist.values):
        percentage = v / total_samples * 100
        plt.text(i, v, f'{percentage:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_distribution.png'))
    plt.close()
    
    # Print class distribution statistics
    print("\nClass Distribution:")
    for class_label, count in class_dist.items():
        percentage = count / total_samples * 100
        print(f"Class {class_label}: {count} samples ({percentage:.1f}%)")

def analyze_seasonal_patterns(df, feature, date_column='date', save_dir='visualizations/eda'):
    """
    Analyze and visualize seasonal patterns in the data.
    
    Args:
        df: DataFrame containing the meteorological data
        feature: Feature to analyze for seasonal patterns
        date_column: Name of the column containing dates
        save_dir: Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract month and season
    df['month'] = df[date_column].dt.month
    df['season'] = pd.cut(df[date_column].dt.month, bins=[0, 3, 6, 9, 12], 
                         labels=['Winter', 'Spring', 'Summer', 'Fall'])
    
    # Monthly analysis
    plt.figure(figsize=(12, 6))
    monthly_avg = df.groupby('month')[feature].mean()
    sns.lineplot(x=monthly_avg.index, y=monthly_avg.values)
    plt.title(f'Monthly Average of {feature}')
    plt.xlabel('Month')
    plt.ylabel(f'Average {feature}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{feature.replace(" ", "_").lower()}_monthly_pattern.png'))
    plt.close()
    
    # Seasonal analysis
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='season', y=feature)
    plt.title(f'Seasonal Distribution of {feature}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{feature.replace(" ", "_").lower()}_seasonal_pattern.png'))
    plt.close()
    
    # Print seasonal statistics
    print(f"\nSeasonal Statistics for {feature}:")
    print(df.groupby('season')[feature].describe())

def generate_eda_report(df, meteorological_features, target_column='precipitation_category'):
    """
    Generate comprehensive EDA visualizations and analysis.
    
    Args:
        df: DataFrame containing the meteorological data
        meteorological_features: List of meteorological features to analyze
        target_column: Name of the binary classification column
    """
    print("=== Starting Exploratory Data Analysis ===\n")
    
    # 1. Feature Distributions
    print("Generating feature distribution plots...")
    plot_feature_distributions(df, meteorological_features)
    
    # 2. Time Series Analysis
    print("\nGenerating time series plots...")
    plot_time_series(df, meteorological_features)
    
    # 3. Correlation Analysis
    print("\nGenerating correlation heatmap...")
    plot_correlation_heatmap(df, meteorological_features)
    
    # 4. Class Imbalance Analysis
    print("\nAnalyzing class imbalance...")
    analyze_class_imbalance(df, target_column)
    
    # 5. Seasonal Pattern Analysis
    print("\nAnalyzing seasonal patterns...")
    for feature in meteorological_features:
        analyze_seasonal_patterns(df, feature)
    
    print("\n=== EDA Analysis Completed ===") 
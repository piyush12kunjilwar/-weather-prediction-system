"""
Test script for verifying the data preparation and model training pipeline
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from data_preparation import (
    prepare_data_for_training,
    compute_class_weights,
    create_sequence_generator
)
from cnn_rnn_model import create_cnn_rnn_model, train_model, evaluate_model

def create_test_data(n_samples=100):
    """Create a test dataset with all required features"""
    test_data = {
        'Date_UTC': pd.date_range(start='2020-01-01', periods=n_samples, freq='h'),
        'Temp (F)': np.random.normal(50, 10, n_samples),
        'RH (%)': np.random.normal(60, 15, n_samples),
        'Precip (in)': np.random.choice([0, 0.1, 0.2, 0.5], n_samples, p=[0.7, 0.15, 0.1, 0.05]),
        'Wind Chill (F)': np.random.normal(45, 10, n_samples),
        'Heat Index (F)': np.random.normal(55, 10, n_samples),
        'Low Cloud Ht (ft)': np.random.normal(5000, 1000, n_samples),
        'Med Cloud Ht (ft)': np.random.normal(10000, 2000, n_samples),
        'High Cloud Ht (ft)': np.random.normal(20000, 3000, n_samples),
        'Visibility (mi)': np.random.normal(10, 2, n_samples),
        'Atm Press (hPa)': np.random.normal(1013, 10, n_samples),
        'Sea Lev Press (hPa)': np.random.normal(1013, 10, n_samples),
        'Altimeter (hPa)': np.random.normal(1013, 10, n_samples),
        'Lake_data_2D': [np.random.rand(64, 64) for _ in range(n_samples)],
        'precipitation_category': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    # Add derived features
    df = pd.DataFrame(test_data)
    df['temp_change_1h'] = df['Temp (F)'].diff()
    df['temp_change_3h'] = df['Temp (F)'].diff(periods=3)
    df['humidity_change_1h'] = df['RH (%)'].diff()
    df['pressure_change_1h'] = df['Atm Press (hPa)'].diff()
    df['pressure_change_3h'] = df['Atm Press (hPa)'].diff(periods=3)
    df['total_cloud_cover'] = df[['Low Cloud Ht (ft)', 'Med Cloud Ht (ft)', 'High Cloud Ht (ft)']].mean(axis=1)
    
    return df

def test_data_preparation():
    """Test data preparation pipeline"""
    print("\n=== Testing Data Preparation ===")
    
    try:
        # Create test dataset with more samples
        df = create_test_data(n_samples=500)  # Increased from 100 to 500
        
        # Test data preparation
        print("\nTesting data preparation...")
        train_data, val_data, test_data, class_weights = prepare_data_for_training(
            df, sequence_length=24, batch_size=32
        )
        
        # Test data generators
        print("\nTesting data generators...")
        for batch_x, batch_y in train_data.take(1):
            print(f"Batch shape: {batch_x.shape}")
            print(f"Batch labels shape: {batch_y.shape}")
            break
        
        # Test class weights
        print("\nTesting class weights...")
        print(f"Class weights: {class_weights}")
        
        return True
    except Exception as e:
        print(f"\nError in data preparation: {str(e)}")
        return False

def test_model_creation():
    """Test model creation and compilation"""
    print("\n=== Testing Model Creation ===")
    
    try:
        # Create model
        model = create_cnn_rnn_model(input_shape=(24, 64, 64, 1))
        
        # Test model summary
        print("\nModel Summary:")
        model.summary()
        
        # Test model compilation
        print("\nTesting model compilation...")
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return True
    except Exception as e:
        print(f"\nError in model creation: {str(e)}")
        return False

def test_training_pipeline():
    """Test the complete training pipeline"""
    print("\n=== Testing Training Pipeline ===")
    
    try:
        # Create test dataset with more samples
        df = create_test_data(n_samples=500)  # Increased from 200 to 500
        
        # Prepare data
        train_data, val_data, test_data, class_weights = prepare_data_for_training(
            df, sequence_length=24, batch_size=32
        )
        
        # Create and compile model
        model = create_cnn_rnn_model()
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Test training for a few epochs
        print("\nTesting model training...")
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=2,
            class_weight=class_weights,
            verbose=1
        )
        
        # Test evaluation
        print("\nTesting model evaluation...")
        results = evaluate_model(model, test_data)
        print(f"Evaluation results: {results}")
        
        return True
    except Exception as e:
        print(f"\nError in training pipeline: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("Starting pipeline tests...")
    
    # Test data preparation
    if test_data_preparation():
        print("\nData preparation tests passed!")
    else:
        print("\nData preparation tests failed!")
        return
    
    # Test model creation
    if test_model_creation():
        print("\nModel creation tests passed!")
    else:
        print("\nModel creation tests failed!")
        return
    
    # Test training pipeline
    if test_training_pipeline():
        print("\nTraining pipeline tests passed!")
    else:
        print("\nTraining pipeline tests failed!")
        return
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main() 
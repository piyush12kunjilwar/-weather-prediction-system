import os
import numpy as np
import pandas as pd
from datetime import datetime
import pytest
import cv2
from data_preprocessing import (
    load_meteorological_data,
    load_and_preprocess_image,
    extract_timestamp_from_filename,
    create_sequences,
    prepare_data
)

def test_load_meteorological_data():
    """Test loading and preprocessing of meteorological data"""
    # Create a sample meteorological data file
    data = {
        'timestamp': ['2023-01-01 00:00', '2023-01-01 01:00', '2023-01-01 02:00'],
        'temp': [32.0, 33.0, 34.0],
        'humidity': [80.0, 82.0, 85.0],
        'wind_speed': [10.0, 12.0, 15.0]
    }
    df = pd.DataFrame(data)
    df.to_csv('test_meteo.csv', index=False)
    
    # Test loading
    meteo_data, features = load_meteorological_data('test_meteo.csv')
    
    # Verify results
    assert isinstance(meteo_data, pd.DataFrame)
    assert len(features) == 3  # temp, humidity, wind_speed
    assert isinstance(meteo_data.index, pd.DatetimeIndex)
    assert meteo_data.shape[0] == 3
    
    # Clean up
    os.remove('test_meteo.csv')

def test_load_and_preprocess_image():
    """Test image loading and preprocessing"""
    # Create a test image
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    cv2.imwrite('test_image.png', test_img)
    
    # Test preprocessing
    processed_img = load_and_preprocess_image('test_image.png', (128, 128))
    
    # Verify results
    assert isinstance(processed_img, np.ndarray)
    assert processed_img.shape == (128, 128, 3)
    assert processed_img.dtype == np.float32
    assert np.all(processed_img >= 0) and np.all(processed_img <= 1)
    
    # Clean up
    os.remove('test_image.png')

def test_extract_timestamp_from_filename():
    """Test timestamp extraction from filenames"""
    # Test valid filename
    filename = "2023-01-01_12-30.png"
    timestamp = extract_timestamp_from_filename(filename)
    assert isinstance(timestamp, datetime)
    assert timestamp.year == 2023
    assert timestamp.month == 1
    assert timestamp.day == 1
    assert timestamp.hour == 12
    assert timestamp.minute == 30
    
    # Test invalid filename
    with pytest.raises(Exception):
        extract_timestamp_from_filename("invalid_filename.png")

def test_create_sequences():
    """Test sequence creation from aligned data"""
    # Create sample data
    timestamps = pd.date_range(start='2023-01-01', periods=10, freq='h')
    meteo_data = pd.DataFrame(
        np.random.rand(10, 3),
        index=timestamps,
        columns=['temp', 'humidity', 'wind_speed']
    )
    
    images = {
        ts: np.random.rand(128, 128, 3) for ts in timestamps
    }
    
    # Test sequence creation
    seq_length = 5
    img_sequences, meteo_sequences = create_sequences(images, meteo_data, seq_length)
    
    # Verify results
    assert isinstance(img_sequences, np.ndarray)
    assert isinstance(meteo_sequences, np.ndarray)
    assert len(img_sequences) == len(meteo_sequences)
    assert img_sequences.shape[1] == seq_length
    assert meteo_sequences.shape[1] == seq_length

def test_prepare_data():
    """Test the complete data preparation pipeline"""
    # Create test data
    os.makedirs('test_data', exist_ok=True)
    
    # Create meteorological data
    meteo_data = {
        'timestamp': ['2023-01-01 00:00', '2023-01-01 01:00', '2023-01-01 02:00'],
        'temp': [32.0, 33.0, 34.0],
        'humidity': [80.0, 82.0, 85.0]
    }
    pd.DataFrame(meteo_data).to_csv('test_data/meteo.csv', index=False)
    
    # Create test images
    os.makedirs('test_data/images', exist_ok=True)
    for i in range(3):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(f'test_data/images/2023-01-01_{i:02d}-00.png', img)
    
    # Test data preparation
    train_data, val_data, features = prepare_data(
        meteo_file='test_data/meteo.csv',
        image_dir='test_data/images',
        seq_length=2
    )
    
    # Verify results
    assert len(train_data) == 2  # images and meteo
    assert len(val_data) == 2    # images and meteo
    assert isinstance(features, list)
    
    # Clean up
    import shutil
    shutil.rmtree('test_data')

if __name__ == '__main__':
    # Run all tests
    pytest.main(['-v', 'test_preprocessing.py']) 
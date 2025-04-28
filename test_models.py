import numpy as np
import tensorflow as tf
from models import get_model, get_callbacks

def create_sample_data(image_shape, meteo_shape, batch_size=2):
    """Create sample data for testing models"""
    # Create random image data
    image_data = np.random.random((batch_size,) + image_shape)
    
    # Create random meteorological data
    meteo_data = np.random.random((batch_size,) + meteo_shape)
    
    # Create random labels
    labels = np.random.randint(0, 2, size=(batch_size, 1))
    
    return image_data, meteo_data, labels

def test_model_creation():
    """Test that all models can be created and compiled"""
    # Define input shapes
    image_shape = (8, 128, 128, 1)  # 8 timesteps, 128x128 images, 1 channel
    meteo_shape = (8, 7)  # 8 timesteps, 7 meteorological features
    
    # Test each model type
    for model_type in range(1, 5):
        print(f"\nTesting Model {model_type}")
        
        # Create model
        model = get_model(model_type, image_shape, meteo_shape)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        model.summary()
        
        # Create sample data
        image_data, meteo_data, labels = create_sample_data(image_shape, meteo_shape)
        
        # Test forward pass
        predictions = model.predict([image_data, meteo_data])
        print(f"Predictions shape: {predictions.shape}")
        print(f"Sample predictions: {predictions[:2]}")

def test_training():
    """Test that models can be trained"""
    # Define input shapes
    image_shape = (8, 128, 128, 1)
    meteo_shape = (8, 7)
    
    # Create sample data
    image_data, meteo_data, labels = create_sample_data(image_shape, meteo_shape, batch_size=32)
    
    # Test each model type
    for model_type in range(1, 5):
        print(f"\nTesting training for Model {model_type}")
        
        # Create and compile model
        model = get_model(model_type, image_shape, meteo_shape)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train for a few steps
        history = model.fit(
            [image_data, meteo_data],
            labels,
            epochs=2,
            batch_size=4,
            validation_split=0.2,
            callbacks=get_callbacks(),
            verbose=1
        )
        
        print(f"Training history: {history.history}")

def main():
    print("Testing model creation...")
    test_model_creation()
    
    print("\nTesting model training...")
    test_training()

if __name__ == '__main__':
    main() 
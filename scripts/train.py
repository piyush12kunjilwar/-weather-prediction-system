import os
import yaml
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)
import json

from utils.logger import setup_logger
from utils.error_handler import handle_error

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        handle_error(e, 'configuration')

def load_data(config, logger):
    """Load preprocessed data."""
    try:
        # Load satellite sequences
        satellite_path = os.path.join(
            config['data']['processed_data_path'],
            'satellite_images',
            f"sequences_{config['data']['satellite_images']['sequence_length']}.npy"
        )
        satellite_sequences = np.load(satellite_path)
        
        # Load meteorological sequences
        meteo_path = os.path.join(
            config['data']['processed_data_path'],
            'meteorological',
            'sequences.npy'
        )
        meteo_sequences = np.load(meteo_path)
        
        # Create dummy labels for demonstration
        # Replace this with actual label loading
        labels = np.random.randint(0, 2, size=len(satellite_sequences))
        
        logger.info(f"Loaded {len(satellite_sequences)} training sequences")
        logger.info(f"Satellite sequence shape: {satellite_sequences.shape}")
        logger.info(f"Meteorological sequence shape: {meteo_sequences.shape}")
        
        return satellite_sequences, meteo_sequences, labels
    except Exception as e:
        handle_error(e, 'data_loading', logger)

def create_model(config, logger):
    """Create the model architecture."""
    try:
        # Get model parameters
        image_size = config['data']['satellite_images']['image_size']
        sequence_length = config['data']['satellite_images']['sequence_length']
        channels = config['data']['satellite_images']['channels']
        meteo_features = len(config['data']['meteorological']['features'])
        
        # Satellite image branch
        satellite_input = tf.keras.layers.Input(
            shape=(sequence_length, *image_size, channels)
        )
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        )(satellite_input)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.MaxPooling2D((2, 2))
        )(x)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        )(x)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.MaxPooling2D((2, 2))
        )(x)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Flatten()
        )(x)
        satellite_output = tf.keras.layers.LSTM(64)(x)
        
        # Meteorological data branch
        meteo_input = tf.keras.layers.Input(
            shape=(sequence_length, meteo_features)
        )
        y = tf.keras.layers.LSTM(32)(meteo_input)
        
        # Combine branches
        combined = tf.keras.layers.concatenate([satellite_output, y])
        z = tf.keras.layers.Dense(64, activation='relu')(combined)
        z = tf.keras.layers.Dropout(0.5)(z)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(z)
        
        # Create model
        model = tf.keras.Model(
            inputs=[satellite_input, meteo_input],
            outputs=output
        )
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Created model architecture")
        model.summary(print_fn=logger.info)
        
        return model
    except Exception as e:
        handle_error(e, 'model_creation', logger)

def setup_callbacks(config, logger):
    """Setup training callbacks."""
    try:
        # Create callbacks directory
        callbacks_dir = os.path.join(
            config['paths']['callbacks'],
            datetime.now().strftime('%Y%m%d')
        )
        os.makedirs(callbacks_dir, exist_ok=True)
        
        # Setup callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(callbacks_dir, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5,
                verbose=1
            ),
            TensorBoard(
                log_dir=os.path.join(callbacks_dir, 'logs'),
                histogram_freq=1
            )
        ]
        
        logger.info("Setup training callbacks")
        
        return callbacks
    except Exception as e:
        handle_error(e, 'callback_setup', logger)

def train_model(model, data, callbacks, config, logger):
    """Train the model."""
    try:
        # Unpack data
        satellite_sequences, meteo_sequences, labels = data
        
        # Split data
        train_size = int(len(satellite_sequences) * 0.8)
        val_size = int(len(satellite_sequences) * 0.1)
        
        train_satellite = satellite_sequences[:train_size]
        train_meteo = meteo_sequences[:train_size]
        train_labels = labels[:train_size]
        
        val_satellite = satellite_sequences[train_size:train_size + val_size]
        val_meteo = meteo_sequences[train_size:train_size + val_size]
        val_labels = labels[train_size:train_size + val_size]
        
        # Train model
        history = model.fit(
            [train_satellite, train_meteo],
            train_labels,
            validation_data=([val_satellite, val_meteo], val_labels),
            epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Model training completed")
        
        return history
    except Exception as e:
        handle_error(e, 'model_training', logger)

def save_training_history(history, config, logger):
    """Save training history."""
    try:
        # Create history directory
        history_dir = os.path.join(
            config['paths']['history'],
            datetime.now().strftime('%Y%m%d')
        )
        os.makedirs(history_dir, exist_ok=True)
        
        # Save history
        history_path = os.path.join(history_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(history.history, f, indent=4)
        
        logger.info(f"Saved training history to {history_path}")
    except Exception as e:
        handle_error(e, 'history_saving', logger)

def main():
    """Main function to train the model."""
    try:
        # Setup
        config = load_config()
        logger = setup_logger('training')
        
        # Load data
        data = load_data(config, logger)
        
        # Create model
        model = create_model(config, logger)
        
        # Setup callbacks
        callbacks = setup_callbacks(config, logger)
        
        # Train model
        history = train_model(model, data, callbacks, config, logger)
        
        # Save training history
        save_training_history(history, config, logger)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        handle_error(e, 'main', logger)

if __name__ == "__main__":
    main() 
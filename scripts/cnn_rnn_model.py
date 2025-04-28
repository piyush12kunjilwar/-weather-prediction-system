import tensorflow as tf
from tensorflow.keras import layers, Model
from utils.logger import setup_logger
from utils.error_handler import handle_error

def create_satellite_branch(input_shape, config, logger):
    """Create the satellite image processing branch."""
    try:
        inputs = layers.Input(shape=input_shape)
        
        # ConvLSTM layers
        x = inputs
        for filters in config['model']['architecture']['satellite_branch']['conv_lstm_filters']:
            x = layers.ConvLSTM2D(
                filters=filters,
                kernel_size=(3, 3),
                padding='same',
                return_sequences=True
            )(x)
            x = layers.BatchNormalization()(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling3D()(x)
        
        # Dense layers
        for units in config['model']['architecture']['satellite_branch']['dense_units']:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(config['model']['architecture']['combined']['dropout_rate'])(x)
        
        return Model(inputs=inputs, outputs=x)
    except Exception as e:
        handle_error(e, 'satellite_branch_creation', logger)

def create_meteo_branch(input_shape, config, logger):
    """Create the meteorological data processing branch."""
    try:
        inputs = layers.Input(shape=input_shape)
        
        # LSTM layers
        x = inputs
        for units in config['model']['architecture']['meteo_branch']['lstm_units']:
            x = layers.LSTM(units, return_sequences=True)(x)
            x = layers.BatchNormalization()(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        for units in config['model']['architecture']['meteo_branch']['dense_units']:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(config['model']['architecture']['combined']['dropout_rate'])(x)
        
        return Model(inputs=inputs, outputs=x)
    except Exception as e:
        handle_error(e, 'meteo_branch_creation', logger)

def create_combined_model(satellite_shape, meteo_shape, config=None, logger=None):
    """
    Create the combined CNN-RNN model for rain prediction.
    
    Args:
        satellite_shape: Shape of satellite image input (time_steps, height, width, channels)
        meteo_shape: Shape of meteorological data input (time_steps, features)
        config: Configuration dictionary
        logger: Logger instance
    """
    try:
        if logger is None:
            logger = setup_logger('model_creation')
        
        if config is None:
            # Default configuration if none provided
            config = {
                'model': {
                    'architecture': {
                        'satellite_branch': {
                            'conv_lstm_filters': [32],
                            'dense_units': [32]
                        },
                        'meteo_branch': {
                            'lstm_units': [16],
                            'dense_units': [16]
                        },
                        'combined': {
                            'dense_units': [32],
                            'dropout_rate': 0.2
                        }
                    }
                }
            }
        
        # Create satellite branch
        satellite_branch = create_satellite_branch(satellite_shape, config, logger)
        logger.info("Created satellite branch")
        
        # Create meteorological branch
        meteo_branch = create_meteo_branch(meteo_shape, config, logger)
        logger.info("Created meteorological branch")
        
        # Combine branches
        satellite_input = layers.Input(shape=satellite_shape)
        meteo_input = layers.Input(shape=meteo_shape)
        
        satellite_features = satellite_branch(satellite_input)
        meteo_features = meteo_branch(meteo_input)
        
        # Concatenate features
        combined = layers.Concatenate()([satellite_features, meteo_features])
        
        # Final dense layers
        for units in config['model']['architecture']['combined']['dense_units']:
            combined = layers.Dense(units, activation='relu')(combined)
            combined = layers.BatchNormalization()(combined)
            combined = layers.Dropout(config['model']['architecture']['combined']['dropout_rate'])(combined)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid')(combined)
        
        # Create and return model
        model = Model(inputs=[satellite_input, meteo_input], outputs=output)
        logger.info("Created combined model")
        
        return model
    except Exception as e:
        handle_error(e, 'model_creation', logger)

def get_callbacks():
    """
    Get training callbacks including EarlyStopping and ReduceLROnPlateau.
    
    Returns:
        List of Keras callbacks
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ] 
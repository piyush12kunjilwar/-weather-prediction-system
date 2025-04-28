import tensorflow as tf
from tensorflow.keras import layers, models

def create_simple_model(input_shape, num_classes=1):
    """
    A simpler model architecture with fewer layers and parameters
    """
    # Satellite image input
    satellite_input = layers.Input(shape=input_shape)
    
    # Single ConvLSTM2D layer
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=False
    )(satellite_input)
    x = layers.BatchNormalization()(x)
    
    # Flatten and Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs=satellite_input, outputs=output)
    return model

def create_model1_shallow(input_shape, num_classes=1):
    """
    Model 1: ConvLSTM2D + LSTM (Shallow)
    """
    # Satellite image input
    satellite_input = layers.Input(shape=input_shape)
    
    # ConvLSTM2D layers
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=True
    )(satellite_input)
    x = layers.BatchNormalization()(x)
    
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=False
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Flatten and Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs=satellite_input, outputs=output)
    return model

def create_model2(input_shape, num_classes=1):
    """
    Model 2: Conv3D + ConvLSTM2D + LSTM (Shallow)
    Architecture:
    - Initial Conv3D layer for spatial feature extraction
    - Single ConvLSTM2D layer for spatiotemporal features
    - Shallow LSTM structure with one dense layer
    - Moderate number of filters (32)
    """
    # Satellite image input
    satellite_input = layers.Input(shape=input_shape)
    
    # Conv3D layer for spatial feature extraction
    x = layers.Conv3D(
        filters=32,
        kernel_size=(3, 3, 3),
        padding='same',
        activation='relu'
    )(satellite_input)
    x = layers.BatchNormalization()(x)
    
    # ConvLSTM2D layer for spatiotemporal features
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=False
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Shallow LSTM structure
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs=satellite_input, outputs=output)
    return model

def create_model3(input_shape, num_classes=1):
    """
    Model 3: ConvLSTM2D + LSTM (Deep)
    """
    # Satellite image input
    satellite_input = layers.Input(shape=input_shape)
    
    # First ConvLSTM2D layer
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=True
    )(satellite_input)
    x = layers.BatchNormalization()(x)
    
    # Second ConvLSTM2D layer
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=False
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Deep LSTM layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs=satellite_input, outputs=output)
    return model

def create_model4(input_shape, num_classes=1):
    """
    Model 4: Conv3D + ConvLSTM2D + LSTM (Deep)
    """
    # Satellite image input
    satellite_input = layers.Input(shape=input_shape)
    
    # Conv3D layers for initial feature extraction
    x = layers.Conv3D(
        filters=64,
        kernel_size=(3, 3, 3),
        padding='same',
        activation='relu'
    )(satellite_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(
        filters=32,
        kernel_size=(3, 3, 3),
        padding='same',
        activation='relu'
    )(x)
    x = layers.BatchNormalization()(x)
    
    # ConvLSTM2D layer
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=False
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Deep LSTM layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs=satellite_input, outputs=output)
    return model

def create_meteo_lstm(input_shape, num_classes=1):
    """
    LSTM model for meteorological data
    """
    # Meteorological data input
    meteo_input = layers.Input(shape=input_shape)
    
    # LSTM layers
    x = layers.LSTM(64, return_sequences=True)(meteo_input)
    x = layers.BatchNormalization()(x)
    
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.BatchNormalization()(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs=meteo_input, outputs=output)
    return model

def create_combined_model(satellite_shape, meteo_shape, num_classes=1):
    """
    Combined model that integrates both satellite imagery and meteorological data
    """
    # Satellite image input
    satellite_input = layers.Input(shape=satellite_shape)
    
    # Meteorological data input
    meteo_input = layers.Input(shape=meteo_shape)
    
    # Process satellite data
    x_sat = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=True
    )(satellite_input)
    x_sat = layers.BatchNormalization()(x_sat)
    
    x_sat = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=False
    )(x_sat)
    x_sat = layers.BatchNormalization()(x_sat)
    x_sat = layers.Flatten()(x_sat)
    
    # Process meteorological data
    x_meteo = layers.LSTM(64, return_sequences=True)(meteo_input)
    x_meteo = layers.BatchNormalization()(x_meteo)
    
    x_meteo = layers.LSTM(128, return_sequences=False)(x_meteo)
    x_meteo = layers.BatchNormalization()(x_meteo)
    
    # Combine the features
    x = layers.Concatenate()([x_sat, x_meteo])
    
    # Final dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs=[satellite_input, meteo_input], outputs=output)
    return model 
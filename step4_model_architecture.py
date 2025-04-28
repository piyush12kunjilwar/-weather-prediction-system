"""
Step 4: Model Architecture
"""

import tensorflow as tf

def create_advanced_rnn_model(input_shape, num_classes=2):
    """
    Create an advanced RNN model with attention mechanism.
    
    Architecture:
    - Input layer
    - LSTM layers with attention
    - Dense layers
    - Output layer
    """
    print("Creating advanced RNN model...")
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # LSTM layers with attention
    x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Attention mechanism
    attention = tf.keras.layers.Dense(1, activation='tanh')(x)
    attention = tf.keras.layers.Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax')(attention)
    attention = tf.keras.layers.RepeatVector(128)(attention)
    attention = tf.keras.layers.Permute([2, 1])(attention)
    
    # Apply attention
    x = tf.keras.layers.multiply([x, attention])
    x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(x)
    
    # Dense layers
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_callbacks():
    """Create training callbacks for model optimization."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    return callbacks 
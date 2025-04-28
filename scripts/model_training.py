import os
import yaml
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
from model_architectures import create_combined_model
import traceback

from utils.logger import setup_logger
from utils.error_handler import handle_error, validate_data

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        handle_error(e, 'configuration')

def create_directories(config):
    """Create necessary directories."""
    try:
        for path in config['paths'].values():
            os.makedirs(path, exist_ok=True)
    except Exception as e:
        handle_error(e, 'directory_creation')

def load_data(config, logger):
    """Load and preprocess data."""
    try:
        # Load satellite sequences
        satellite_path = os.path.join(
            config['data']['processed_data_path'],
            'satellite_images',
            f"sequences_{config['data']['satellite_images']['sequence_length']}.npy"
        )
        satellite_sequences = np.load(satellite_path)
        logger.info(f"Loaded satellite sequences with shape: {satellite_sequences.shape}")

        # Load meteorological data
        meteo_path = os.path.join(
            config['data']['processed_data_path'],
            'meteorological_data.csv'
        )
        meteo_data = pd.read_csv(meteo_path)
        meteo_features = meteo_data.drop('timestamp', axis=1).values
        logger.info(f"Loaded meteorological data with shape: {meteo_features.shape}")

        # Load labels
        labels_path = os.path.join(
            config['data']['processed_data_path'],
            'rain_labels.npy'
        )
        labels = np.load(labels_path)
        logger.info(f"Loaded labels with shape: {labels.shape}")

        return satellite_sequences, meteo_features, labels
    except Exception as e:
        handle_error(e, 'data_loading', logger)

def preprocess_data(satellite_sequences, meteo_features, labels, config, logger):
    """Preprocess data for model training."""
    try:
        # Use only the first matching length for all data
        min_length = min(len(satellite_sequences), len(meteo_features), len(labels))
        
        # Process satellite data
        X_sat = satellite_sequences[:min_length].astype(np.float32)
        if len(X_sat.shape) == 4:  # If missing channel dimension
            X_sat = X_sat[..., np.newaxis]
        
        # Process meteorological data
        X_meteo = meteo_features[:min_length].astype(np.float32)
        X_meteo = np.repeat(X_meteo[:, np.newaxis, :], 
                          config['data']['satellite_images']['sequence_length'], 
                          axis=1)
        
        # Process labels
        y = labels[:min_length].astype(np.float32)
        
        logger.info(f"Processed data shapes:")
        logger.info(f"Satellite: {X_sat.shape}")
        logger.info(f"Meteorological: {X_meteo.shape}")
        logger.info(f"Labels: {y.shape}")
        
        return X_sat, X_meteo, y
    except Exception as e:
        handle_error(e, 'data_preprocessing', logger)

def train_and_evaluate_model(model, X_sat_train, X_meteo_train, y_train, 
                           X_sat_test, X_meteo_test, y_test, 
                           config, logger):
    """Train and evaluate the model."""
    try:
        logger.info("Starting model training...")
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config['model']['training']['learning_rate']
            ),
            loss=config['model']['training']['loss'],
            metrics=config['model']['training']['metrics']
        )
        
        # Train model
        history = model.fit(
            [X_sat_train, X_meteo_train], y_train,
            validation_data=([X_sat_test, X_meteo_test], y_test),
            epochs=config['model']['training']['epochs'],
            batch_size=config['model']['training']['batch_size'],
            verbose=1
        )
        
        # Save model
        model_path = os.path.join(
            config['paths']['models'],
            f"Combined_Model_seq{config['data']['satellite_images']['sequence_length']}.h5"
        )
        model.save(model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Evaluate model
        y_pred = (model.predict([X_sat_test, X_meteo_test]) > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        f1 = f1_score(y_test, y_pred)
        
        # Log results
        logger.info(f"Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        # Generate visualizations
        generate_visualizations(history, y_test, y_pred, config, logger)
        
        return history, accuracy, f1
    except Exception as e:
        handle_error(e, 'model_training', logger)

def generate_visualizations(history, y_test, y_pred, config, logger):
    """Generate and save visualizations."""
    try:
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(
            config['paths']['visualizations'],
            f"confusion_matrix_seq{config['data']['satellite_images']['sequence_length']}.png"
        ))
        plt.close()
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(
            config['paths']['visualizations'],
            f"training_history_seq{config['data']['satellite_images']['sequence_length']}.png"
        ))
        plt.close()
        
        logger.info("Generated visualizations")
    except Exception as e:
        handle_error(e, 'visualization', logger)

def main():
    """Main function to train the model."""
    try:
        # Setup
        config = load_config()
        logger = setup_logger('model_training')
        create_directories(config)
        
        # Load and preprocess data
        satellite_sequences, meteo_features, labels = load_data(config, logger)
        X_sat, X_meteo, y = preprocess_data(
            satellite_sequences, meteo_features, labels, config, logger
        )
        
        # Split data
        X_sat_train, X_sat_test, X_meteo_train, X_meteo_test, y_train, y_test = train_test_split(
            X_sat, X_meteo, y,
            test_size=config['model']['evaluation']['test_size'],
            random_state=config['model']['evaluation']['random_state']
        )
        
        # Create and train model
        model = create_combined_model(
            satellite_shape=X_sat.shape[1:],
            meteo_shape=X_meteo.shape[1:]
        )
        
        logger.info("Model created successfully")
        model.summary()
        
        # Train and evaluate
        history, accuracy, f1 = train_and_evaluate_model(
            model, X_sat_train, X_meteo_train, y_train,
            X_sat_test, X_meteo_test, y_test,
            config, logger
        )
        
        logger.info("Training completed successfully")
        logger.info(f"Final Accuracy: {accuracy:.4f}")
        logger.info(f"Final F1 Score: {f1:.4f}")
        
    except Exception as e:
        handle_error(e, 'main', logger)

if __name__ == "__main__":
    main() 
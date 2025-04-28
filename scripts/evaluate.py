import os
import yaml
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)
import seaborn as sns
import json
from datetime import datetime

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

def load_model(config, logger):
    """Load trained model."""
    try:
        model_path = os.path.join(
            config['paths']['models'],
            'best_model.h5'
        )
        model = tf.keras.models.load_model(model_path)
        logger.info("Loaded trained model")
        return model
    except Exception as e:
        handle_error(e, 'model_loading', logger)

def load_test_data(config, logger):
    """Load test data."""
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
        
        # Split into train and test sets
        train_size = int(len(satellite_sequences) * 0.8)
        val_size = int(len(satellite_sequences) * 0.1)
        
        test_satellite = satellite_sequences[train_size + val_size:]
        test_meteo = meteo_sequences[train_size + val_size:]
        test_labels = labels[train_size + val_size:]
        
        logger.info(f"Loaded {len(test_satellite)} test sequences")
        logger.info(f"Satellite sequence shape: {test_satellite.shape}")
        logger.info(f"Meteorological sequence shape: {test_meteo.shape}")
        
        return test_satellite, test_meteo, test_labels
    except Exception as e:
        handle_error(e, 'data_loading', logger)

def evaluate_model(model, data, config, logger):
    """Evaluate the model."""
    try:
        # Unpack data
        test_satellite, test_meteo, test_labels = data
        
        # Make predictions
        predictions = model.predict([test_satellite, test_meteo])
        predicted_labels = (predictions > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predicted_labels)
        precision = precision_score(test_labels, predicted_labels)
        recall = recall_score(test_labels, predicted_labels)
        f1 = f1_score(test_labels, predicted_labels)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(test_labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        # Calculate confusion matrix
        cm = confusion_matrix(test_labels, predicted_labels)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        logger.info("Model evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return metrics, fpr, tpr, cm
    except Exception as e:
        handle_error(e, 'model_evaluation', logger)

def plot_evaluation_results(metrics, fpr, tpr, cm, config, logger):
    """Plot evaluation results."""
    try:
        # Create plots directory
        plots_dir = os.path.join(
            config['paths']['visualizations'],
            'evaluation'
        )
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {metrics["roc_auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(plots_dir, 'roc_curve.png'))
        plt.close()
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Plot metrics
        plt.figure(figsize=(8, 6))
        metrics_to_plot = {
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score']
        }
        plt.bar(metrics_to_plot.keys(), metrics_to_plot.values())
        plt.ylim(0, 1)
        plt.title('Model Evaluation Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'metrics.png'))
        plt.close()
        
        logger.info("Saved evaluation plots")
    except Exception as e:
        handle_error(e, 'plotting', logger)

def save_evaluation_results(metrics, config, logger):
    """Save evaluation results."""
    try:
        # Create results directory
        results_dir = os.path.join(
            config['paths']['evaluation'],
            datetime.now().strftime('%Y%m%d')
        )
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics
        metrics_path = os.path.join(results_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Saved evaluation metrics to {metrics_path}")
    except Exception as e:
        handle_error(e, 'results_saving', logger)

def main():
    """Main function to evaluate the model."""
    try:
        # Setup
        config = load_config()
        logger = setup_logger('evaluation')
        
        # Load model
        model = load_model(config, logger)
        
        # Load test data
        data = load_test_data(config, logger)
        
        # Evaluate model
        metrics, fpr, tpr, cm = evaluate_model(model, data, config, logger)
        
        # Plot results
        plot_evaluation_results(metrics, fpr, tpr, cm, config, logger)
        
        # Save results
        save_evaluation_results(metrics, config, logger)
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        handle_error(e, 'main', logger)

if __name__ == "__main__":
    main() 
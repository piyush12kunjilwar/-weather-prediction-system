import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import prepare_data
from models import (
    create_model_1,
    create_model_2,
    create_model_3,
    create_model_4,
    get_callbacks,
    get_model
)
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from datetime import datetime
from data_loader import DataLoader
import json
from typing import Dict, Tuple, List
import argparse

def plot_training_history(history, save_path):
    """Plot training and validation loss/accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, X_test, y_test, save_dir):
    """Evaluate model and save results."""
    # Make predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, os.path.join(save_dir, 'confusion_matrix.png'))
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)
    
    return metrics

def train_and_evaluate(
    model_name: str,
    meteo_file: str,
    image_dir: str,
    seq_length: int,
    target_size: tuple = (128, 128),
    batch_size: int = 32,
    epochs: int = 100,
    save_dir: str = 'results'
):
    """
    Train and evaluate a model.
    
    Args:
        model_name: Name of the model to train
        meteo_file: Path to meteorological data
        image_dir: Path to satellite images
        seq_length: Length of sequences
        target_size: Target size for images
        batch_size: Batch size for training
        epochs: Number of epochs
        save_dir: Directory to save results
    """
    # Create save directory
    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Prepare data
    print("Preparing data...")
    train_data, val_data, features = prepare_data(
        meteo_file=meteo_file,
        image_dir=image_dir,
        seq_length=seq_length,
        target_size=target_size
    )
    
    # Get model
    print(f"Creating {model_name}...")
    if model_name == 'model_1':
        model = create_model_1(
            input_shape=(seq_length, *target_size, 3),
            meteo_shape=(seq_length, len(features))
        )
    elif model_name == 'model_2':
        model = create_model_2(
            input_shape=(seq_length, *target_size, 3),
            meteo_shape=(seq_length, len(features))
        )
    elif model_name == 'model_3':
        model = create_model_3(
            input_shape=(seq_length, *target_size, 3),
            meteo_shape=(seq_length, len(features))
        )
    elif model_name == 'model_4':
        model = create_model_4(
            input_shape=(seq_length, *target_size, 3),
            meteo_shape=(seq_length, len(features))
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Train model
    print("Training model...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=get_callbacks()
    )
    
    # Plot training history
    plot_training_history(history, os.path.join(model_save_dir, 'training_history.png'))
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, val_data[0], val_data[1], model_save_dir)
    
    # Save model
    model.save(os.path.join(model_save_dir, 'model.h5'))
    
    print(f"Results saved to {model_save_dir}")
    print(f"Metrics: {metrics}")
    
    return model, metrics

class Trainer:
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        model_type: int,
        meteo_window: int = 24,
        image_window: int = 8,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """
        Initialize the trainer
        
        Args:
            data_dir: Directory containing data
            output_dir: Directory to save model and results
            model_type: Type of model to train (1-4)
            meteo_window: Number of timesteps for meteorological data
            image_window: Number of timesteps for satellite images
            batch_size: Batch size for training
            learning_rate: Initial learning rate
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model_type = model_type
        self.meteo_window = meteo_window
        self.image_window = image_window
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data loader
        self.data_loader = DataLoader(
            data_dir=data_dir,
            image_dir='processed/satellite_images',
            meteo_file='processed/meteorological_data.csv'
        )
        
        # Load and preprocess data
        self._prepare_data()
        
        # Create model
        self._create_model()
    
    def _prepare_data(self):
        """Prepare data for training"""
        print("Preparing data...")
        
        # Create sequences
        self.image_sequences, self.meteo_sequences, self.labels = self.data_loader.create_sequences(
            meteo_window=self.meteo_window,
            image_window=self.image_window
        )
        
        # Split data
        self.splits = self.data_loader.split_data(
            self.image_sequences,
            self.meteo_sequences,
            self.labels
        )
    
    def _create_model(self):
        """Create and compile model"""
        print(f"Creating model type {self.model_type}...")
        
        # Get input shapes
        image_shape = (self.image_window, 128, 128, 1)
        meteo_shape = (self.meteo_window, 7)  # 7 meteorological features
        
        # Create model
        self.model = get_model(
            model_type=self.model_type,
            image_shape=image_shape,
            meteo_shape=meteo_shape
        )
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        # Print model summary
        self.model.summary()
    
    def train(self, epochs: int = 100):
        """
        Train the model
        
        Args:
            epochs: Number of epochs to train
        """
        print("Starting training...")
        
        # Create callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(self.output_dir, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True
            ),
            TensorBoard(
                log_dir=os.path.join(self.output_dir, 'logs'),
                histogram_freq=1
            )
        ] + get_callbacks()
        
        # Train model
        history = self.model.fit(
            [self.splits['train'][0], self.splits['train'][1]],
            self.splits['train'][2],
            validation_data=(
                [self.splits['val'][0], self.splits['val'][1]],
                self.splits['val'][2]
            ),
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history
        self._save_history(history)
        
        # Plot training curves
        self._plot_training_curves(history)
        
        # Evaluate on test set
        self._evaluate_model()
    
    def _save_history(self, history):
        """Save training history"""
        history_dict = {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy'],
            'auc': history.history['auc'],
            'val_auc': history.history['val_auc'],
            'precision': history.history['precision'],
            'val_precision': history.history['val_precision'],
            'recall': history.history['recall'],
            'val_recall': history.history['val_recall']
        }
        
        with open(os.path.join(self.output_dir, 'history.json'), 'w') as f:
            json.dump(history_dict, f)
    
    def _plot_training_curves(self, history):
        """Plot training curves"""
        metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall']
        
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 3, i)
            plt.plot(history.history[metric], label=f'Training {metric}')
            plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
            plt.title(f'{metric.capitalize()} vs. Epoch')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'))
        plt.close()
    
    def _evaluate_model(self):
        """Evaluate model on test set"""
        print("\nEvaluating model on test set...")
        
        # Evaluate model
        test_loss, test_acc, test_auc, test_precision, test_recall = self.model.evaluate(
            [self.splits['test'][0], self.splits['test'][1]],
            self.splits['test'][2],
            verbose=1
        )
        
        # Print metrics
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        
        # Save metrics
        metrics = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_auc': float(test_auc),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall)
        }
        
        with open(os.path.join(self.output_dir, 'test_metrics.json'), 'w') as f:
            json.dump(metrics, f)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train rainfall prediction models')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--model_type', type=int, choices=[1, 2, 3, 4], required=True, help='Model type (1-4)')
    parser.add_argument('--meteo_window', type=int, default=24, help='Meteorological window size')
    parser.add_argument('--image_window', type=int, default=8, help='Image window size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(
        args.output_dir,
        f'model_{args.model_type}_meteo{args.meteo_window}_img{args.image_window}_{timestamp}'
    )
    
    # Create and train model
    trainer = Trainer(
        data_dir=args.data_dir,
        output_dir=output_dir,
        model_type=args.model_type,
        meteo_window=args.meteo_window,
        image_window=args.image_window,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    trainer.train(epochs=args.epochs)

if __name__ == '__main__':
    main() 
"""
Step 5: Model Training and Evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import os
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)
from datetime import datetime
import tensorflow as tf

def create_callbacks(model_save_path='models', patience=10):
    """
    Create training callbacks for model optimization.
    
    Args:
        model_save_path: Directory to save model checkpoints
        patience: Number of epochs to wait before early stopping
    
    Returns:
        List of callbacks
    """
    try:
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            # Save best model
            ModelCheckpoint(
                filepath=os.path.join(model_save_path, f'best_model_{timestamp}.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=patience//2,
                min_lr=1e-6,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=os.path.join('logs', timestamp),
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        return callbacks
        
    except Exception as e:
        print(f"Error creating callbacks: {str(e)}")
        raise

def train_model(model, X_train, y_train=None, X_val=None, y_val=None, class_weights=None, batch_size=32, epochs=50):
    """
    Train the model with validation and callbacks.
    
    Args:
        model: Compiled Keras model
        X_train: Training data (can be a tf.data.Dataset)
        y_train: Training labels (not needed if X_train is a dataset)
        X_val: Validation data (can be a tf.data.Dataset)
        y_val: Validation labels (not needed if X_val is a dataset)
        class_weights: Dictionary of class weights for handling imbalance
        batch_size: Batch size for training
        epochs: Number of training epochs
    """
    try:
        print("Starting model training...")
        
        # Check if inputs are tf.data.Dataset
        is_dataset = isinstance(X_train, tf.data.Dataset)
        
        if is_dataset:
            print("Using TensorFlow Dataset API for training")
            train_data = X_train
            val_data = X_val
        else:
            print(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
            print(f"Validation data shape: X_val: {X_val.shape}, y_val: {y_val.shape}")
            train_data = (X_train, y_train)
            val_data = (X_val, y_val) if X_val is not None else None
        
        if class_weights:
            print(f"Using class weights: {class_weights}")
        
        # Get callbacks
        callbacks = create_callbacks()
        
        # Train model
        history = model.fit(
            train_data,
            validation_data=val_data,
            batch_size=None if is_dataset else batch_size,  # batch_size is handled by the dataset
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        print("Model training completed successfully!")
        return history, model
        
    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test=None):
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained Keras model
        X_test: Test data (can be a tf.data.Dataset)
        y_test: Test labels (not needed if X_test is a dataset)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        print("Starting model evaluation...")
        
        # Check if input is tf.data.Dataset
        is_dataset = isinstance(X_test, tf.data.Dataset)
        
        if is_dataset:
            print("Using TensorFlow Dataset API for evaluation")
            # Get predictions from the dataset
            y_true = []
            y_pred_all = []
            
            # Iterate through the dataset to get predictions
            for batch_x, batch_y in X_test:
                y_true.extend(batch_y.numpy())
                y_pred = model.predict(batch_x, verbose=0)
                y_pred_all.extend(y_pred)
            
            y_true = np.array(y_true)
            y_pred = np.array(y_pred_all)
        else:
            print(f"Test data shape: X_test: {X_test.shape}")
            # Get predictions
            print("Generating predictions...")
            y_pred = model.predict(X_test, verbose=1)
            y_true = y_test
        
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print("Calculating metrics...")
        # Calculate metrics
        accuracy = np.mean(y_pred_classes == y_true)
        print(f"Model accuracy: {accuracy:.4f}")
        
        # Calculate precision, recall, and F1-score
        conf_matrix = confusion_matrix(y_true, y_pred_classes)
        class_report = classification_report(y_true, y_pred_classes, output_dict=True)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])
        roc_auc = auc(fpr, tpr)
        print(f"ROC AUC Score: {roc_auc:.4f}")
        
        # Calculate additional metrics
        tn, fp, fn, tp = conf_matrix.ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        
        metrics = {
            'accuracy': accuracy,
            'precision': class_report['weighted avg']['precision'],
            'recall': class_report['weighted avg']['recall'],
            'f1_score': class_report['weighted avg']['f1-score'],
            'specificity': specificity,
            'sensitivity': sensitivity,
            'roc_auc': roc_auc,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'roc_curve': (fpr, tpr)
        }
        
        # Print detailed metrics
        print("\nDetailed Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"Error in evaluate_model: {str(e)}")
        raise

def plot_training_history(history, save_dir='visualizations'):
    """Plot training history with error handling."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        # Clear any existing plots
        plt.close('all')
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'))
        plt.close()
        print("Training history plot saved successfully.")
        
    except Exception as e:
        print(f"Error plotting training history: {str(e)}")

def plot_confusion_matrix(y_true, y_pred, save_dir='visualizations'):
    """Plot confusion matrix with error handling."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        # Clear any existing plots
        plt.close('all')
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()
        print("Confusion matrix plot saved successfully.")
        
    except Exception as e:
        print(f"Error plotting confusion matrix: {str(e)}")

def plot_roc_curve(y_true, y_pred_proba, save_dir='visualizations'):
    """Plot ROC curve with error handling."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        # Clear any existing plots
        plt.close('all')
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
        plt.close()
        print("ROC curve plot saved successfully.")
        
    except Exception as e:
        print(f"Error plotting ROC curve: {str(e)}")

def evaluate_and_visualize_results(model, history, X_test, y_test):
    """Evaluate model and create visualizations with error handling."""
    try:
        print("\n=== Starting Model Evaluation and Visualization ===")
        
        # Create visualizations directory
        viz_dir = 'visualizations'
        os.makedirs(viz_dir, exist_ok=True)
        print(f"Created visualization directory at: {os.path.abspath(viz_dir)}")
        
        # Plot training history
        print("\nGenerating training history plot...")
        plot_training_history(history)
        
        print("\nGenerating predictions...")
        # Generate predictions
        y_pred = model.predict(X_test, verbose=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print("\nGenerating confusion matrix...")
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred_classes)
        
        print("\nGenerating ROC curve...")
        # Plot ROC curve
        plot_roc_curve(y_test, y_pred)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes))
        
        print("\n=== Evaluation and Visualization Completed ===")
        
    except Exception as e:
        print(f"Error in evaluate_and_visualize_results: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        raise 
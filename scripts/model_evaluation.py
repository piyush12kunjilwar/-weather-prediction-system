import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add parent directory to Python path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from process_satellite_images import load_cached_data

def create_model(input_shape, learning_rate=0.001, dropout_rate=0.3):
    """Create the CNN-LSTM model with given hyperparameters."""
    model = models.Sequential([
        # CNN layers
        layers.Conv3D(32, (3, 1, 1), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv3D(64, (3, 1, 1), activation='relu'),
        layers.BatchNormalization(),
        
        # Reshape for LSTM
        layers.Reshape((-1, 64)),
        
        # LSTM layers
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.LSTM(64),
        layers.Dropout(dropout_rate),
        
        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )
    
    return model

def plot_training_history(history, fold=None):
    """Plot training history metrics."""
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        if metric in history.history:
            axes[idx].plot(history.history[metric], label='train')
            axes[idx].plot(history.history[f'val_{metric}'], label='validation')
            axes[idx].set_title(f'Model {metric}')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric)
            axes[idx].legend()
    
    plt.tight_layout()
    fold_str = f'_fold_{fold}' if fold is not None else ''
    plt.savefig(f'results/training_history{fold_str}.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, fold=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred.round())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    fold_str = f'_fold_{fold}' if fold is not None else ''
    plt.savefig(f'results/confusion_matrix{fold_str}.png')
    plt.close()

def evaluate_model(model, X_test, y_test, fold=None):
    """Evaluate model and generate classification report."""
    y_pred = model.predict(X_test)
    
    # Generate classification report
    report = classification_report(y_test, y_pred.round(), output_dict=True)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, fold)
    
    return report

def cross_validate(X, y, n_splits=5, **model_params):
    """Perform k-fold cross-validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nTraining fold {fold + 1}/{n_splits}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Calculate class weights for imbalanced data
        total = len(y_train)
        neg = np.sum(y_train == 0)
        pos = np.sum(y_train == 1)
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        class_weights = {0: weight_for_0, 1: weight_for_1}
        
        # Create and train model
        model = create_model(input_shape=X.shape[1:], **model_params)
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(
                f'models/best_model_fold_{fold}.h5',
                save_best_only=True
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training history
        plot_training_history(history, fold)
        
        # Evaluate model
        results = evaluate_model(model, X_val, y_val, fold)
        fold_results.append(results)
        
        # Save fold results
        with open(f'results/fold_{fold}_results.json', 'w') as f:
            json.dump(results, f, indent=4)
    
    return fold_results

def hyperparameter_tuning(X, y):
    """Perform grid search for hyperparameter tuning."""
    param_grid = {
        'learning_rate': [0.1, 0.01, 0.001],
        'dropout_rate': [0.2, 0.3, 0.4]
    }
    
    best_score = 0
    best_params = None
    results = []
    
    # Split data for tuning
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    for lr in param_grid['learning_rate']:
        for dr in param_grid['dropout_rate']:
            print(f"\nTrying learning_rate={lr}, dropout_rate={dr}")
            
            model = create_model(
                input_shape=X.shape[1:],
                learning_rate=lr,
                dropout_rate=dr
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=30,
                batch_size=32,
                verbose=0
            )
            
            # Get best validation score
            val_score = max(history.history['val_accuracy'])
            results.append({
                'learning_rate': lr,
                'dropout_rate': dr,
                'val_accuracy': val_score
            })
            
            if val_score > best_score:
                best_score = val_score
                best_params = {'learning_rate': lr, 'dropout_rate': dr}
    
    # Save tuning results
    with open('results/hyperparameter_tuning_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return best_params, X_train, y_train, X_val, y_val

def main():
    """Main function to run model evaluation and optimization."""
    # Create necessary directories
    Path('results').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    
    # Load cached data
    print("Loading cached data...")
    X, y, metadata = load_cached_data()
    
    if X is None:
        print("Error: No cached data found")
        return
    
    print("\nData shapes:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    # Perform hyperparameter tuning
    print("\nPerforming hyperparameter tuning...")
    best_params, X_train, y_train, X_val, y_val = hyperparameter_tuning(X, y)
    print(f"Best parameters: {best_params}")
    
    # Perform cross-validation with best parameters
    print("\nPerforming cross-validation...")
    fold_results = cross_validate(X, y, n_splits=5, **best_params)
    
    # Calculate and save average results
    avg_results = {
        metric: np.mean([fold[metric] for fold in fold_results])
        for metric in fold_results[0].keys()
    }
    
    with open('results/average_results.json', 'w') as f:
        json.dump(avg_results, f, indent=4)
    
    print("\nEvaluation complete! Results saved in 'results' directory.")

if __name__ == "__main__":
    main() 
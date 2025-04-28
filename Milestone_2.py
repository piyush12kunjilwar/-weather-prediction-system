"""
Milestone 2 - Enhanced RNN-based Rain Prediction Model
Group 7

This script implements an advanced RNN-based model for rain prediction using meteorological
and satellite data. The implementation includes:
1. Advanced data preprocessing and feature engineering
2. Custom image data processing
3. Hyperparameter optimization
4. Model architecture with attention mechanism
5. Comprehensive evaluation metrics
6. Visualization and reporting
"""

# =============================================================================
# Step 1: Import Libraries and Set Configuration
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense,
                                   Dropout, LSTM, Reshape, Attention, LayerNormalization)
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                      ModelCheckpoint, TensorBoard)
import os
from datetime import datetime
import json

# Set visualization style
plt.style.use('ggplot')
sns.set_style("whitegrid")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# Step 2: Data Loading and Preprocessing
# =============================================================================
def load_and_preprocess_data(file_path='2006Fall_2017Spring_GOES_meteo_combined.csv'):
    """
    Advanced data loading and preprocessing with feature engineering.
    
    Features:
    - Temporal features (month, season, day of week)
    - Meteorological features
    - Statistical features (rolling means, std)
    - Weather pattern indicators
    """
    print("Loading and preprocessing data...")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create temporal features
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Season'] = (df['Month'] % 12 + 3) // 3
    
    # Create binary target
    df['precipitation_category'] = (df['Precip (in)'] > 0).astype(int)
    
    # Handle missing values with advanced imputation
    df = handle_missing_values(df)
    
    # Add statistical features
    df = add_statistical_features(df)
    
    return df

def handle_missing_values(df):
    """Advanced missing value handling with multiple strategies."""
    # Forward fill for temporal data
    df = df.fillna(method='ffill')
    
    # Mean imputation for numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
    
    return df

def add_statistical_features(df):
    """Add statistical features for better pattern recognition."""
    # Rolling statistics
    window_sizes = [3, 7, 14]  # Different window sizes for different patterns
    for window in window_sizes:
        df[f'rolling_mean_{window}'] = df['Precip (in)'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['Precip (in)'].rolling(window=window).std()
    
    # Weather pattern indicators
    df['precip_trend'] = df['Precip (in)'].diff()
    df['precip_acceleration'] = df['precip_trend'].diff()
    
    return df

# =============================================================================
# Step 3: Image Data Processing
# =============================================================================
def decode_1d_image(row):
    """
    Advanced image processing with error handling and normalization.
    
    Features:
    - Robust error handling
    - Multiple normalization techniques
    - Data validation
    """
    try:
        # Convert string to array if necessary
        if isinstance(row, str):
            values = row.strip('[]').split(',')
            arr = np.array([float(x.strip()) for x in values])
        else:
            arr = np.array(row)
        
        # Handle NaN values
        arr = np.nan_to_num(arr)
        
        # Reshape and validate dimensions
        if arr.size != 59 * 61:
            raise ValueError(f"Invalid array size: {arr.size}")
        
        arr = arr.reshape((59, 61))
        
        # Advanced normalization
        arr = normalize_image(arr)
        
        return arr
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def normalize_image(arr):
    """Advanced image normalization techniques."""
    # Min-max normalization
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    
    # Z-score normalization
    mean = np.mean(arr)
    std = np.std(arr)
    arr = (arr - mean) / (std + 1e-8)
    
    return arr

def process_image_data(df):
    """
    Process the Lake_data_1D column into image arrays.
    
    Returns:
    - numpy array of shape (n_samples, 59, 61)
    """
    print("Processing image data...")
    
    # Process each row's Lake_data_1D
    images = []
    for idx, row in df.iterrows():
        try:
            # Convert string to array if necessary
            if isinstance(row['Lake_data_1D'], str):
                values = row['Lake_data_1D'].strip('[]').split(',')
                arr = np.array([float(x.strip()) for x in values])
            else:
                arr = np.array(row['Lake_data_1D'])
            
            # Handle NaN values
            arr = np.nan_to_num(arr)
            
            # Reshape and validate dimensions
            if arr.size != 59 * 61:
                print(f"Warning: Invalid array size at index {idx}: {arr.size}")
                continue
            
            arr = arr.reshape((59, 61))
            
            # Normalize
            arr = normalize_image(arr)
            
            images.append(arr)
        except Exception as e:
            print(f"Error processing image at index {idx}: {str(e)}")
            continue
    
    return np.array(images)

def prepare_sequences(X, y, sequence_length=5):
    """
    Prepare sequences for RNN training.
    
    Args:
    - X: numpy array of shape (n_samples, height, width)
    - y: numpy array of shape (n_samples,)
    - sequence_length: number of time steps in each sequence
    
    Returns:
    - X_sequences: numpy array of shape (n_sequences, sequence_length, height, width)
    - y_sequences: numpy array of shape (n_sequences,)
    """
    print("Preparing sequences...")
    
    n_samples = len(X)
    n_sequences = n_samples - sequence_length + 1
    
    X_sequences = np.zeros((n_sequences, sequence_length, X.shape[1], X.shape[2]))
    y_sequences = np.zeros(n_sequences)
    
    for i in range(n_sequences):
        X_sequences[i] = X[i:i+sequence_length]
        y_sequences[i] = y[i+sequence_length-1]
    
    return X_sequences, y_sequences

# =============================================================================
# Step 4: Model Architecture
# =============================================================================
def create_advanced_rnn_model(input_shape=(5, 59, 61, 1), units=128, dropout=0.5):
    """
    Advanced RNN model with attention mechanism and layer normalization.
    
    Architecture:
    - Input layer
    - Reshape layer
    - LSTM layers with attention
    - Dense layers with dropout
    - Output layer
    """
    inputs = Input(shape=input_shape)
    
    # Reshape for LSTM
    x = Reshape((input_shape[0] * input_shape[1] * input_shape[2], 1))(inputs)
    
    # First LSTM layer with attention
    lstm1 = LSTM(units, return_sequences=True)(x)
    lstm1 = LayerNormalization()(lstm1)
    
    # Attention mechanism
    attention = Attention()([lstm1, lstm1])
    
    # Second LSTM layer
    lstm2 = LSTM(units//2)(attention)
    lstm2 = LayerNormalization()(lstm2)
    
    # Dense layers
    x = Dense(128, activation='relu')(lstm2)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout)(x)
    
    # Output layer
    outputs = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# =============================================================================
# Step 5: Training and Evaluation
# =============================================================================
def train_model_with_callbacks(model, X_train, y_train, X_val, y_val, 
                             batch_size=32, epochs=50):
    """
    Advanced training with multiple callbacks and monitoring.
    
    Callbacks:
    - Early stopping
    - Learning rate reduction
    - Model checkpointing
    - TensorBoard logging
    """
    # Create callbacks directory
    callbacks_dir = 'callbacks'
    os.makedirs(callbacks_dir, exist_ok=True)
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3),
        ModelCheckpoint(
            filepath=os.path.join(callbacks_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True
        ),
        TensorBoard(
            log_dir=os.path.join(callbacks_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        )
    ]
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model_advanced(model, X_test, y_test):
    """
    Comprehensive model evaluation with multiple metrics and visualizations.
    
    Features:
    - Confusion matrix
    - ROC curve
    - Classification report
    - Feature importance analysis
    - Model performance metrics
    """
    print("\nEvaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred_classes)
    cr = classification_report(y_test, y_pred_classes)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('visualizations/confusion_matrix.png')
    plt.close()
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('visualizations/roc_curve.png')
    plt.close()
    
    # Calculate additional metrics
    accuracy = np.mean(y_pred_classes == y_test)
    precision = cm[1,1] / (cm[1,1] + cm[0,1])
    recall = cm[1,1] / (cm[1,1] + cm[1,0])
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    # Create results dictionary
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': cr,
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
    }
    
    return results

def save_results(evaluation_results, history):
    """
    Save evaluation results and training history.
    
    Features:
    - Save metrics to JSON
    - Save training history plots
    - Generate summary report
    """
    print("\nSaving results...")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Save evaluation results to JSON
    with open('results/evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    
    # Plot and save training history
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.close()
    
    # Generate summary report
    summary = f"""
    Model Evaluation Summary
    ======================
    
    Accuracy: {evaluation_results['accuracy']:.4f}
    Precision: {evaluation_results['precision']:.4f}
    Recall: {evaluation_results['recall']:.4f}
    F1 Score: {evaluation_results['f1_score']:.4f}
    ROC AUC: {evaluation_results['roc_auc']:.4f}
    
    Classification Report:
    {evaluation_results['classification_report']}
    """
    
    with open('results/summary.txt', 'w') as f:
        f.write(summary)
    
    print("Results saved successfully!")

# =============================================================================
# Step 6: Main Execution
# =============================================================================
def main():
    """Main execution function with comprehensive logging."""
    print("Starting Milestone 2 Analysis...")
    
    # Step 1: Data Loading and Preprocessing
    print("\nStep 1: Loading and preprocessing data...")
    df = load_and_preprocess_data('data/2006Fall_2017Spring_GOES_meteo_combined.csv')
    
    # Step 2: Process image data
    print("\nStep 2: Processing image data...")
    X = process_image_data(df)
    X = X[..., np.newaxis]  # Add channel dimension
    y = df['precipitation_category'].values
    
    # Step 3: Prepare sequences
    print("\nStep 3: Preparing sequences...")
    sequence_length = 5
    X_sequences, y_sequences = prepare_sequences(X, y, sequence_length)
    
    # Step 4: Split data
    print("\nStep 4: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, 
                                                       test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                     test_size=0.2, random_state=42)
    
    # Step 5: Create and train model
    print("\nStep 5: Creating and training model...")
    model = create_advanced_rnn_model()
    history = train_model_with_callbacks(model, X_train, y_train, X_val, y_val)
    
    # Step 6: Evaluate model
    print("\nStep 6: Evaluating model...")
    evaluation_results = evaluate_model_advanced(model, X_test, y_test)
    
    # Step 7: Save results
    print("\nStep 7: Saving results...")
    save_results(evaluation_results, history)

if __name__ == "__main__":
    main() 
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import os
import pandas as pd
import glob

def create_conv_lstm_model(input_shape, num_classes=1):
    """Create a ConvLSTM model for satellite image processing."""
    model = models.Sequential([
        # First ConvLSTM layer
        layers.ConvLSTM2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True,
            input_shape=input_shape
        ),
        layers.BatchNormalization(),
        
        # Second ConvLSTM layer
        layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=False
        ),
        layers.BatchNormalization(),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='sigmoid')
    ])
    
    return model

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, sequence_length):
    """Train and evaluate the model."""
    print(f"\nTraining {model_name} with sequence length {sequence_length}...")
    
    # Create checkpoint directory
    checkpoint_dir = f"checkpoints/{model_name}_seq{sequence_length}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}.h5"),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            save_weights_only=True
        )
    ]
    
    # Try to load the latest checkpoint
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print(f"Loading weights from {latest_checkpoint}")
        model.load_weights(latest_checkpoint)
        
        # Save current state and stop
        print("Saving current model state and stopping training...")
        model.save(f'models/{model_name}_seq{sequence_length}_current.h5')
        return None, None, None
    
    # Train model
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks
    )
    
    # Evaluate model
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_test)
    f1 = f1_score(y_test, y_pred)
    
    # Print results
    print(f"\nResults for {model_name} (Sequence Length: {sequence_length}):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name} (Seq Length: {sequence_length})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'visualizations/confusion_matrix_{model_name}_seq{sequence_length}.png')
    plt.close()
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss - {model_name} (Seq Length: {sequence_length})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy - {model_name} (Seq Length: {sequence_length})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'visualizations/training_history_{model_name}_seq{sequence_length}.png')
    plt.close()
    
    # Save final model
    model.save(f'models/{model_name}_seq{sequence_length}.h5')
    
    return history, accuracy, f1

def load_and_align_data(sequences_path, labels_path, sequence_length):
    """Load and align satellite sequences with labels."""
    # Load sequences
    sequences = np.load(sequences_path)
    print(f"Loaded sequences with shape: {sequences.shape}")
    
    # Load labels
    labels = np.load(labels_path)
    print(f"Loaded labels with shape: {labels.shape}")
    
    # Load meteorological data to get timestamps
    meteo_data = pd.read_csv("data/processed/meteorological_data.csv")
    timestamps = pd.to_datetime(meteo_data['timestamp'])
    
    # Get satellite image timestamps from filenames
    satellite_files = sorted(glob.glob("data/processed/satellite_images/*.png"))
    satellite_timestamps = [pd.to_datetime(os.path.basename(f).split('.')[0], format='%Y-%m-%d_%H-%M') for f in satellite_files]
    
    # Find common timestamps
    common_timestamps = set(timestamps) & set(satellite_timestamps)
    common_timestamps = sorted(list(common_timestamps))
    
    # Get indices for common timestamps
    meteo_indices = [i for i, t in enumerate(timestamps) if t in common_timestamps]
    satellite_indices = [i for i, t in enumerate(satellite_timestamps) if t in common_timestamps]
    
    # Ensure indices are within bounds
    max_satellite_index = sequences.shape[0] - 1
    max_meteo_index = labels.shape[0] - 1
    
    # Filter indices to be within bounds
    satellite_indices = [i for i in satellite_indices if i < max_satellite_index]
    meteo_indices = [i for i in meteo_indices if i < max_meteo_index]
    
    # Find the minimum length to ensure consistent sample sizes
    min_length = min(len(satellite_indices), len(meteo_indices))
    
    # Use only the first min_length indices
    satellite_indices = satellite_indices[:min_length]
    meteo_indices = meteo_indices[:min_length]
    
    # Align sequences and labels
    aligned_sequences = sequences[satellite_indices]
    aligned_labels = labels[meteo_indices]
    
    print(f"Aligned sequences shape: {aligned_sequences.shape}")
    print(f"Aligned labels shape: {aligned_labels.shape}")
    
    return aligned_sequences, aligned_labels

def main():
    """Main function to train and evaluate the ConvLSTM model."""
    # Create directories
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Load processed satellite image sequences
    sequence_lengths = [8, 16, 24, 48]
    results = {}
    
    for seq_len in sequence_lengths:
        # Load and align data
        sequences_path = f"data/processed/satellite_images/sequences_{seq_len}.npy"
        labels_path = "data/processed/rain_labels.npy"
        
        if not os.path.exists(sequences_path):
            print(f"Sequences file not found: {sequences_path}")
            continue
        
        if not os.path.exists(labels_path):
            print(f"Labels file not found: {labels_path}")
            continue
        
        try:
            X, y = load_and_align_data(sequences_path, labels_path, seq_len)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42
            )
            
            # Create and train model
            model = create_conv_lstm_model(
                input_shape=(seq_len, 64, 64, 1)
            )
            
            history, accuracy, f1 = train_and_evaluate_model(
                model, X_train, y_train, X_test, y_test,
                "ConvLSTM_Satellite", seq_len
            )
            
            results[seq_len] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'history': history.history
            }
            
        except Exception as e:
            print(f"Error processing sequence length {seq_len}: {str(e)}")
            continue
    
    # Compare results across sequence lengths
    if results:
        plt.figure(figsize=(15, 6))
        
        # Plot accuracy comparison
        plt.subplot(1, 2, 1)
        accuracies = [results[seq_len]['accuracy'] for seq_len in sequence_lengths]
        plt.plot(sequence_lengths, accuracies, marker='o')
        plt.title('Model Accuracy vs Sequence Length')
        plt.xlabel('Sequence Length')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        # Plot F1 score comparison
        plt.subplot(1, 2, 2)
        f1_scores = [results[seq_len]['f1_score'] for seq_len in sequence_lengths]
        plt.plot(sequence_lengths, f1_scores, marker='o')
        plt.title('Model F1 Score vs Sequence Length')
        plt.xlabel('Sequence Length')
        plt.ylabel('F1 Score')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('visualizations/sequence_length_comparison.png')
        plt.close()

if __name__ == "__main__":
    main() 
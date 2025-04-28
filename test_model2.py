import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import os
import pandas as pd
import glob
from model_architectures import create_model2

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Enable data parallelism if multiple GPUs are available
strategy = tf.distribute.MirroredStrategy()

def load_and_preprocess_data(sequences_path, labels_path, sequence_length):
    """Load and preprocess satellite sequences and labels."""
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

def create_dataset(X, y, batch_size):
    """Create a tf.data.Dataset for efficient data loading."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, sequence_length):
    """Train and evaluate the model."""
    print(f"\nTraining {model_name} with sequence length {sequence_length}...")
    
    # Create checkpoint directory
    checkpoint_dir = f"checkpoints/{model_name}_seq{sequence_length}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Compile model with faster learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Save current state and stop
    print("Saving current model state and stopping training...")
    
    # Save model weights
    weights_path = f"checkpoints/{model_name}_seq{sequence_length}_weights.h5"
    model.save_weights(weights_path)
    print(f"Saved model weights to {weights_path}")
    
    # Save full model
    model_path = f"models/{model_name}_seq{sequence_length}.h5"
    model.save(model_path)
    print(f"Saved full model to {model_path}")
    
    return None, None, None

def main():
    """Main function to test Model 2."""
    # Create directories
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Test with different sequence lengths
    sequence_lengths = [8, 16]
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
            X, y = load_and_preprocess_data(sequences_path, labels_path, seq_len)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42
            )
            
            # Create and train model
            model = create_model2(
                input_shape=(seq_len, 64, 64, 1)
            )
            
            history, accuracy, f1 = train_and_evaluate_model(
                model, X_train, y_train, X_test, y_test,
                "Model2", seq_len
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
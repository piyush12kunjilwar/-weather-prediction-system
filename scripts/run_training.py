"""
Script to run the complete model training and evaluation pipeline
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step2_data_loading import load_and_preprocess_data
from data_preparation import prepare_data_for_training
from cnn_rnn_model import create_cnn_rnn_model
from step5_model_training import train_model, evaluate_model, evaluate_and_visualize_results

def main():
    """Run the complete training and evaluation pipeline"""
    try:
        print("Starting training pipeline...")
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Load and preprocess data
        print("\nLoading and preprocessing data...")
        df = load_and_preprocess_data()
        
        # Prepare data for training
        print("\nPreparing data for training...")
        train_data, val_data, test_data, class_weights = prepare_data_for_training(
            df=df,
            sequence_length=24,
            batch_size=32
        )
        
        # Create model
        print("\nCreating model...")
        model = create_cnn_rnn_model()
        
        # Compile model
        print("\nCompiling model...")
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        print("\nTraining model...")
        history, model = train_model(
            model=model,
            X_train=train_data,
            y_train=None,  # Labels are included in the data generator
            X_val=val_data,
            y_val=None,    # Labels are included in the data generator
            class_weights=class_weights,
            batch_size=32,
            epochs=50
        )
        
        # Evaluate model
        print("\nEvaluating model...")
        metrics = evaluate_model(model, test_data, None)  # Labels are included in the data generator
        
        # Create visualizations
        print("\nCreating visualizations...")
        evaluate_and_visualize_results(model, history, test_data, None)
        
        print("\nTraining pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 
"""
Main script to run all steps of the Rain Prediction Model
"""

# Import necessary modules
from step1_imports import *
from step2_data_loading import load_and_preprocess_data
from step3_image_processing import decode_image_data
from step4_model_architecture import create_advanced_rnn_model, create_callbacks
from step5_model_training import train_model, evaluate_model, plot_training_history
from step6_results import generate_results_report, save_model

def main():
    print("Starting Rain Prediction Model Pipeline...")
    
    # Step 2: Load and preprocess data
    print("\nStep 2: Loading and preprocessing data...")
    df = load_and_preprocess_data()
    print("Data shape:", df.shape)
    
    # Step 3: Process image data
    print("\nStep 3: Processing image data...")
    processed_images = decode_image_data(df)
    
    # Prepare features and target
    X = processed_images  # Assuming this is your feature set
    y = df['precipitation_category'].values
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Step 4: Create model
    print("\nStep 4: Creating model architecture...")
    input_shape = X_train.shape[1:]  # Get input shape from training data
    model = create_advanced_rnn_model(input_shape)
    print(model.summary())
    
    # Step 5: Train and evaluate model
    print("\nStep 5: Training model...")
    history, trained_model = train_model(model, X_train, y_train, X_val, y_val)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluation_results = evaluate_model(trained_model, X_test, y_test)
    
    # Step 6: Generate results and save model
    print("\nStep 6: Generating results and saving model...")
    generate_results_report(evaluation_results, model.summary())
    save_model(trained_model)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main() 
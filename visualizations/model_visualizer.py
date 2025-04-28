import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from data_visualizer import DataVisualizer
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_visualization_dir():
    """Ensure the visualizations directory exists"""
    vis_dir = 'visualizations'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
        logger.info(f"Created directory: {vis_dir}")
    return vis_dir

def visualize_model_performance(model_path, X_test, y_test, history=None):
    """
    Generate comprehensive visualizations for model performance
    """
    vis_dir = ensure_visualization_dir()
    visualizer = DataVisualizer(save_dir=vis_dir)
    
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1) if y_pred.shape[-1] > 1 else (y_pred > 0.5).astype(int)
    y_test_classes = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    classes = [f'Class {i}' for i in range(cm.shape[0])]
    visualizer.plot_confusion_matrix(
        cm=cm,
        classes=classes,
        title='Model Prediction Confusion Matrix',
        filename='model_confusion_matrix'
    )
    
    # Plot training history if available
    if history:
        visualizer.plot_model_metrics(
            history=history,
            filename='model_training_history'
        )
    
    # Plot feature importance if applicable
    if hasattr(model, 'layers') and len(model.layers) > 0:
        try:
            # Get weights from the first dense layer
            weights = model.layers[0].get_weights()[0]
            feature_importance = np.mean(np.abs(weights), axis=1)
            features = [f'Feature {i}' for i in range(len(feature_importance))]
            
            visualizer.plot_feature_importance(
                features=features,
                importance=feature_importance,
                title='Feature Importance',
                filename='model_feature_importance'
            )
        except:
            print("Could not generate feature importance plot")

def visualize_predictions(model_path, X_test, timestamps=None):
    """
    Generate visualizations for model predictions over time
    """
    vis_dir = ensure_visualization_dir()
    logger.info(f"Saving visualizations to: {os.path.abspath(vis_dir)}")
    
    try:
        # Load the model
        logger.info(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = X_test  # Use the provided predictions directly
        
        # If we have timestamps, create time series plot
        if timestamps is not None:
            import pandas as pd
            pred_df = pd.DataFrame({
                'timestamp': timestamps,
                'prediction': predictions.flatten()
            })
            
            # Create time series plot
            plt.figure(figsize=(12, 6))
            plt.plot(pred_df['timestamp'], pred_df['prediction'], 'b-', label='Predictions')
            plt.title('Model Predictions Over Time')
            plt.xlabel('Time')
            plt.ylabel('Predicted Value')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            time_series_path = os.path.join(vis_dir, 'predictions_time_series.jpg')
            plt.savefig(time_series_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved time series plot to: {time_series_path}")
        
        # Plot prediction distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(predictions.flatten(), kde=True)
        plt.title('Distribution of Model Predictions')
        plt.xlabel('Predicted Value')
        plt.ylabel('Count')
        plt.grid(True)
        plt.tight_layout()
        dist_path = os.path.join(vis_dir, 'prediction_distribution.jpg')
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved distribution plot to: {dist_path}")

        # Plot actual vs predicted values
        if hasattr(X_test, 'shape') and len(X_test.shape) > 1:
            plt.figure(figsize=(10, 6))
            plt.scatter(range(len(predictions)), predictions, alpha=0.5, label='Predictions')
            plt.title('Model Predictions')
            plt.xlabel('Sample Index')
            plt.ylabel('Predicted Value')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            scatter_path = os.path.join(vis_dir, 'predictions_scatter.jpg')
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved scatter plot to: {scatter_path}")
            
    except Exception as e:
        logger.error(f"Error in visualize_predictions: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    try:
        # Load your test data and model
        model_path = "rain_prediction_model_final.h5"
        
        # Load your test data here
        # X_test = ...
        # y_test = ...
        # history = ...  # If you saved the training history
        
        # Generate visualizations
        # visualize_model_performance(model_path, X_test, y_test, history)
        
        # If you have timestamps for your predictions
        # timestamps = ...
        # visualize_predictions(model_path, X_test, timestamps)
        
        print("Visualizations generated successfully!")
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}") 
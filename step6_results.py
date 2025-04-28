"""
Step 6: Results and Reporting
"""

import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def generate_results_report(evaluation_results, model_summary, save_path='results'):
    """
    Generate comprehensive results report.
    
    Args:
        evaluation_results: Dictionary containing evaluation metrics
        model_summary: Model summary string
        save_path: Directory to save results
    """
    print("Generating results report...")
    
    # Create results directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model summary
    with open(f'{save_path}/model_summary_{timestamp}.txt', 'w') as f:
        f.write(model_summary)
    
    # Save evaluation metrics
    with open(f'{save_path}/evaluation_metrics_{timestamp}.txt', 'w') as f:
        f.write(f"Accuracy: {evaluation_results['accuracy']:.4f}\n")
        f.write(f"AUC: {evaluation_results['auc']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(evaluation_results['classification_report'])
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(evaluation_results['confusion_matrix'], 
                annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(f'{save_path}/confusion_matrix_{timestamp}.png')
    plt.close()
    
    # Plot and save ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr = evaluation_results['roc_curve']
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {evaluation_results["auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(f'{save_path}/roc_curve_{timestamp}.png')
    plt.close()
    
    print(f"Results saved to {save_path}")

def save_model(model, save_path='models'):
    """
    Save the trained model and its weights.
    
    Args:
        model: Trained Keras model
        save_path: Directory to save the model
    """
    print("Saving model...")
    
    # Create models directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model architecture and weights
    model.save(f'{save_path}/rain_prediction_model_{timestamp}.h5')
    
    print(f"Model saved to {save_path}") 
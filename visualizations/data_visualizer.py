import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

class DataVisualizer:
    def __init__(self, save_dir='visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Set style for matplotlib
        plt.style.use('seaborn-v0_8')
        sns.set_theme()
        sns.set_palette("husl")

    def plot_time_series(self, data, date_column, value_column, title, filename):
        """Create a time series plot using plotly"""
        fig = px.line(data, x=date_column, y=value_column, 
                     title=title)
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=value_column,
            template="plotly_white"
        )
        fig.write_html(self.save_dir / f"{filename}.html")
        fig.write_image(self.save_dir / f"{filename}.png")

    def plot_correlation_heatmap(self, data, title, filename):
        """Create a correlation heatmap using seaborn"""
        plt.figure(figsize=(12, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{filename}.png")
        plt.close()

    def plot_distribution(self, data, column, title, filename):
        """Create a distribution plot using seaborn"""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x=column, kde=True)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{filename}.png")
        plt.close()

    def plot_model_metrics(self, history, filename):
        """Plot model training history metrics"""
        metrics = history.history
        epochs = range(1, len(metrics['loss']) + 1)

        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, metrics['loss'], 'b-', label='Training Loss')
        if 'val_loss' in metrics:
            plt.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        if 'accuracy' in metrics:
            plt.plot(epochs, metrics['accuracy'], 'b-', label='Training Accuracy')
            if 'val_accuracy' in metrics:
                plt.plot(epochs, metrics['val_accuracy'], 'r-', label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

        plt.tight_layout()
        plt.savefig(self.save_dir / f"{filename}.png")
        plt.close()

    def plot_confusion_matrix(self, cm, classes, title, filename):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{filename}.png")
        plt.close()

    def plot_feature_importance(self, features, importance, title, filename):
        """Plot feature importance"""
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame({'feature': features, 'importance': importance})
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title(title)
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{filename}.png")
        plt.close()

    def plot_scatter_matrix(self, data, columns, title, filename):
        """Create a scatter matrix plot using plotly"""
        fig = px.scatter_matrix(data, dimensions=columns, title=title)
        fig.update_layout(title_x=0.5)
        fig.write_html(self.save_dir / f"{filename}.html")
        fig.write_image(self.save_dir / f"{filename}.png")

# Example usage
if __name__ == "__main__":
    # Create visualizer instance
    visualizer = DataVisualizer()

    # Load your data
    try:
        data = pd.read_csv("2006Fall_2017Spring_GOES_meteo_combined.csv")
        
        # Time series visualization
        visualizer.plot_time_series(
            data=data.head(1000),  # Using subset for example
            date_column='timestamp',
            value_column='temperature',
            title='Temperature Over Time',
            filename='temperature_time_series'
        )

        # Correlation heatmap
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        visualizer.plot_correlation_heatmap(
            data[numeric_columns],
            title='Feature Correlations',
            filename='correlation_heatmap'
        )

        # Distribution plots
        visualizer.plot_distribution(
            data=data,
            column='temperature',
            title='Temperature Distribution',
            filename='temperature_distribution'
        )

        # Scatter matrix
        selected_columns = ['temperature', 'humidity', 'pressure']
        visualizer.plot_scatter_matrix(
            data=data,
            columns=selected_columns,
            title='Weather Parameters Relationships',
            filename='weather_scatter_matrix'
        )

    except Exception as e:
        print(f"Error generating visualizations: {str(e)}") 
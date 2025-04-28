import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import ast
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist."""
    Path("debug").mkdir(exist_ok=True)
    Path("visualizations").mkdir(exist_ok=True)
    Path("visualizations/arrays").mkdir(exist_ok=True)

def parse_array_string(array_str):
    """Convert string representation of numpy array to actual array."""
    try:
        # Remove 'array(' and ')' and any extra whitespace
        array_str = array_str.replace('array(', '').replace(')', '').strip()
        # Convert string to list
        array_list = ast.literal_eval(array_str)
        # Convert to numpy array
        return np.array(array_list)
    except Exception as e:
        logger.error(f"Error parsing array string: {e}")
        return None

def analyze_array(array, name=""):
    """Analyze a single array and return statistics."""
    try:
        if array is None:
            return None
            
        stats = {
            "name": name,
            "length": len(array),
            "mean": float(np.mean(array)) if not np.all(np.isnan(array)) else None,
            "std": float(np.std(array)) if not np.all(np.isnan(array)) else None,
            "min": float(np.min(array)) if not np.all(np.isnan(array)) else None,
            "max": float(np.max(array)) if not np.all(np.isnan(array)) else None,
            "nan_count": int(np.isnan(array).sum()),
            "all_nan": bool(np.all(np.isnan(array)))
        }
        return stats
    except Exception as e:
        logger.error(f"Error analyzing array: {e}")
        return None

def visualize_array(array, name, index):
    """Visualize array as 60x60 heatmap."""
    try:
        if array is None or len(array) != 3599:
            return
            
        # Pad array to 3600 elements
        padded_array = np.pad(array, (0, 1), mode='constant', constant_values=np.nan)
        
        # Reshape to 60x60
        reshaped = padded_array.reshape(60, 60)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(reshaped, cmap='viridis')
        plt.colorbar(label='Value')
        plt.title(f'{name} - Array {index}')
        
        # Save plot
        filename = f"visualizations/arrays/{name.replace(' ', '_')}_{index}.png"
        plt.savefig(filename)
        plt.close()
        
    except Exception as e:
        logger.error(f"Error visualizing array: {e}")

def process_satellite_data():
    """Process satellite data arrays."""
    try:
        # Load data
        logger.info("Loading data...")
        df = pd.read_csv('data/2006Fall_2017Spring_GOES_meteo_combined.csv')
        
        # Initialize statistics storage
        all_stats = []
        
        # Process each row
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                logger.info(f"Processing row {idx}")
                
            # Process lake arrays
            for col in ['Lake_Erie', 'Lake_Huron', 'Lake_Michigan', 'Lake_Ontario', 'Lake_Superior']:
                if col in row:
                    array = parse_array_string(str(row[col]))
                    if array is not None:
                        stats = analyze_array(array, name=col)
                        if stats:
                            all_stats.append(stats)
                            if idx % 1000 == 0:  # Visualize every 1000th array
                                visualize_array(array, col, idx)
        
        # Save statistics
        with open('debug/array_statistics.json', 'w') as f:
            json.dump(all_stats, f, indent=2)
            
        logger.info("Processing complete. Check debug/ and visualizations/ directories for results.")
        
    except Exception as e:
        logger.error(f"Error processing satellite data: {e}")

if __name__ == "__main__":
    setup_directories()
    process_satellite_data() 
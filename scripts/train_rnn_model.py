import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight

def create_directories():
    """Create necessary directories for models and visualizations"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)

def load_and_preprocess_data(file_path):
    """Load and preprocess the data"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Convert datetime columns
    df['datetime'] = pd.to_datetime(df['Date_UTC'] + ' ' + df['Time_UTC'])
    
    # Convert numeric columns with correct column names
    numeric_cols = [
        'Temp (F)', 'RH (%)', 'Dewpt (F)', 'Wind Spd (mph)', 
        'Wind Direction (deg)', 'Peak Wind Gust(mph)', 'Low Cloud Ht (ft)',
        'Med Cloud Ht (ft)', 'High Cloud Ht (ft)', 'Visibility (mi)',
        'Atm Press (hPa)', 'Sea Lev Press (hPa)', 'Altimeter (hPa)',
        'Precip (in)', 'Wind Chill (F)', 'Heat Index (F)'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Process Lake_data_1D column
    def parse_array(x):
        try:
            if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
                arr = np.array(ast.literal_eval(x), dtype=np.float32)
                if len(arr) == 3599:
                    # Pad array to 3600 elements
                    arr = np.pad(arr, (0, 1), mode='constant', constant_values=0)
                    # Normalize array
                    arr = (arr - np.nanmean(arr)) / (np.nanstd(arr) + 1e-8)
                    # Replace any remaining NaN values with 0
                    arr = np.nan_to_num(arr, nan=0.0)
                    return arr
                return None
            return None
        except:
            return None
    
    print("Processing lake data arrays...")
    df['lake_array'] = df['Lake_data_1D'].apply(parse_array)
    
    # Filter out rows with invalid arrays
    valid_rows = df['lake_array'].notna()
    df = df[valid_rows].copy()
    
    print(f"Found {len(df)} valid arrays")
    
    return df

def create_model():
    """Create the RNN model architecture"""
    model = Sequential([
        # Input layer
        Input(shape=(3600,)),
        
        # Reshape layer to convert 1D to sequence
        Reshape((60, 60)),
        
        # Add batch normalization after reshape
        BatchNormalization(),
        
        # LSTM layers with gradient clipping
        LSTM(128, return_sequences=True, 
             kernel_initializer='glorot_uniform',
             recurrent_initializer='orthogonal',
             bias_initializer='zeros'),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(64,
             kernel_initializer='glorot_uniform',
             recurrent_initializer='orthogonal',
             bias_initializer='zeros'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers
        Dense(32, activation='relu',
              kernel_initializer='glorot_uniform'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    return model

def train_model():
    """Train the RNN model"""
    # Load and preprocess data
    df = load_and_preprocess_data('processed_data/processed_with_categories.csv')
    
    # Prepare features and target
    X = np.stack(df['lake_array'].values)
    y = df['precipitation_category'].values
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for class_label, count in zip(unique, counts):
        print(f"Class {class_label}: {count} samples ({count/len(y)*100:.2f}%)")
    
    # Calculate class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=unique,
        y=y
    )
    class_weight_dict = dict(zip(unique, class_weights))
    print("\nClass weights:", class_weight_dict)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train model
    model = create_model()
    print("\nModel summary:")
    model.summary()
    
    early_stopping = EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Save model
    model.save('models/rnn_model.keras')
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_results = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test loss: {test_results[0]:.4f}")
    print(f"Test accuracy: {test_results[1]:.4f}")
    print(f"Test AUC: {test_results[2]:.4f}")
    
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('visualizations/confusion_matrix.png')
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['AUC'], label='Training AUC')
    plt.plot(history.history['val_AUC'], label='Validation AUC')
    plt.title('Model AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/training_history.png')
    plt.close()

if __name__ == "__main__":
    create_directories()
    train_model() 
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import cv2
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Create directories
os.makedirs("visualizations", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load meteorological data
print("Loading and preprocessing meteorological data...")
df = pd.read_csv("data/2006Fall_2017Spring_GOES_meteo_combined.csv")
print(f"Dataset shape: {df.shape}")

# Convert datetime columns
df["datetime_utc"] = pd.to_datetime(df["Date_UTC"] + " " + df["Time_UTC"])
df["datetime_cst"] = pd.to_datetime(df["Date_CST"] + " " + df["Time_CST"])
df.set_index("datetime_utc", inplace=True)

# Convert numeric columns
numeric_columns = ["Temp (F)", "RH (%)", "Wind Direction (deg)",
                  "Low Cloud Ht (ft)", "Med Cloud Ht (ft)",
                  "High Cloud Ht (ft)", "Visibility (mi)", "Atm Press (hPa)",
                  "Sea Lev Press (hPa)", "Altimeter (hPa)", "Precip (in)",
                  "Wind Chill (F)", "Heat Index (F)"]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Handle missing values
df = df.ffill()  # Forward fill
df = df.bfill()  # Backward fill

# Create temporal features
df["month"] = df.index.month
df["day_of_week"] = df.index.dayofweek
df["hour"] = df.index.hour
df["season"] = (df["month"] % 12 + 3) // 3

# Create binary target
df["precipitation_category"] = (df["Precip (in)"] > 0).astype(int)

print("\nPrecipitation distribution:")
print(df["precipitation_category"].value_counts())

# Function to create sequences
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

# Prepare meteorological data sequences
meteo_features = ["Temp (F)", "RH (%)", "Wind Direction (deg)",
                 "Low Cloud Ht (ft)", "Med Cloud Ht (ft)",
                 "High Cloud Ht (ft)", "Visibility (mi)", "Atm Press (hPa)",
                 "Sea Lev Press (hPa)", "Altimeter (hPa)"]

meteo_data = df[meteo_features].values

# Create sequences with different window sizes
sequence_lengths = [8, 16, 24, 48]  # Different window sizes to experiment with
meteo_sequences = {}
for seq_len in sequence_lengths:
    meteo_sequences[seq_len] = create_sequences(meteo_data, seq_len)

# Prepare targets
targets = {}
for seq_len in sequence_lengths:
    targets[seq_len] = df["precipitation_category"].values[seq_len-1:]

# Model architectures
def create_model1(input_shape_meteo):
    # Meteorological branch
    meteo_input = layers.Input(shape=input_shape_meteo)
    x = layers.LSTM(64, return_sequences=True)(meteo_input)
    x = layers.LSTM(32)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=meteo_input, outputs=output)
    return model

def create_model2(input_shape_meteo):
    # Meteorological branch with deeper architecture
    meteo_input = layers.Input(shape=input_shape_meteo)
    x = layers.LSTM(64, return_sequences=True)(meteo_input)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=meteo_input, outputs=output)
    return model

# Function to train and evaluate model
def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, sequence_length):
    print(f"\nTraining {model_name} with sequence length {sequence_length}...")
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]
    
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
    
    # Save model
    model.save(f'models/{model_name}_seq{sequence_length}.h5')
    
    return history, accuracy, f1

# Train all models
print("\nStarting model training...")
models_to_train = [
    (create_model1, "Model1_LSTM_Shallow"),
    (create_model2, "Model2_LSTM_Deep")
]

results = {}
for model_func, model_name in models_to_train:
    results[model_name] = {}
    for seq_len in sequence_lengths:
        print(f"\nTraining {model_name} with sequence length {seq_len}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            meteo_sequences[seq_len], targets[seq_len],
            test_size=0.2, random_state=42
        )
        
        # Create and train model
        model = model_func(
            input_shape_meteo=(seq_len, len(meteo_features))
        )
        
        history, accuracy, f1 = train_and_evaluate_model(
            model, model_name,
            X_train, y_train,
            X_test, y_test,
            seq_len
        )
        
        results[model_name][seq_len] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'history': history.history
        }

# Compare models
print("\nGenerating comparison plots...")
plt.figure(figsize=(15, 6))

# Plot accuracy comparison
plt.subplot(1, 2, 1)
for model_name in results.keys():
    accuracies = [results[model_name][seq_len]['accuracy'] for seq_len in sequence_lengths]
    plt.plot(sequence_lengths, accuracies, marker='o', label=model_name)

plt.title('Model Accuracy Comparison')
plt.xlabel('Sequence Length')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot F1 score comparison
plt.subplot(1, 2, 2)
for model_name in results.keys():
    f1_scores = [results[model_name][seq_len]['f1_score'] for seq_len in sequence_lengths]
    plt.plot(sequence_lengths, f1_scores, marker='o', label=model_name)

plt.title('Model F1 Score Comparison')
plt.xlabel('Sequence Length')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('visualizations/model_comparison.png')
plt.close()

# Save results to CSV
results_df = pd.DataFrame(columns=['Model', 'Sequence Length', 'Accuracy', 'F1 Score'])
for model_name in results.keys():
    for seq_len in sequence_lengths:
        results_df = pd.concat([results_df, pd.DataFrame({
            'Model': [model_name],
            'Sequence Length': [seq_len],
            'Accuracy': [results[model_name][seq_len]['accuracy']],
            'F1 Score': [results[model_name][seq_len]['f1_score']]
        })], ignore_index=True)

results_df.to_csv('results/model_comparison_results.csv', index=False)
print("\nResults saved to results/model_comparison_results.csv") 
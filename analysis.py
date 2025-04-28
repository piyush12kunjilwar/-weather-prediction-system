import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
print("Loading data...")
df = pd.read_csv("data/2006Fall_2017Spring_GOES_meteo_combined.csv")

# Convert numeric columns
print("Processing data...")
numeric_cols = ["Temp (F)", "RH (%)", "Wind Spd (mph)", "Precip (in)"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Create timestamp
df["timestamp"] = pd.to_datetime(df["Date_UTC"] + " " + df["Time_UTC"])

# Display basic info
print("\nDataset Info:")
print("Number of records:", len(df))
print("Time range:", df["timestamp"].min(), "to", df["timestamp"].max())

# Create figure
plt.figure(figsize=(15, 10))

# Plot each variable
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 2, i)
    plt.hist(df[col].dropna(), bins=50, alpha=0.7)
    plt.title("Distribution of " + col)
    plt.xlabel(col)
    plt.grid(True)

plt.tight_layout()
plt.show()

# Display summary statistics
print("\nSummary Statistics:")
print(df[numeric_cols].describe()) 
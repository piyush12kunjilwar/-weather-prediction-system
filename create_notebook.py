#!/usr/bin/env python3
import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Create markdown cell
markdown = nbf.v4.new_markdown_cell('''# Meteorological Data Analysis

This notebook analyzes the meteorological dataset with time series analysis, statistical summaries, seasonal patterns, and correlations.''')

# Create code cells
imports = nbf.v4.new_code_cell('''# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Create visualizations directory and subdirectories
os.makedirs('visualizations/time_series', exist_ok=True)
os.makedirs('visualizations/patterns', exist_ok=True)
os.makedirs('visualizations/correlation', exist_ok=True)

# Set plot style
plt.style.use('default')
print('Libraries imported and directories created successfully')''')

load_data = nbf.v4.new_code_cell('''# Load and preprocess the data
print('Loading data...')
data_path = 'data/2006Fall_2017Spring_GOES_meteo_combined.csv'
df = pd.read_csv(data_path)

# Convert datetime
print('Converting datetime...')
df['datetime'] = pd.to_datetime(df['Date_UTC'] + ' ' + df['Time_UTC'])
df.set_index('datetime', inplace=True)

# Convert numeric columns
print('\\nConverting numeric columns...')
numeric_columns = ['Temp (F)', 'Wind Spd (mph)', 'RH (%)', 'Precip (in)']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col].replace('m', np.nan), errors='coerce')

# Convert units
print('\\nConverting units...')
df['Temperature (C)'] = (df['Temp (F)'] - 32) * 5/9
df['Wind Speed (m/s)'] = df['Wind Spd (mph)'] * 0.44704
df['Relative Humidity (%)'] = df['RH (%)']
df['Precip (in)'] = df['Precip (in)'].fillna(0)

print('\\nProcessed data head:')
df[['Temperature (C)', 'Wind Speed (m/s)', 'Relative Humidity (%)', 'Precip (in)']].head()''')

time_series = nbf.v4.new_code_cell('''# Time Series Analysis
variables = ['Temperature (C)', 'Wind Speed (m/s)', 'Relative Humidity (%)', 'Precip (in)']

for var in variables:
    plt.figure(figsize=(15, 5))
    plt.plot(df.index, df[var])
    plt.title(f'Time Series of {var}')
    plt.xlabel('Date')
    plt.ylabel(var)
    plt.grid(True)
    safe_filename = var.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    plt.savefig(f'visualizations/time_series/{safe_filename}_timeseries.png')
    plt.close()

print('Time series plots saved in visualizations/time_series directory')''')

# Add cells to notebook
nb.cells = [markdown, imports, load_data, time_series]

# Write the notebook to a file
with open('analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f) 
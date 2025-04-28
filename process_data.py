import os
import numpy as np
import pandas as pd
from PIL import Image
import shutil
from sklearn.preprocessing import MinMaxScaler
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import ast

def arrays_2_png(lat, lon, val, output_path):
    """Convert lat/lon/value arrays to a PNG image"""
    if len(lat) == len(lon) == len(val):
        plt.figure(figsize=(10, 10))
        plt.scatter(lon, lat, c=val, cmap=plt.cm.gray, marker='s')
        plt.axis('off')  # Turn off axes
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return True
    return False

def process_meteorological_data(input_path, output_path):
    """Process meteorological data and extract rain labels"""
    print("Processing meteorological data...")
    
    # Read the data
    df = pd.read_csv(input_path)
    
    # Create timestamp from Date_UTC and Time_UTC
    df['timestamp'] = pd.to_datetime(df['Date_UTC'] + ' ' + df['Time_UTC'])
    
    # Filter daytime data (14:00 UTC to 21:00 UTC)
    df['hour'] = df['Time_UTC'].str[:2].astype(int)
    df = df[(df['hour'] >= 14) & (df['hour'] <= 21)]
    
    # Select relevant features
    features = [
        'Temp (F)', 'RH (%)', 'Wind Spd (mph)', 'Wind Direction (deg)',
        'Visibility (mi)', 'Atm Press (hPa)', 'Precip (in)'
    ]
    
    # Convert features to numeric, replacing errors with NaN
    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')
    
    # Create processed dataframe with timestamp and features
    processed_df = df[['timestamp'] + features].copy()
    
    # Fill missing values
    processed_df = processed_df.ffill().bfill()
    
    # Extract rain labels (precipitation > 0)
    rain_labels = (processed_df['Precip (in)'] > 0).astype(int)
    
    # Save processed data
    processed_df.to_csv(output_path, index=False)
    
    # Save rain labels
    np.save(os.path.join(os.path.dirname(output_path), 'rain_labels.npy'), rain_labels.values)
    print(f"Saved {len(rain_labels)} rain labels")
    
    return df, processed_df

def process_satellite_images(df, output_dir, lat_lon_file, target_size=(128, 128)):
    """Process and save satellite images"""
    print("Processing satellite images...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load lat/lon data
    lat_lon_df = pd.read_csv(lat_lon_file)
    lat = lat_lon_df['latitude'].values
    lon = lat_lon_df['longitude'].values
    
    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        try:
            # Extract timestamp
            timestamp = pd.to_datetime(f"{row['Date_UTC']} {row['Time_UTC']}")
            
            # Convert Lake_data_1D from string to array
            try:
                val = ast.literal_eval(row['Lake_data_1D'])
            except:
                print(f"Failed to parse Lake_data_1D for {timestamp}")
                continue
            
            # Convert to numpy array
            val = np.array(val)
            
            # Create image filename
            img_file = timestamp.strftime('%Y-%m-%d_%H-%M.png')
            output_path = os.path.join(output_dir, img_file)
            
            # Convert arrays to PNG
            if arrays_2_png(lat, lon, val, output_path):
                # Read the saved image and resize
                img = cv2.imread(output_path)
                if img is not None:
                    img = cv2.resize(img, target_size)
                    cv2.imwrite(output_path, img)
            
        except Exception as e:
            print(f"Error processing image for {timestamp}: {str(e)}")
    
    print(f"Processed satellite images saved to {output_dir}")

def main():
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    processed_dir = os.path.join(data_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Process meteorological data
    meteo_input = os.path.join(data_dir, '2006Fall_2017Spring_GOES_meteo_combined.csv')
    meteo_output = os.path.join(processed_dir, 'meteorological_data.csv')
    
    if os.path.exists(meteo_input):
        df, processed_df = process_meteorological_data(meteo_input, meteo_output)
        if processed_df is not None:
            print("Successfully processed meteorological data")
            
            # Process satellite images
            lat_lon_file = os.path.join('ml-fp-les-preprocessing', 'lat_long_1D_labels_for_plotting.csv')
            satellite_output = os.path.join(processed_dir, 'satellite_images')
            
            if os.path.exists(lat_lon_file):
                process_satellite_images(df, satellite_output, lat_lon_file)
                print("Successfully processed satellite images")
            else:
                print(f"Lat/lon file not found: {lat_lon_file}")
    else:
        print(f"Meteorological data file not found: {meteo_input}")

if __name__ == '__main__':
    main() 
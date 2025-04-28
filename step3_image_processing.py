"""
Step 3: Image Data Processing
"""

import numpy as np

def decode_image_data(df):
    """
    Decode and process image data from the dataset.
    
    Features:
    - Image normalization
    - Channel-wise processing
    - Image augmentation
    """
    print("Processing image data...")
    
    # Initialize lists to store processed images
    processed_images = []
    
    # Process each image in the dataset
    for idx, row in df.iterrows():
        # Decode image data (assuming it's stored as base64 or similar)
        image_data = row['image_data']  # Adjust column name as needed
        
        # Normalize image
        normalized_image = normalize_image(image_data)
        
        # Add to processed images list
        processed_images.append(normalized_image)
    
    return np.array(processed_images)

def normalize_image(image_data):
    """Normalize image data to [0,1] range."""
    # Convert to float32
    image = image_data.astype(np.float32)
    
    # Normalize to [0,1] range
    image = (image - image.min()) / (image.max() - image.min())
    
    return image

def augment_image(image):
    """Apply basic image augmentation techniques."""
    # Random horizontal flip
    if np.random.random() > 0.5:
        image = np.fliplr(image)
    
    # Random vertical flip
    if np.random.random() > 0.5:
        image = np.flipud(image)
    
    # Random rotation (90 degrees)
    if np.random.random() > 0.5:
        image = np.rot90(image)
    
    return image 
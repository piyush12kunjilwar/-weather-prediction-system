import os
from PIL import Image
import glob

def convert_png_to_jpeg():
    # Get all PNG files in the current directory
    png_files = glob.glob('*.png')
    
    # Create a directory for JPEG files if it doesn't exist
    jpeg_dir = 'jpeg_visualizations'
    if not os.path.exists(jpeg_dir):
        os.makedirs(jpeg_dir)
    
    # Convert each PNG to JPEG
    for png_file in png_files:
        try:
            # Open the PNG image
            img = Image.open(png_file)
            
            # Convert to RGB if necessary (in case of RGBA images)
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                img = img.convert('RGB')
            
            # Create the output filename
            jpeg_filename = os.path.join(jpeg_dir, os.path.splitext(png_file)[0] + '.jpg')
            
            # Save as JPEG with high quality
            img.save(jpeg_filename, 'JPEG', quality=95)
            print(f"Converted {png_file} to {jpeg_filename}")
            
        except Exception as e:
            print(f"Error converting {png_file}: {str(e)}")

if __name__ == "__main__":
    convert_png_to_jpeg() 
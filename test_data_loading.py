"""
Test script for data loading
"""
from step2_data_loading import load_and_preprocess_data

def main():
    print("Testing data loading step...")
    try:
        df = load_and_preprocess_data()
        print("\nData loading successful!")
    except Exception as e:
        print(f"\nError loading data: {str(e)}")
        print("\nPlease check if:")
        print("1. The data file '2006Fall_2017Spring_GOES_meteo_combined.csv' exists in the current directory")
        print("2. The file is not corrupted")
        print("3. You have enough memory to load the file")

if __name__ == "__main__":
    main() 
# Weather Prediction System using Machine Learning

A sophisticated weather prediction system that uses machine learning to analyze and predict weather patterns based on meteorological data.

## Project Overview

This project implements a machine learning-based weather prediction system that processes large-scale meteorological data to make accurate weather predictions. The system includes:

- Data preprocessing and analysis
- Machine learning model training
- Weather pattern visualization
- Time series forecasting

## Features

- **Data Processing**: Handles large-scale meteorological datasets (4.5GB+)
- **Machine Learning Models**: Implements deep learning models using TensorFlow
- **Visualization Tools**: Interactive plots and graphs for weather pattern analysis
- **Time Series Analysis**: Advanced forecasting capabilities

## Technical Stack

- Python
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn

## Project Structure

```
├── data/                    # Data files
├── models/                  # Trained models
├── visualizations/          # Visualization scripts and outputs
├── scripts/                 # Utility scripts
├── tests/                   # Test files
└── requirements.txt         # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/weather-prediction-system.git
cd weather-prediction-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preprocessing:
```bash
python scripts/data_preprocessing.py
```

2. Model Training:
```bash
python scripts/train.py
```

3. Generate Visualizations:
```bash
python visualizations/run_visualizations.py
```

## Results

The system provides:
- Weather pattern predictions
- Interactive data visualizations
- Time series analysis
- Model performance metrics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Special thanks to the meteorological data providers
- Inspired by the need for accurate weather prediction systems

# Rain Prediction using Meteorological and Satellite Data

This project implements a CNN-RNN hybrid model for rain prediction using meteorological and satellite data from Lake Michigan.

## Latest Results

The current model achieved the following performance metrics:

- Overall Accuracy: 91.15%
- F1 Score: 0.89 (for rain class)
- Class Distribution:
  - No Rain: 2472045 samples
  - Rain: 24000 samples

### Class Imbalance Analysis

The model shows significant class imbalance issues:
1. The dataset is heavily skewed towards "no rain" events (91.15% of samples)
2. The model is currently biased towards predicting "no rain" for all cases
3. While the overall accuracy appears high, the model fails to detect rain events (F1 score of 0.0)

This imbalance affects the model's ability to:
- Detect rare rain events
- Make accurate predictions for rain occurrences
- Generalize to real-world scenarios where rain detection is crucial

## Project Structure

```
.
├── config/
│   └── config.yaml           # Configuration file
├── data/
│   ├── raw/                 # Raw data files
│   ├── processed/           # Processed data files
│   └── cache/              # Cached data files
├── logs/                    # Log files
├── models/                  # Trained models
├── results/                 # Evaluation results
├── scripts/
│   ├── utils/
│   │   ├── logger.py       # Logging utilities
│   │   ├── config_manager.py # Configuration management
│   │   └── error_handler.py # Error handling utilities
│   ├── data_preparation.py # Data loading and preprocessing
│   ├── cnn_rnn_model.py    # Model architecture
│   ├── model_evaluation.py # Model evaluation
│   └── main_pipeline.py    # Main pipeline script
├── tests/                   # Test files
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rain-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Model

### Quick Start

To run the model with default settings:

```bash
python train_combined_model.py
```

This will:
1. Load the preprocessed data
2. Train the model for 1 epoch
3. Generate visualizations in the `visualizations/` directory
4. Save the trained model to `models/Combined_Model_seq8.h5`

### Data Requirements

The model expects the following data structure:
```
data/
├── processed/
│   ├── satellite_images/
│   │   └── sequences_8.npy    # Satellite image sequences
│   ├── meteorological_data.csv # Meteorological data
│   └── rain_labels.npy        # Rain labels
```

### Model Architecture

The current model uses:
- ConvLSTM2D for satellite image processing
- LSTM for meteorological data
- Combined dense layers for final prediction

### Output Files

After running the model, you'll find:
- Model file: `models/Combined_Model_seq8.h5`
- Training history plot: `visualizations/training_history_Combined_Model_seq8.png`
- Confusion matrix: `visualizations/confusion_matrix_Combined_Model_seq8.png`

## Configuration

The project uses a YAML configuration file (`config/config.yaml`) for all settings. You can override any configuration value using environment variables with the `RAIN_PREDICTION_` prefix:

```bash
# Example environment variables
export RAIN_PREDICTION_DATA__RAW_DATA_PATH="/path/to/raw/data"
export RAIN_PREDICTION_MODEL__TRAINING__EPOCHS=100
export RAIN_PREDICTION_LOGGING__LEVEL="DEBUG"
```

## Known Issues

1. **Class Imbalance**: The current model struggles with the imbalanced dataset. Future improvements should focus on:
   - Implementing class weights
   - Using data augmentation
   - Collecting more rain event data
   - Trying different model architectures

2. **Training Time**: The model takes significant time to train due to:
   - Large input dimensions
   - Complex architecture
   - Limited computational resources

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# ConvLSTM Data Preprocessing

This repository contains scripts for preprocessing meteorological and satellite image data for a ConvLSTM model.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Data Organization

Organize your data as follows:
```
data/
├── meteorological_data.csv    # CSV file containing meteorological data
└── satellite_images/         # Directory containing satellite images
    ├── YYYY-MM-DD_HH-MM.png  # Images named with timestamps
    └── ...
```

The meteorological data CSV should contain the following columns:
- Temp (F)
- RH (%)
- Wind Spd (mph)
- Wind Direction (deg)
- Visibility (mi)
- Atm Press (hPa)
- Precip (in)

## Usage

```python
from data_preprocessing import prepare_data

# Specify your data paths
meteo_file = 'data/meteorological_data.csv'
image_dir = 'data/satellite_images'

# Prepare the data (returns training and validation sets)
(X_train, y_train), (X_val, y_val), feature_names = prepare_data(
    meteo_file=meteo_file,
    image_dir=image_dir,
    seq_length=5  # Adjust sequence length as needed
)

# The returned data is ready for training a ConvLSTM model
print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Features used:", feature_names)
```

## Data Preprocessing Details

1. **Meteorological Data**:
   - Handles missing values using forward/backward fill
   - Normalizes features to [0, 1] range
   - Converts timestamps to datetime format

2. **Satellite Images**:
   - Resizes images to 128x128 pixels (configurable)
   - Normalizes pixel values to [0, 1]
   - Extracts timestamps from filenames

3. **Sequence Creation**:
   - Creates sequences of specified length
   - Each sequence becomes an input
   - The next image becomes the target
   - Data is split 80-20 for training/validation 

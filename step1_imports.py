"""
Step 1: Import Libraries and Set Configuration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense,
                                   Dropout, LSTM, Reshape, Attention, LayerNormalization)
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                      ModelCheckpoint, TensorBoard)
import os
from datetime import datetime

# Set visualization style
plt.style.use('ggplot')
sns.set_style("whitegrid")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42) 
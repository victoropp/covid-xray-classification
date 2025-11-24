"""
Configuration settings for COVID-19 X-ray Classification project.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
METADATA_DIR = DATA_DIR / "metadata"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
for dir_path in [PROCESSED_DATA_DIR, SPLITS_DIR, METADATA_DIR, MODELS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset configuration
CLASSES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
NUM_CLASSES = len(CLASSES)
CLASS_NAMES = {
    'COVID': 'COVID-19',
    'Lung_Opacity': 'Lung Opacity',
    'Normal': 'Normal',
    'Viral Pneumonia': 'Viral Pneumonia'
}

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Image preprocessing
IMG_SIZE = 224  # Standard for transfer learning models
IMG_CHANNELS = 3  # RGB
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet std

# Data augmentation (training only)
AUGMENTATION_CONFIG = {
    'horizontal_flip': True,
    'rotation_range': 10,
    'brightness_range': 0.2,
    'contrast_range': 0.2,
    'zoom_range': 0.1,
}

# Training configuration
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# Model configuration
MODELS_TO_TRAIN = ['resnet50', 'efficientnet_b3', 'densenet121']
PRETRAINED = True
FREEZE_LAYERS = True  # Freeze early layers for transfer learning
DROPOUT_RATE = 0.5

# Class weights (to handle imbalance)
# Will be calculated dynamically based on class distribution
USE_CLASS_WEIGHTS = True

# Clinical thresholds
# For COVID detection, we prioritize sensitivity (minimize false negatives)
COVID_SENSITIVITY_TARGET = 0.95  # Target 95% sensitivity for COVID class

# Device configuration
DEVICE = 'cuda'  # Will fallback to 'cpu' if CUDA not available

# Random seed for reproducibility
RANDOM_SEED = 42

# Grad-CAM configuration
GRADCAM_LAYER = 'layer4'  # For ResNet models
GRADCAM_ALPHA = 0.4  # Overlay transparency

# Streamlit configuration
STREAMLIT_PAGE_TITLE = "COVID-19 X-ray Classification"
STREAMLIT_PAGE_ICON = "🏥"
STREAMLIT_LAYOUT = "wide"

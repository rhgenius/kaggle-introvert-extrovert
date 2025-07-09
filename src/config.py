"""
Configuration file for the Kaggle competition
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
RAW_DATA_PATH = DATA_PATH / 'raw'
PROCESSED_DATA_PATH = DATA_PATH / 'processed'
MODELS_PATH = PROJECT_ROOT / 'models'
SUBMISSIONS_PATH = PROJECT_ROOT / 'submissions'
NOTEBOOKS_PATH = PROJECT_ROOT / 'notebooks'

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature engineering parameters
MAX_FEATURES = 50
SCALE_FEATURES = True
CREATE_INTERACTIONS = True

# Model hyperparameters
RF_PARAMS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

XGB_PARAMS = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2]
}

LGB_PARAMS = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Competition details
COMPETITION_NAME = 'playground-series-s5e7'
TARGET_COLUMN = 'Personality'
ID_COLUMN = 'id'
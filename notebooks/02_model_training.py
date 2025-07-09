import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
from pathlib import Path
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=== Kaggle Competition: Model Training ===")

# Load data
data_path = Path('../data/raw')
train_df = pd.read_csv(data_path / 'train.csv')
test_df = pd.read_csv(data_path / 'test.csv')
sample_submission = pd.read_csv(data_path / 'sample_submission.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Feature Engineering
print("\n=== Feature Engineering ===")
fe = FeatureEngineer()
X_train, X_test, y_train = fe.preprocess_data(train_df, test_df)

# Split training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"\nTraining split shape: {X_train_split.shape}")
print(f"Validation split shape: {X_val_split.shape}")

# Model Training
print("\n=== Model Training ===")
trainer = ModelTrainer(random_state=42)

# Train and evaluate models
best_model, best_score = trainer.train_best_model(X_train, y_train, use_ensemble=True)

print(f"\nBest model CV score: {best_score:.4f}")

# Validate on hold-out set
y_val_pred = trainer.predict(X_val_split)
val_accuracy = accuracy_score(y_val_split, y_val_pred)
print(f"Validation accuracy: {val_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_val_split, y_val_pred))

# Save model
model_path = Path('../models/best_model.pkl')
model_path.parent.mkdir(exist_ok=True)
trainer.save_model(model_path)

# Make predictions on test set
print("\n=== Making Predictions ===")
test_predictions = trainer.predict(X_test)

# Create submission file
submission = sample_submission.copy()
submission.iloc[:, 1] = test_predictions  # Assuming second column is target

# Save submission
submission_path = Path('../submissions/submission.csv')
submission_path.parent.mkdir(exist_ok=True)
submission.to_csv(submission_path, index=False)

print(f"Submission saved to {submission_path}")
print(f"Submission shape: {submission.shape}")
print("\nFirst 5 predictions:")
print(submission.head())

print("\n=== Training Complete ===")
print(f"Best validation accuracy: {val_accuracy:.4f}")
print(f"Model saved to: {model_path}")
print(f"Submission saved to: {submission_path}")
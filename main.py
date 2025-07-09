#!/usr/bin/env python3
"""
Main pipeline for Kaggle Introvert-Extrovert Classification Competition

Usage:
    python main.py --step all
    python main.py --step eda
    python main.py --step preprocess
    python main.py --step train
    python main.py --step predict
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_preprocessing import preprocess_data
from model_training import main as train_models
from utils import data_quality_check
import pandas as pd

def run_eda():
    """Run exploratory data analysis"""
    print("=== Running Exploratory Data Analysis ===")
    
    # Load raw data
    data_path = Path('data/raw')
    if not data_path.exists():
        print("Error: Raw data directory not found. Please download the dataset first.")
        return
    
    train_df = pd.read_csv(data_path / 'train.csv')
    test_df = pd.read_csv(data_path / 'test.csv')
    
    # Data quality checks
    data_quality_check(train_df, "Training Data")
    data_quality_check(test_df, "Test Data")
    
    print("\nFor detailed EDA, run: python notebooks/01_exploratory_data_analysis.py")

def run_preprocessing():
    """Run data preprocessing"""
    print("=== Running Data Preprocessing ===")
    preprocess_data()

def run_training():
    """Run model training"""
    print("=== Running Model Training ===")
    train_models()

def run_prediction():
    """Generate final predictions"""
    print("=== Generating Final Predictions ===")
    
    from utils import load_model, generate_submission_file
    
    # Load best model
    model_path = Path('models/best_model.pkl')
    if not model_path.exists():
        print("Error: Best model not found. Please run training first.")
        return
    
    model = load_model(model_path)
    
    # Load test data
    X_test = pd.read_csv('data/processed/X_test.csv')
    test_df = pd.read_csv('data/raw/test.csv')
    
    # Generate predictions
    predictions = model.predict(X_test)
    
    # Create submission
    test_ids = test_df['id'] if 'id' in test_df.columns else range(len(predictions))
    generate_submission_file(predictions, test_ids, 'final_submission.csv')

def main():
    parser = argparse.ArgumentParser(description='Kaggle Introvert-Extrovert Classification Pipeline')
    parser.add_argument('--step', choices=['all', 'eda', 'preprocess', 'train', 'predict'], 
                       default='all', help='Pipeline step to run')
    
    args = parser.parse_args()
    
    print("Kaggle Introvert-Extrovert Classification Pipeline")
    print("=" * 50)
    
    if args.step in ['all', 'eda']:
        run_eda()
        print()
    
    if args.step in ['all', 'preprocess']:
        run_preprocessing()
        print()
    
    if args.step in ['all', 'train']:
        run_training()
        print()
    
    if args.step in ['all', 'predict']:
        run_prediction()
        print()
    
    print("Pipeline completed!")

if __name__ == "__main__":
    main()
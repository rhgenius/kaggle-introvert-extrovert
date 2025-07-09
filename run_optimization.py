#!/usr/bin/env python3
"""
Optimized training runner for Kaggle competition

Usage:
    python run_optimization.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from optimized_training import optimized_training_pipeline

def main():
    print("Starting optimized training pipeline...")
    
    try:
        results = optimized_training_pipeline()
        
        print("\n" + "="*50)
        print("OPTIMIZATION COMPLETE!")
        print("="*50)
        print(f"Best CV Score: {results['best_cv_score']:.4f}")
        print(f"Validation Accuracy: {results['validation_accuracy']:.4f}")
        print(f"Submission file: {results['submission_path']}")
        print(f"Model saved to: {results['model_path']}")
        print("\nReady for Kaggle submission!")
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
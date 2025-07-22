import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from feature_engineering import FeatureEngineer
from advanced_feature_engineering import AdvancedFeatureEngineer
from ensemble_optimizer import EnsembleOptimizer
from utils import load_data, create_submission, evaluate_model

# Tambahkan import
from data_augmentation import DataAugmenter
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def optimized_training_pipeline():
    """Complete optimized training pipeline"""
    print("=== Optimized Kaggle Training Pipeline ===")
    
    # Load data - Fix: Use absolute path from project root
    project_root = Path(__file__).parent.parent  # Go up from src to project root
    data_path = project_root / 'data' / 'raw'
    train_df, test_df, sample_submission = load_data(data_path)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Basic feature engineering
    print("\n=== Basic Feature Engineering ===")
    fe_basic = FeatureEngineer()
    X_train_basic, X_test_basic, y_train = fe_basic.preprocess_data(train_df, test_df)
    
    # Advanced feature engineering
    print("\n=== Advanced Feature Engineering ===")
    fe_advanced = AdvancedFeatureEngineer()
    X_train_final, X_test_final, selected_features = fe_advanced.fit_transform(
        X_train_basic, y_train, X_test_basic
    )
    
    print(f"Final training shape: {X_train_final.shape}")
    print(f"Final test shape: {X_test_final.shape}")
    
    # Split for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_final, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Ensemble optimization
    print("\n=== Ensemble Optimization ===")
    ensemble_optimizer = EnsembleOptimizer(random_state=42)
    best_ensemble, best_score = ensemble_optimizer.fit_best_ensemble(
        X_train_split, y_train_split, X_test_final
    )
    
    # Validation
    print("\n=== Validation ===")
    val_predictions = best_ensemble.predict(X_val_split)
    val_accuracy = accuracy_score(y_val_split, val_predictions)
    
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print("\nValidation Classification Report:")
    print(classification_report(y_val_split, val_predictions))
    
    # Retrain on full data
    print("\n=== Final Training ===")
    final_ensemble = EnsembleOptimizer(random_state=42)
    final_model, _ = final_ensemble.fit_best_ensemble(X_train_final, y_train, X_test_final)
    
    # Final predictions
    print("\n=== Final Predictions ===")
    test_predictions = final_model.predict(X_test_final)
    
    # Convert back to original labels
    test_predictions_labels = fe_basic.target_encoder.inverse_transform(test_predictions)
    
    # Create submission - Fix: Use absolute path
    submission_path = project_root / 'submissions' / 'optimized_submission.csv'
    submission = create_submission(test_predictions_labels, sample_submission, submission_path)
    
    print(f"\nSubmission saved to: {submission_path}")
    print(f"Prediction distribution:")
    unique, counts = np.unique(test_predictions_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"{label}: {count} ({count/len(test_predictions_labels)*100:.1f}%)")
    
    # Save model - Fix: Use absolute path
    import joblib
    model_path = project_root / 'models' / 'optimized_ensemble.pkl'
    joblib.dump({
        'model': final_model,
        'feature_engineer_basic': fe_basic,
        'feature_engineer_advanced': fe_advanced,
        'selected_features': selected_features
    }, model_path)
    
    print(f"Model saved to: {model_path}")
    
    return {
        'validation_accuracy': val_accuracy,
        'best_cv_score': best_score,
        'submission_path': submission_path,
        'model_path': model_path
    }

if __name__ == "__main__":
    results = optimized_training_pipeline()
    print("\n=== Training Complete ===")
    print(f"Best CV Score: {results['best_cv_score']:.4f}")
    print(f"Validation Accuracy: {results['validation_accuracy']:.4f}")
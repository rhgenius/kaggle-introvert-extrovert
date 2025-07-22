"""
Utility functions for the Kaggle competition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
from pathlib import Path

def load_data(data_path):
    """
    Load training and test data
    """
    train_df = pd.read_csv(data_path / 'train.csv')
    test_df = pd.read_csv(data_path / 'test.csv')
    sample_submission = pd.read_csv(data_path / 'sample_submission.csv')
    
    return train_df, test_df, sample_submission

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluate model performance
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"=== {model_name} Performance ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def evaluate_model_comprehensive(y_true, y_pred, y_prob=None, model_name="Model"):
    """Comprehensive model evaluation"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import roc_auc_score, log_loss, matthews_corrcoef
    
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)  # Matthews correlation coefficient
    
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            metrics['log_loss'] = log_loss(y_true, y_prob)
        except:
            pass
    
    print(f"=== {model_name} Comprehensive Performance ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return cm

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    else:
        print("Model does not have feature_importances_ attribute")
        return None

def create_submission(predictions, sample_submission, filename):
    """
    Create submission file
    """
    submission = sample_submission.copy()
    submission.iloc[:, 1] = predictions
    submission.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
    return submission

def save_results(results, filename):
    """
    Save results to file
    """
    results_df = pd.DataFrame([results])
    results_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def load_model(model_path):
    """
    Load saved model
    """
    return joblib.load(model_path)

def save_model(model, model_path):
    """
    Save model to file
    """
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
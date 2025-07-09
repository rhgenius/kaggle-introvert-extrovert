import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Model training class for binary classification
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
    def initialize_models(self):
        """
        Initialize different models for ensemble
        """
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=-1
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state
            )
        }
    
    def train_single_model(self, model_name, X_train, y_train, X_val=None, y_val=None):
        """
        Train a single model
        """
        print(f"Training {model_name}...")
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            print(f"{model_name} validation accuracy: {accuracy:.4f}")
            return model, accuracy
        
        return model, None
    
    def cross_validate_models(self, X, y, cv=5):
        """
        Perform cross-validation for all models
        """
        print("Performing cross-validation...")
        
        cv_scores = {}
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        for model_name, model in self.models.items():
            print(f"Cross-validating {model_name}...")
            scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
            cv_scores[model_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            print(f"{model_name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_scores
    
    def hyperparameter_tuning(self, model_name, X, y, param_grid, cv=3):
        """
        Perform hyperparameter tuning for a specific model
        """
        print(f"Tuning hyperparameters for {model_name}...")
        
        model = self.models[model_name]
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_score_
    
    def create_ensemble(self, X, y):
        """
        Create an ensemble of best performing models
        """
        print("Creating ensemble model...")
        
        # Select top 3 models based on cross-validation
        cv_scores = self.cross_validate_models(X, y)
        top_models = sorted(cv_scores.items(), key=lambda x: x[1]['mean'], reverse=True)[:3]
        
        ensemble_models = []
        for model_name, _ in top_models:
            ensemble_models.append((model_name, self.models[model_name]))
        
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'
        )
        
        # Cross-validate ensemble
        ensemble_scores = cross_val_score(ensemble, X, y, cv=5, scoring='accuracy')
        print(f"Ensemble CV score: {ensemble_scores.mean():.4f} (+/- {ensemble_scores.std() * 2:.4f})")
        
        return ensemble, ensemble_scores.mean()
    
    def train_best_model(self, X, y, use_ensemble=True):
        """
        Train the best model based on cross-validation
        """
        self.initialize_models()
        
        if use_ensemble:
            self.best_model, self.best_score = self.create_ensemble(X, y)
        else:
            cv_scores = self.cross_validate_models(X, y)
            best_model_name = max(cv_scores.items(), key=lambda x: x[1]['mean'])[0]
            self.best_model = self.models[best_model_name]
            self.best_score = cv_scores[best_model_name]['mean']
        
        # Train on full dataset
        print("Training final model on full dataset...")
        self.best_model.fit(X, y)
        
        return self.best_model, self.best_score
    
    def save_model(self, model_path):
        """
        Save the trained model
        """
        if self.best_model is not None:
            joblib.dump(self.best_model, model_path)
            print(f"Model saved to {model_path}")
        else:
            print("No model to save. Train a model first.")
    
    def load_model(self, model_path):
        """
        Load a trained model
        """
        self.best_model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return self.best_model
    
    def predict(self, X):
        """
        Make predictions using the best model
        """
        if self.best_model is not None:
            return self.best_model.predict(X)
        else:
            raise ValueError("No model available. Train or load a model first.")
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        """
        if self.best_model is not None:
            return self.best_model.predict_proba(X)
        else:
            raise ValueError("No model available. Train or load a model first.")
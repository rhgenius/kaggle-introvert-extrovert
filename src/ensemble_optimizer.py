import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

class EnsembleOptimizer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.base_models = {}
        self.optimized_models = {}
        self.ensemble_model = None
        self.cv_scores = {}
        
    def create_base_models(self):
        """Create base models with default parameters"""
        self.base_models = {
            'rf': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            # Temporarily disable XGBoost due to pandas compatibility issues
            # 'xgb': xgb.XGBClassifier(
            #     random_state=self.random_state, 
            #     eval_metric='logloss',
            #     enable_categorical=False,
            #     tree_method='hist',
            #     verbosity=0,
            #     use_label_encoder=False,
            #     objective='binary:logistic'
            # ),
            'lgb': lgb.LGBMClassifier(random_state=self.random_state, verbose=-1),
            'cat': CatBoostClassifier(random_seed=self.random_state, verbose=False),
            'lr': LogisticRegression(random_state=self.random_state, max_iter=1000)
        }
        
    def optimize_hyperparameters(self, X_train, y_train, cv_folds=5, n_iter=20):
        """Optimize hyperparameters for each model"""
        print("Optimizing hyperparameters...")
        
        # Parameter grids
        param_grids = {
            'rf': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            # Temporarily disable XGBoost parameter grid
            # 'xgb': {
            #     'n_estimators': [100, 200, 300, 500],
            #     'max_depth': [3, 4, 5, 6, 8],
            #     'learning_rate': [0.01, 0.05, 0.1, 0.2],
            #     'subsample': [0.8, 0.9, 1.0],
            #     'colsample_bytree': [0.8, 0.9, 1.0],
            #     'reg_alpha': [0, 0.1, 0.5],
            #     'reg_lambda': [0, 0.1, 0.5]
            # },
            'lgb': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 4, 5, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [31, 50, 70, 100],
                'feature_fraction': [0.8, 0.9, 1.0],
                'bagging_fraction': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            },
            'cat': {
                'iterations': [100, 200, 300, 500],
                'depth': [4, 5, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5, 7, 9]
            },
            'lr': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in self.base_models.items():
            print(f"Optimizing {name}...")
            
            if name in param_grids:
                # Use RandomizedSearchCV for efficiency
                search = RandomizedSearchCV(
                    model, param_grids[name], 
                    n_iter=n_iter, cv=cv, 
                    scoring='accuracy', 
                    random_state=self.random_state,
                    n_jobs=-1
                )
                search.fit(X_train, y_train)
                
                self.optimized_models[name] = search.best_estimator_
                self.cv_scores[name] = search.best_score_
                
                print(f"{name} best score: {search.best_score_:.4f}")
                print(f"{name} best params: {search.best_params_}")
            else:
                # For models without param grid, use default
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                self.optimized_models[name] = model
                self.cv_scores[name] = scores.mean()
                
                print(f"{name} CV score: {scores.mean():.4f}")
    
    def create_voting_ensemble(self, X_train, y_train, top_n=5):
        """Create voting ensemble from top models"""
        print(f"Creating voting ensemble with top {top_n} models...")
        
        # Sort models by CV score
        sorted_models = sorted(self.cv_scores.items(), key=lambda x: x[1], reverse=True)
        top_models = sorted_models[:top_n]
        
        estimators = [(name, self.optimized_models[name]) for name, _ in top_models]
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        # Cross-validate ensemble
        cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=5, scoring='accuracy')
        print(f"Voting ensemble CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return voting_clf, cv_scores.mean()
    
    def create_stacking_ensemble(self, X_train, y_train, top_n=5):
        """Create stacking ensemble"""
        print(f"Creating stacking ensemble with top {top_n} models...")
        
        # Sort models by CV score
        sorted_models = sorted(self.cv_scores.items(), key=lambda x: x[1], reverse=True)
        top_models = sorted_models[:top_n]
        
        estimators = [(name, self.optimized_models[name]) for name, _ in top_models]
        
        # Use logistic regression as meta-learner
        meta_learner = LogisticRegression(random_state=self.random_state, max_iter=1000)
        
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )
        
        # Cross-validate ensemble
        cv_scores = cross_val_score(stacking_clf, X_train, y_train, cv=5, scoring='accuracy')
        print(f"Stacking ensemble CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return stacking_clf, cv_scores.mean()
    
    def create_weighted_ensemble(self, X_train, y_train, X_test):
        """Create weighted ensemble based on CV scores"""
        print("Creating weighted ensemble...")
        
        # Get predictions from all models
        train_predictions = {}
        test_predictions = {}
        
        for name, model in self.optimized_models.items():
            model.fit(X_train, y_train)
            train_predictions[name] = model.predict_proba(X_train)
            test_predictions[name] = model.predict_proba(X_test)
        
        # Calculate weights based on CV scores
        total_score = sum(self.cv_scores.values())
        weights = {name: score/total_score for name, score in self.cv_scores.items()}
        
        # Weighted average of predictions
        weighted_train_pred = np.zeros_like(list(train_predictions.values())[0])
        weighted_test_pred = np.zeros_like(list(test_predictions.values())[0])
        
        for name, weight in weights.items():
            weighted_train_pred += weight * train_predictions[name]
            weighted_test_pred += weight * test_predictions[name]
        
        return weighted_train_pred, weighted_test_pred, weights
    
    def fit_best_ensemble(self, X_train, y_train, X_test):
        """Fit and compare different ensemble methods"""
        print("Comparing ensemble methods...")
        
        # Create base models
        self.create_base_models()
        
        # Optimize hyperparameters
        self.optimize_hyperparameters(X_train, y_train)
        
        # Compare ensemble methods
        ensemble_scores = {}
        
        # Voting ensemble
        voting_clf, voting_score = self.create_voting_ensemble(X_train, y_train)
        ensemble_scores['voting'] = voting_score
        
        # Stacking ensemble
        stacking_clf, stacking_score = self.create_stacking_ensemble(X_train, y_train)
        ensemble_scores['stacking'] = stacking_score
        
        # Weighted ensemble
        _, _, weights = self.create_weighted_ensemble(X_train, y_train, X_test)
        ensemble_scores['weighted'] = max(self.cv_scores.values())  # Use best individual score as proxy
        
        # Select best ensemble
        best_method = max(ensemble_scores.items(), key=lambda x: x[1])
        print(f"\nBest ensemble method: {best_method[0]} with score: {best_method[1]:.4f}")
        
        if best_method[0] == 'voting':
            self.ensemble_model = voting_clf
        elif best_method[0] == 'stacking':
            self.ensemble_model = stacking_clf
        else:
            # For weighted, we'll use the best individual model
            best_individual = max(self.cv_scores.items(), key=lambda x: x[1])
            self.ensemble_model = self.optimized_models[best_individual[0]]
        
        # Fit the best ensemble
        self.ensemble_model.fit(X_train, y_train)
        
        return self.ensemble_model, best_method[1]
    
    def predict(self, X_test):
        """Make predictions using the best ensemble"""
        if self.ensemble_model is None:
            raise ValueError("No ensemble model fitted. Call fit_best_ensemble first.")
        
        return self.ensemble_model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities"""
        if self.ensemble_model is None:
            raise ValueError("No ensemble model fitted. Call fit_best_ensemble first.")
        
        return self.ensemble_model.predict_proba(X_test)
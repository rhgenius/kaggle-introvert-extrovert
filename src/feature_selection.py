import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel, RFE, RFECV, SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

class AdvancedFeatureSelector:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.selected_features = None
        self.feature_importances = None
        
    def select_features_with_boruta(self, X, y, max_iter=100):
        """Select features using Boruta algorithm"""
        # Initialize Boruta
        rf = RandomForestClassifier(n_jobs=-1, random_state=self.random_state)
        boruta = BorutaPy(rf, n_estimators='auto', max_iter=max_iter, random_state=self.random_state)
        
        # Fit Boruta
        boruta.fit(X.values, y)
        
        # Get selected features
        selected_features = X.columns[boruta.support_]
        
        self.selected_features = selected_features
        self.feature_importances = pd.Series(boruta.ranking_, index=X.columns)
        
        return X[selected_features]
    
    def select_features_with_rfecv(self, X, y, step=1):
        """Select features using Recursive Feature Elimination with Cross-Validation"""
        # Initialize RFECV
        rf = RandomForestClassifier(n_jobs=-1, random_state=self.random_state)
        rfecv = RFECV(estimator=rf, step=step, cv=5, scoring='accuracy', n_jobs=-1)
        
        # Fit RFECV
        rfecv.fit(X, y)
        
        # Get selected features
        selected_features = X.columns[rfecv.support_]
        
        self.selected_features = selected_features
        self.feature_importances = pd.Series(rfecv.ranking_, index=X.columns)
        
        return X[selected_features]
    
    def select_features_with_mutual_info(self, X, y, k='all'):
        """Select features using Mutual Information"""
        # Initialize SelectKBest with mutual_info_classif
        selector = SelectKBest(mutual_info_classif, k=k)
        
        # Fit selector
        selector.fit(X, y)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask]
        
        self.selected_features = selected_features
        self.feature_importances = pd.Series(selector.scores_, index=X.columns)
        
        return X[selected_features]
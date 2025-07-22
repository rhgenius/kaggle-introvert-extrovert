import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN

class DataAugmenter:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.smote = SMOTE(random_state=random_state)
        self.adasyn = ADASYN(random_state=random_state)
        
    def augment_with_smote(self, X, y):
        """Augment data using SMOTE"""
        print("Augmenting data with SMOTE...")
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def augment_with_adasyn(self, X, y):
        """Augment data using ADASYN"""
        print("Augmenting data with ADASYN...")
        X_resampled, y_resampled = self.adasyn.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def add_gaussian_noise(self, X, noise_level=0.05):
        """Add Gaussian noise to features"""
        print("Adding Gaussian noise...")
        X_noisy = X.copy()
        noise = np.random.normal(0, noise_level, X.shape)
        X_noisy = X_noisy + noise
        return X_noisy
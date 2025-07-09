import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    def __init__(self):
        self.poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        self.quantile_transformer = QuantileTransformer(n_quantiles=100, random_state=42)
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        self.feature_selector = None
        
    def create_interaction_features(self, X, max_features=10):
        """Create polynomial and interaction features"""
        print("Creating interaction features...")
        
        # Select top numeric features for interactions to avoid explosion
        numeric_cols = X.select_dtypes(include=[np.number]).columns[:max_features]
        
        if len(numeric_cols) > 1:
            X_interactions = self.poly_features.fit_transform(X[numeric_cols])
            
            # Get feature names
            feature_names = self.poly_features.get_feature_names_out(numeric_cols)
            X_interactions_df = pd.DataFrame(X_interactions, columns=feature_names, index=X.index)
            
            # Remove original features to avoid duplication
            interaction_only = X_interactions_df.drop(columns=numeric_cols, errors='ignore')
            
            return pd.concat([X, interaction_only], axis=1)
        
        return X
    
    def create_statistical_features(self, X):
        """Create statistical aggregation features"""
        print("Creating statistical features...")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            # Row-wise statistics
            X['row_mean'] = X[numeric_cols].mean(axis=1)
            X['row_std'] = X[numeric_cols].std(axis=1)
            X['row_min'] = X[numeric_cols].min(axis=1)
            X['row_max'] = X[numeric_cols].max(axis=1)
            X['row_median'] = X[numeric_cols].median(axis=1)
            X['row_skew'] = X[numeric_cols].skew(axis=1)
            X['row_range'] = X['row_max'] - X['row_min']
            X['row_iqr'] = X[numeric_cols].quantile(0.75, axis=1) - X[numeric_cols].quantile(0.25, axis=1)
            
            # Count features
            X['positive_count'] = (X[numeric_cols] > 0).sum(axis=1)
            X['negative_count'] = (X[numeric_cols] < 0).sum(axis=1)
            X['zero_count'] = (X[numeric_cols] == 0).sum(axis=1)
        
        return X
    
    def apply_transformations(self, X):
        """Apply various transformations for skewed features"""
        print("Applying feature transformations...")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if X[col].skew() > 1:  # Right skewed
                X[f'{col}_log'] = np.log1p(X[col] - X[col].min() + 1)
            
            if X[col].skew() < -1:  # Left skewed
                X[f'{col}_exp'] = np.expm1(X[col] - X[col].max())
            
            # Square root for positive values
            if X[col].min() >= 0:
                X[f'{col}_sqrt'] = np.sqrt(X[col])
            
            # Box-Cox like transformation
            if X[col].min() > 0:
                X[f'{col}_boxcox'] = np.log(X[col])
        
        return X
    
    def create_binning_features(self, X, n_bins=5):
        """Create binning features for continuous variables"""
        print("Creating binning features...")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            try:
                X[f'{col}_binned'] = pd.cut(X[col], bins=n_bins, labels=False, duplicates='drop')
            except:
                # If binning fails, create quantile-based bins
                X[f'{col}_binned'] = pd.qcut(X[col], q=n_bins, labels=False, duplicates='drop')
        
        return X
    
    def remove_outliers(self, X, method='iqr', factor=1.5):
        """Remove or cap outliers"""
        print("Handling outliers...")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                # Cap outliers instead of removing
                X[col] = X[col].clip(lower_bound, upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs((X[col] - X[col].mean()) / X[col].std())
                X[col] = X[col].where(z_scores < 3, X[col].median())
        
        return X
    
    def select_best_features(self, X_train, y_train, X_test, method='combined', k=50):
        """Advanced feature selection"""
        print(f"Selecting best {k} features using {method} method...")
        
        if method == 'statistical':
            selector = SelectKBest(score_func=f_classif, k=k)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            selected_features = X_train.columns[selector.get_support()]
            
        elif method == 'rfe':
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            selector = RFE(rf, n_features_to_select=k)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            selected_features = X_train.columns[selector.get_support()]
            
        elif method == 'combined':
            # Use both methods and take union
            selector1 = SelectKBest(score_func=f_classif, k=k)
            selector1.fit(X_train, y_train)
            features1 = set(X_train.columns[selector1.get_support()])
            
            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            selector2 = RFE(rf, n_features_to_select=k)
            selector2.fit(X_train, y_train)
            features2 = set(X_train.columns[selector2.get_support()])
            
            # Take intersection for most important features
            selected_features = list(features1 & features2)
            
            # If intersection is too small, add from union
            if len(selected_features) < k//2:
                union_features = list(features1 | features2)
                selected_features.extend(union_features[:k-len(selected_features)])
            
            selected_features = selected_features[:k]
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
        
        self.feature_selector = selected_features
        print(f"Selected {len(selected_features)} features")
        
        return X_train_selected, X_test_selected, selected_features
    
    def fit_transform(self, X_train, y_train, X_test):
        """Complete advanced feature engineering pipeline"""
        print("Starting advanced feature engineering...")
        
        # Make copies
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()
        
        # Step 1: Handle outliers
        X_train_processed = self.remove_outliers(X_train_processed)
        
        # Step 2: Create interaction features
        X_train_processed = self.create_interaction_features(X_train_processed)
        X_test_processed = self.create_interaction_features(X_test_processed)
        
        # Step 3: Create statistical features
        X_train_processed = self.create_statistical_features(X_train_processed)
        X_test_processed = self.create_statistical_features(X_test_processed)
        
        # Step 4: Apply transformations
        X_train_processed = self.apply_transformations(X_train_processed)
        X_test_processed = self.apply_transformations(X_test_processed)
        
        # Step 5: Create binning features
        X_train_processed = self.create_binning_features(X_train_processed)
        X_test_processed = self.create_binning_features(X_test_processed)
        
        # Step 6: Align columns
        common_cols = list(set(X_train_processed.columns) & set(X_test_processed.columns))
        X_train_processed = X_train_processed[common_cols]
        X_test_processed = X_test_processed[common_cols]
        
        # Step 7: Feature selection
        X_train_final, X_test_final, selected_features = self.select_best_features(
            X_train_processed, y_train, X_test_processed, k=min(100, len(common_cols))
        )
        
        print(f"Advanced feature engineering completed!")
        print(f"Original features: {X_train.shape[1]}")
        print(f"Final features: {X_train_final.shape[1]}")
        
        return X_train_final, X_test_final, selected_features
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Feature engineering class for the Introvert vs Extrovert prediction task
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        self.feature_columns = None  # Store all feature columns from training
        self.target_encoder = None  # Add target encoder
        
    def preprocess_data(self, train_df, test_df, target_col='Personality'):
        """
        Complete preprocessing pipeline with consistent column handling
        """
        print("Starting feature engineering...")
        
        # Step 1: Ensure both datasets have the same base columns
        train_processed = train_df.copy()
        test_processed = test_df.copy()
        
        # Get all columns from training set (excluding target)
        base_columns = [col for col in train_processed.columns if col != target_col]
        
        # Ensure test set has all base columns
        for col in base_columns:
            if col not in test_processed.columns:
                test_processed[col] = np.nan
        
        # Reorder test columns to match training
        test_processed = test_processed[base_columns]
        
        print(f"Base train columns: {len(base_columns)}")
        print(f"Base test columns: {len(test_processed.columns)}")
        
        # Step 2: Handle missing values consistently
        # Fill missing values for categorical columns
        categorical_cols = train_processed.select_dtypes(include=['object']).columns.tolist()
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        for col in categorical_cols:
            # Fill missing values with 'Unknown'
            train_processed[col] = train_processed[col].fillna('Unknown')
            test_processed[col] = test_processed[col].fillna('Unknown')
        
        # Fill missing values for numeric columns
        numeric_cols = train_processed.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        for col in numeric_cols:
            # Fill with median
            median_val = train_processed[col].median()
            train_processed[col] = train_processed[col].fillna(median_val)
            test_processed[col] = test_processed[col].fillna(median_val)
        
        # Step 3: Create features
        print("Creating new features...")
        train_processed = self.create_features(train_processed, fit=True)
        test_processed = self.create_features(test_processed, fit=False)
        
        # Step 4: Ensure column alignment after feature creation
        train_feature_cols = [col for col in train_processed.columns if col != target_col]
        
        # Add missing columns to test set
        for col in train_feature_cols:
            if col not in test_processed.columns:
                test_processed[col] = 0.0  # Fill with 0 for engineered features
        
        # Reorder test columns to match training
        test_processed = test_processed[train_feature_cols]
        
        print(f"Train features after creation: {train_processed.shape[1]}")
        print(f"Test features after creation: {test_processed.shape[1]}")
        
        # Step 5: Encode categorical features
        print("Encoding categorical features...")
        train_processed = self.encode_categorical_features(train_processed, fit=True)
        test_processed = self.encode_categorical_features(test_processed, fit=False)
        
        print(f"Train features after encoding: {train_processed.shape[1]}")
        print(f"Test features after encoding: {test_processed.shape[1]}")
        
        # Step 6: Final column alignment before scaling
        train_feature_cols = [col for col in train_processed.columns if col != target_col]
        test_processed = test_processed.reindex(columns=train_feature_cols, fill_value=0)
        
        # Step 7: Scale features
        print("Scaling features...")
        train_processed = self.scale_features(train_processed, fit=True)
        test_processed = self.scale_features(test_processed, fit=False)
        
        print(f"Train features after scaling: {train_processed.shape[1]}")
        print(f"Test features after scaling: {test_processed.shape[1]}")
        
        # Step 8: Separate features and target
        if target_col in train_processed.columns:
            X_train = train_processed.drop(columns=[target_col])
            y_train_raw = train_processed[target_col]
            
            # Encode target variable
            print("Encoding target variable...")
            self.target_encoder = LabelEncoder()
            y_train = self.target_encoder.fit_transform(y_train_raw)
            print(f"Target mapping: {dict(zip(self.target_encoder.classes_, self.target_encoder.transform(self.target_encoder.classes_)))}")
        else:
            X_train = train_processed
            y_train = None
        
        X_test = test_processed
        
        # Step 9: Feature selection
        if y_train is not None:
            print("Selecting features...")
            X_train = self.select_features(X_train, y_train)
            X_test = self.transform_features(X_test)
        
        print("Feature engineering complete!")
        print(f"Final training shape: {X_train.shape}")
        print(f"Final test shape: {X_test.shape}")
        
        return X_train, X_test, y_train
    
    def create_features(self, df, fit=True):
        """
        Create new features from existing ones
        """
        df_new = df.copy()
        
        # Get numeric columns (excluding target)
        numeric_cols = df_new.select_dtypes(include=[np.number]).columns.tolist()
        if 'Personality' in numeric_cols:
            numeric_cols.remove('Personality')
        
        if len(numeric_cols) >= 2:
            # Feature interactions (limit to avoid too many features)
            for i in range(min(len(numeric_cols), 5)):  # Limit to first 5 numeric columns
                for j in range(i+1, min(i+3, len(numeric_cols))):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    # Ratio features
                    df_new[f'{col1}_{col2}_ratio'] = df_new[col1] / (df_new[col2] + 1e-8)
                    # Sum features
                    df_new[f'{col1}_{col2}_sum'] = df_new[col1] + df_new[col2]
                    # Difference features
                    df_new[f'{col1}_{col2}_diff'] = df_new[col1] - df_new[col2]
            
            # Statistical features
            df_new['numeric_mean'] = df_new[numeric_cols].mean(axis=1)
            df_new['numeric_std'] = df_new[numeric_cols].std(axis=1)
            df_new['numeric_min'] = df_new[numeric_cols].min(axis=1)
            df_new['numeric_max'] = df_new[numeric_cols].max(axis=1)
            df_new['numeric_range'] = df_new['numeric_max'] - df_new['numeric_min']
        
        return df_new
    
    def encode_categorical_features(self, df, fit=True):
        """
        Encode categorical features using Label Encoding
        """
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target if present
        if 'Personality' in categorical_cols:
            categorical_cols.remove('Personality')
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df_encoded[col] = df_encoded[col].astype(str)
                    
                    # Handle unseen categories
                    unique_values = df_encoded[col].unique()
                    for val in unique_values:
                        if val not in le.classes_:
                            # Add unseen category to encoder
                            le.classes_ = np.append(le.classes_, val)
                    
                    df_encoded[col] = le.transform(df_encoded[col])
        
        return df_encoded
    
    def scale_features(self, df, fit=True):
        """
        Scale numeric features using StandardScaler
        """
        df_scaled = df.copy()
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target if present
        if 'Personality' in numeric_cols:
            numeric_cols.remove('Personality')
        
        if len(numeric_cols) > 0:
            if fit:
                df_scaled[numeric_cols] = self.scaler.fit_transform(df_scaled[numeric_cols])
            else:
                df_scaled[numeric_cols] = self.scaler.transform(df_scaled[numeric_cols])
        
        return df_scaled
    
    def select_features(self, X, y, k=50):
        """
        Select top k features using univariate feature selection
        """
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        print(f"Selected {len(self.selected_features)} features out of {X.shape[1]}")
        print(f"Selected features: {self.selected_features[:10]}...")  # Show first 10
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def transform_features(self, X):
        """
        Transform features using fitted feature selector
        """
        if self.feature_selector is None:
            return X
        
        # Ensure X has all selected features
        X_aligned = X.reindex(columns=self.selected_features, fill_value=0)
        
        X_selected = self.feature_selector.transform(X_aligned)
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
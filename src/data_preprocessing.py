import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from pathlib import Path
import joblib

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = None
        self.target_encoder = None
        
    def fit_transform(self, train_df, target_col=None):
        """Fit preprocessors and transform training data"""
        df = train_df.copy()
        
        # Separate features and target
        if target_col:
            X = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            X = df
            y = None
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Handle missing values
        X = self._handle_missing_values(X, fit=True)
        
        # Encode categorical variables
        X = self._encode_categorical(X, fit=True)
        
        # Scale numerical features
        X = self._scale_features(X, fit=True)
        
        # Encode target if provided
        if y is not None:
            y = self._encode_target(y, fit=True)
            return X, y
        
        return X
    
    def transform(self, test_df):
        """Transform test data using fitted preprocessors"""
        df = test_df.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df, fit=False)
        
        # Encode categorical variables
        df = self._encode_categorical(df, fit=False)
        
        # Scale numerical features
        df = self._scale_features(df, fit=False)
        
        return df
    
    def _handle_missing_values(self, df, fit=False):
        """Handle missing values in the dataset"""
        df_processed = df.copy()
        
        # Separate numeric and categorical columns
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        # Handle numeric missing values
        if len(numeric_cols) > 0:
            if fit:
                self.imputers['numeric'] = SimpleImputer(strategy='median')
                df_processed[numeric_cols] = self.imputers['numeric'].fit_transform(df_processed[numeric_cols])
            else:
                if 'numeric' in self.imputers:
                    df_processed[numeric_cols] = self.imputers['numeric'].transform(df_processed[numeric_cols])
        
        # Handle categorical missing values
        if len(categorical_cols) > 0:
            if fit:
                self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
                df_processed[categorical_cols] = self.imputers['categorical'].fit_transform(df_processed[categorical_cols])
            else:
                if 'categorical' in self.imputers:
                    df_processed[categorical_cols] = self.imputers['categorical'].transform(df_processed[categorical_cols])
        
        return df_processed
    
    def _encode_categorical(self, df, fit=False):
        """Encode categorical variables"""
        df_processed = df.copy()
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if fit:
                self.encoders[col] = LabelEncoder()
                df_processed[col] = self.encoders[col].fit_transform(df_processed[col].astype(str))
            else:
                if col in self.encoders:
                    # Handle unseen categories
                    unique_values = set(df_processed[col].astype(str))
                    known_values = set(self.encoders[col].classes_)
                    unseen_values = unique_values - known_values
                    
                    if unseen_values:
                        # Replace unseen values with most frequent class
                        most_frequent = self.encoders[col].classes_[0]
                        df_processed[col] = df_processed[col].astype(str).replace(list(unseen_values), most_frequent)
                    
                    df_processed[col] = self.encoders[col].transform(df_processed[col].astype(str))
        
        return df_processed
    
    def _scale_features(self, df, fit=False):
        """Scale numerical features"""
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            if fit:
                self.scalers['standard'] = StandardScaler()
                df_processed[numeric_cols] = self.scalers['standard'].fit_transform(df_processed[numeric_cols])
            else:
                if 'standard' in self.scalers:
                    df_processed[numeric_cols] = self.scalers['standard'].transform(df_processed[numeric_cols])
        
        return df_processed
    
    def _encode_target(self, y, fit=False):
        """Encode target variable if it's categorical"""
        if y.dtype == 'object':
            if fit:
                self.target_encoder = LabelEncoder()
                return self.target_encoder.fit_transform(y)
            else:
                if self.target_encoder:
                    return self.target_encoder.transform(y)
        return y
    
    def inverse_transform_target(self, y):
        """Inverse transform target variable"""
        if self.target_encoder:
            return self.target_encoder.inverse_transform(y)
        return y
    
    def save_preprocessors(self, filepath):
        """Save fitted preprocessors"""
        preprocessor_data = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'imputers': self.imputers,
            'feature_names': self.feature_names,
            'target_encoder': self.target_encoder
        }
        joblib.dump(preprocessor_data, filepath)
    
    def load_preprocessors(self, filepath):
        """Load fitted preprocessors"""
        preprocessor_data = joblib.load(filepath)
        self.scalers = preprocessor_data['scalers']
        self.encoders = preprocessor_data['encoders']
        self.imputers = preprocessor_data['imputers']
        self.feature_names = preprocessor_data['feature_names']
        self.target_encoder = preprocessor_data['target_encoder']

def preprocess_data():
    """Main preprocessing function"""
    # Load data
    data_path = Path('../data/raw')
    train_df = pd.read_csv(data_path / 'train.csv')
    test_df = pd.read_csv(data_path / 'test.csv')
    
    # Find target column
    target_col = None
    for col in train_df.columns:
        if col.lower() in ['personality', 'target', 'label', 'class']:
            target_col = col
            break
    
    if not target_col:
        print("Target column not found. Available columns:", list(train_df.columns))
        return
    
    print(f"Using '{target_col}' as target column")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Fit and transform training data
    X_train, y_train = preprocessor.fit_transform(train_df, target_col)
    
    # Transform test data
    X_test = preprocessor.transform(test_df)
    
    # Save processed data
    processed_path = Path('../data/processed')
    processed_path.mkdir(exist_ok=True)
    
    pd.DataFrame(X_train).to_csv(processed_path / 'X_train.csv', index=False)
    pd.DataFrame(y_train, columns=[target_col]).to_csv(processed_path / 'y_train.csv', index=False)
    pd.DataFrame(X_test).to_csv(processed_path / 'X_test.csv', index=False)
    
    # Save preprocessors
    preprocessor.save_preprocessors(processed_path / 'preprocessors.pkl')
    
    print(f"Processed data saved to {processed_path}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Target distribution: {pd.Series(y_train).value_counts()}")

if __name__ == "__main__":
    preprocess_data()
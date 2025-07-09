import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data():
    """Load training and test datasets"""
    data_path = Path('../data/raw')
    train_df = pd.read_csv(data_path / 'train.csv')
    test_df = pd.read_csv(data_path / 'test.csv')
    sample_submission = pd.read_csv(data_path / 'sample_submission.csv')
    return train_df, test_df, sample_submission

def basic_info(train_df, test_df, sample_submission):
    """Display basic information about datasets"""
    print("=== Dataset Info ===")
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Sample submission shape: {sample_submission.shape}")
    
    print("\n=== Training Data Info ===")
    print(train_df.info())
    
    print("\n=== Test Data Info ===")
    print(test_df.info())
    
    print("\n=== Missing Values in Training Data ===")
    missing_train = train_df.isnull().sum()
    print(missing_train[missing_train > 0])
    
    print("\n=== Missing Values in Test Data ===")
    missing_test = test_df.isnull().sum()
    print(missing_test[missing_test > 0])

def target_analysis(train_df):
    """Analyze target variable distribution"""
    target_col = None
    for col in train_df.columns:
        if col.lower() in ['personality', 'target', 'label', 'class']:
            target_col = col
            break
    
    if target_col:
        print(f"\n=== Target Variable: {target_col} ===")
        print(train_df[target_col].value_counts())
        print(f"\nTarget distribution (%):\n{train_df[target_col].value_counts(normalize=True) * 100}")
        
        # Plot target distribution
        plt.figure(figsize=(10, 6))
        train_df[target_col].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
        plt.title(f'Distribution of {target_col}')
        plt.xlabel(target_col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return target_col
    else:
        print("\n=== Target column not found ===")
        print("Available columns:", list(train_df.columns))
        return None

def feature_analysis(train_df, target_col):
    """Analyze features in the dataset"""
    print("\n=== Feature Analysis ===")
    
    # Separate numeric and categorical features
    numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_features:
        numeric_features.remove(target_col)
    
    categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_features:
        categorical_features.remove(target_col)
    
    print(f"Numeric features ({len(numeric_features)}): {numeric_features[:10]}...")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features[:10]}...")
    
    # Statistical summary for numeric features
    if numeric_features:
        print("\n=== Numeric Features Summary ===")
        print(train_df[numeric_features].describe())
        
        # Correlation with target if target is numeric
        if target_col and train_df[target_col].dtype in ['int64', 'float64']:
            correlations = train_df[numeric_features + [target_col]].corr()[target_col].sort_values(ascending=False)
            print(f"\n=== Top 10 Features Correlated with {target_col} ===")
            print(correlations.head(11))  # 11 to include target itself
    
    # Categorical features analysis
    if categorical_features:
        print("\n=== Categorical Features Summary ===")
        for col in categorical_features[:5]:  # Show first 5 categorical features
            print(f"\n{col}: {train_df[col].nunique()} unique values")
            print(train_df[col].value_counts().head())

def plot_distributions(train_df, target_col):
    """Plot feature distributions"""
    numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_features:
        numeric_features.remove(target_col)
    
    if len(numeric_features) > 0:
        # Plot histograms for numeric features
        n_features = min(12, len(numeric_features))
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.ravel()
        
        for i, feature in enumerate(numeric_features[:n_features]):
            train_df[feature].hist(bins=30, ax=axes[i], alpha=0.7)
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(n_features, 12):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()

def correlation_analysis(train_df, target_col):
    """Analyze correlations between features"""
    numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_features) > 1:
        # Correlation matrix
        plt.figure(figsize=(15, 12))
        correlation_matrix = train_df[numeric_features].corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            print("\n=== Highly Correlated Feature Pairs (|correlation| > 0.8) ===")
            for feat1, feat2, corr in high_corr_pairs:
                print(f"{feat1} - {feat2}: {corr:.3f}")

def main():
    """Main EDA function"""
    print("Starting Exploratory Data Analysis...\n")
    
    # Load data
    train_df, test_df, sample_submission = load_data()
    
    # Basic information
    basic_info(train_df, test_df, sample_submission)
    
    # Target analysis
    target_col = target_analysis(train_df)
    
    # Feature analysis
    feature_analysis(train_df, target_col)
    
    # Plot distributions
    plot_distributions(train_df, target_col)
    
    # Correlation analysis
    correlation_analysis(train_df, target_col)
    
    print("\n=== EDA Complete ===")
    print("Next steps:")
    print("1. Data preprocessing and feature engineering")
    print("2. Model selection and training")
    print("3. Model evaluation and hyperparameter tuning")
    print("4. Generate predictions for submission")

if __name__ == "__main__":
    main()
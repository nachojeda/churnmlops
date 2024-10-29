import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    def __init__(self):
        self.numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean
            ('scaler', StandardScaler())  # Scale features
        ])
        
        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with most frequent
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # OneHotEncode categorical features
        ])
        
        # Initialize the full preprocessor
        self.preprocessor = None

    def preprocess(self, X):
        # Separate numeric and categorical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        # Create the column transformer that applies preprocessing to each column type
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numerical_transformer, numerical_cols),
                ('cat', self.categorical_transformer, categorical_cols)
            ]
        )
        
        # Fit and transform the data
        X_preprocessed = self.preprocessor.fit_transform(X)
        
        return X_preprocessed
    
    def transform(self, X):
        """ Apply the same transformations to test/validation data """
        if self.preprocessor:
            return self.preprocessor.transform(X)
        else:
            raise ValueError("The preprocessor is not fitted yet. Please run 'preprocess()' first.")

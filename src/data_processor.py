"""
Data Processing Module for Material Strength & Selection System
Handles data ingestion, cleaning, and feature engineering for material properties.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
from typing import Tuple, Dict, Any

class MaterialDataProcessor:
    """
    A class to handle all data processing operations for material data.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the data processor.
        
        Args:
            data_path (str): Path to the material data CSV file
        """
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = []
        self.target_columns = []
        
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load material data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file. If None, uses self.data_path
            
        Returns:
            pd.DataFrame: Loaded data
        """
        if file_path:
            self.data_path = file_path
        
        if not self.data_path or not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def explore_data(self) -> Dict[str, Any]:
        """
        Perform basic data exploration and return summary statistics.
        
        Returns:
            Dict: Summary of data exploration
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please load data first.")
        
        exploration_summary = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'data_types': dict(self.data.dtypes),
            'missing_values': dict(self.data.isnull().sum()),
            'numeric_summary': self.data.describe(),
            'categorical_summary': {}
        }
        
        # Get categorical column summaries
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            exploration_summary['categorical_summary'][col] = {
                'unique_count': self.data[col].nunique(),
                'unique_values': list(self.data[col].unique())
            }
        
        return exploration_summary
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the data by handling missing values and outliers.
        
        Returns:
            pd.DataFrame: Cleaned data
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please load data first.")
        
        # Create a copy for processing
        cleaned_data = self.data.copy()
        
        # Handle missing values
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
        categorical_columns = cleaned_data.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_columns:
            if cleaned_data[col].isnull().sum() > 0:
                cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_columns:
            if cleaned_data[col].isnull().sum() > 0:
                cleaned_data[col].fillna(cleaned_data[col].mode()[0], inplace=True)
        
        # Remove extreme outliers using IQR method for key numeric columns
        strength_columns = ['tensile_strength_mpa', 'yield_strength_mpa']
        for col in strength_columns:
            if col in cleaned_data.columns:
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # Use 3*IQR for more conservative outlier removal
                upper_bound = Q3 + 3 * IQR
                
                # Cap outliers instead of removing them to preserve data
                cleaned_data[col] = cleaned_data[col].clip(lower=lower_bound, upper=upper_bound)
        
        print(f"Data cleaned. Shape after cleaning: {cleaned_data.shape}")
        return cleaned_data
    
    def engineer_features(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Perform feature engineering on the material data.
        
        Args:
            data (pd.DataFrame): Data to engineer features for. If None, uses self.data
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data available for feature engineering.")
            data = self.data.copy()
        else:
            data = data.copy()
        
        # Calculate strength-to-weight ratio
        if 'tensile_strength_mpa' in data.columns and 'density_kg_m3' in data.columns:
            data['strength_to_weight_ratio'] = data['tensile_strength_mpa'] / data['density_kg_m3']
        
        # Calculate cost per strength unit
        if 'cost_per_kg_usd' in data.columns and 'tensile_strength_mpa' in data.columns:
            data['cost_per_strength'] = data['cost_per_kg_usd'] / (data['tensile_strength_mpa'] + 1)  # +1 to avoid division by zero
        
        # Calculate overall performance score
        if all(col in data.columns for col in ['tensile_strength_mpa', 'availability_score', 'environmental_impact_score']):
            # Normalize scores to 0-1 range for combination
            strength_norm = (data['tensile_strength_mpa'] - data['tensile_strength_mpa'].min()) / (data['tensile_strength_mpa'].max() - data['tensile_strength_mpa'].min())
            availability_norm = data['availability_score'] / 10  # Assuming availability is 0-10 scale
            environmental_norm = data['environmental_impact_score'] / 10  # Assuming environmental impact is 0-10 scale
            
            # Combined performance score (higher is better)
            data['performance_score'] = (strength_norm * 0.4 + availability_norm * 0.3 + environmental_norm * 0.3)
        
        # Create material category groups
        if 'material_type' in data.columns:
            # Group similar materials
            metal_types = ['Steel', 'Aluminum', 'Titanium', 'Copper', 'Brass', 'Iron', 'Magnesium', 'Zinc', 'Nickel', 'Bronze']
            data['is_metal'] = data['material_type'].isin(metal_types).astype(int)
            
            plastic_types = ['Plastic', 'Rubber']
            data['is_polymer'] = data['material_type'].isin(plastic_types).astype(int)
            
            advanced_types = ['Composite', 'Ceramic']
            data['is_advanced'] = data['material_type'].isin(advanced_types).astype(int)
        
        # Create cost categories
        if 'cost_per_kg_usd' in data.columns:
            data['cost_category'] = pd.cut(data['cost_per_kg_usd'], 
                                         bins=[0, 1, 5, 15, 50, float('inf')], 
                                         labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        print(f"Feature engineering completed. New shape: {data.shape}")
        return data
    
    def encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features for machine learning.
        
        Args:
            data (pd.DataFrame): Data with categorical features to encode
            
        Returns:
            pd.DataFrame: Data with encoded categorical features
        """
        encoded_data = data.copy()
        
        # Get categorical columns
        categorical_columns = encoded_data.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in ['material_name']:  # Don't encode material names
                # Use label encoding for now (could be extended to one-hot encoding)
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                
                encoded_data[col] = self.label_encoders[col].fit_transform(encoded_data[col].astype(str))
        
        return encoded_data
    
    def prepare_features_targets(self, data: pd.DataFrame, 
                               target_columns: list = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features and targets for machine learning.
        
        Args:
            data (pd.DataFrame): Processed data
            target_columns (list): List of target column names
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Features and targets
        """
        if target_columns is None:
            target_columns = ['tensile_strength_mpa', 'cost_per_kg_usd']
        
        # Define feature columns (exclude ID, name, and target columns)
        exclude_columns = ['material_id', 'material_name'] + target_columns
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        
        X = data[feature_columns].copy()
        y = data[target_columns].copy()
        
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        
        return X, y
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using StandardScaler.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features (optional)
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Scaled training and test features
        """
        # Identify numeric columns for scaling
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return X_train, X_test
        
        # Initialize scaler
        self.scalers['features'] = StandardScaler()
        
        # Scale training data
        X_train_scaled = X_train.copy()
        X_train_scaled[numeric_columns] = self.scalers['features'].fit_transform(X_train[numeric_columns])
        
        # Scale test data if provided
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numeric_columns] = self.scalers['features'].transform(X_test[numeric_columns])
        
        return X_train_scaled, X_test_scaled
    
    def process_pipeline(self, file_path: str = None, test_size: float = 0.2, 
                        random_state: int = 42) -> Dict[str, Any]:
        """
        Complete data processing pipeline.
        
        Args:
            file_path (str): Path to data file
            test_size (float): Proportion of data for testing
            random_state (int): Random state for reproducibility
            
        Returns:
            Dict: Processed data split into train/test sets
        """
        # Load data
        self.load_data(file_path)
        
        # Explore data
        exploration = self.explore_data()
        print("Data exploration completed.")
        
        # Clean data
        cleaned_data = self.clean_data()
        
        # Engineer features
        engineered_data = self.engineer_features(cleaned_data)
        
        # Encode categorical features
        encoded_data = self.encode_categorical_features(engineered_data)
        
        # Prepare features and targets
        X, y = self.prepare_features_targets(encoded_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Store processed data
        self.processed_data = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'exploration': exploration,
            'original_data': engineered_data
        }
        
        print("Data processing pipeline completed successfully!")
        return self.processed_data
    
    def get_material_info(self, material_id: int = None, material_name: str = None) -> pd.DataFrame:
        """
        Get information for a specific material.
        
        Args:
            material_id (int): Material ID
            material_name (str): Material name
            
        Returns:
            pd.DataFrame: Material information
        """
        if self.data is None:
            raise ValueError("Data not loaded.")
        
        if material_id is not None:
            return self.data[self.data['material_id'] == material_id]
        elif material_name is not None:
            return self.data[self.data['material_name'].str.contains(material_name, case=False)]
        else:
            raise ValueError("Either material_id or material_name must be provided.")

if __name__ == "__main__":
    # Example usage
    processor = MaterialDataProcessor()
    
    # Process the data
    data_path = "../data/materials.csv"
    if os.path.exists(data_path):
        processed_data = processor.process_pipeline(data_path)
        print(f"Training set shape: {processed_data['X_train'].shape}")
        print(f"Test set shape: {processed_data['X_test'].shape}")
        print(f"Feature columns: {processed_data['feature_columns']}")
    else:
        print(f"Data file not found: {data_path}")
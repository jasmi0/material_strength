"""
Machine Learning Models Module for Material Strength & Selection System
Implements Random Forest for strength prediction and Linear Regression for cost prediction.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MaterialStrengthPredictor:
    """
    Machine Learning model for predicting material strength properties.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the strength predictor.
        
        Args:
            model_type (str): Type of model ('random_forest', 'linear', 'ridge', 'lasso')
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_importance_ = None
        self.performance_metrics = {}
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(random_state=42, n_jobs=-1)
        elif model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(random_state=42)
        elif model_type == 'lasso':
            self.model = Lasso(random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def get_hyperparameter_grid(self) -> Dict[str, Any]:
        """
        Get hyperparameter grid for model tuning.
        
        Returns:
            Dict: Hyperparameter grid
        """
        if self.model_type == 'random_forest':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        elif self.model_type == 'ridge':
            return {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        elif self.model_type == 'lasso':
            return {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        else:
            return {}
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           cv_folds: int = 5) -> Dict[str, Any]:
        """
        Tune hyperparameters using grid search with cross-validation.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training targets
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dict: Best parameters and scores
        """
        param_grid = self.get_hyperparameter_grid()
        
        if not param_grid:
            print(f"No hyperparameters to tune for {self.model_type}")
            return {'best_params': {}, 'best_score': None}
        
        print(f"Tuning hyperparameters for {self.model_type} model...")
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,  # Convert back to positive MSE
            'cv_results': grid_search.cv_results_
        }
        
        print(f"Best parameters: {results['best_params']}")
        print(f"Best cross-validation score (MSE): {results['best_score']:.4f}")
        
        return results
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              tune_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training targets
            tune_hyperparameters (bool): Whether to tune hyperparameters
            
        Returns:
            Dict: Training results
        """
        print(f"Training {self.model_type} model for strength prediction...")
        
        # Tune hyperparameters if requested
        tuning_results = {}
        if tune_hyperparameters:
            tuning_results = self.tune_hyperparameters(X_train, y_train)
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Store feature importance for tree-based models
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, train_pred)
        
        results = {
            'model_type': self.model_type,
            'tuning_results': tuning_results,
            'train_metrics': train_metrics,
            'feature_importance': self.feature_importance_
        }
        
        print(f"Training completed. R² score: {train_metrics['r2']:.4f}")
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (pd.DataFrame): Features for prediction
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test targets
            
        Returns:
            Dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation.")
        
        predictions = self.predict(X_test)
        metrics = self._calculate_metrics(y_test, predictions)
        
        self.performance_metrics = metrics
        return metrics
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true (pd.Series): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            Dict: Calculated metrics
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving.")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance_,
            'performance_metrics': self.performance_metrics
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_importance_ = model_data.get('feature_importance')
        self.performance_metrics = model_data.get('performance_metrics', {})
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")

class MaterialCostPredictor:
    """
    Machine Learning model for predicting material costs.
    """
    
    def __init__(self, model_type: str = 'linear'):
        """
        Initialize the cost predictor.
        
        Args:
            model_type (str): Type of model ('linear', 'ridge', 'lasso', 'random_forest')
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_importance_ = None
        self.performance_metrics = {}
        
        # Initialize model based on type
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(random_state=42)
        elif model_type == 'lasso':
            self.model = Lasso(random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train the cost prediction model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training targets (costs)
            
        Returns:
            Dict: Training results
        """
        print(f"Training {self.model_type} model for cost prediction...")
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Store feature importance for tree-based models
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, train_pred)
        
        results = {
            'model_type': self.model_type,
            'train_metrics': train_metrics,
            'feature_importance': self.feature_importance_
        }
        
        print(f"Cost prediction training completed. R² score: {train_metrics['r2']:.4f}")
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make cost predictions.
        
        Args:
            X (pd.DataFrame): Features for prediction
            
        Returns:
            np.ndarray: Cost predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the cost prediction model.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test targets
            
        Returns:
            Dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation.")
        
        predictions = self.predict(X_test)
        metrics = self._calculate_metrics(y_test, predictions)
        
        self.performance_metrics = metrics
        return metrics
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true (pd.Series): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            Dict: Calculated metrics
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

class MaterialModelManager:
    """
    Manager class for handling multiple material prediction models.
    """
    
    def __init__(self):
        """
        Initialize the model manager.
        """
        self.strength_model = None
        self.cost_model = None
        self.models_trained = False
        self.feature_columns = None
    
    def train_models(self, processed_data: Dict[str, Any], 
                    strength_model_type: str = 'random_forest',
                    cost_model_type: str = 'linear') -> Dict[str, Any]:
        """
        Train both strength and cost prediction models.
        
        Args:
            processed_data (Dict): Processed data from MaterialDataProcessor
            strength_model_type (str): Type of model for strength prediction
            cost_model_type (str): Type of model for cost prediction
            
        Returns:
            Dict: Training results for both models
        """
        X_train = processed_data['X_train']
        X_test = processed_data['X_test']
        y_train = processed_data['y_train']
        y_test = processed_data['y_test']
        
        self.feature_columns = processed_data['feature_columns']
        
        # Train strength prediction model
        self.strength_model = MaterialStrengthPredictor(strength_model_type)
        strength_results = self.strength_model.train(
            X_train, y_train['tensile_strength_mpa'], tune_hyperparameters=True
        )
        
        # Evaluate strength model
        strength_test_metrics = self.strength_model.evaluate(
            X_test, y_test['tensile_strength_mpa']
        )
        
        # Train cost prediction model
        self.cost_model = MaterialCostPredictor(cost_model_type)
        cost_results = self.cost_model.train(
            X_train, y_train['cost_per_kg_usd']
        )
        
        # Evaluate cost model
        cost_test_metrics = self.cost_model.evaluate(
            X_test, y_test['cost_per_kg_usd']
        )
        
        self.models_trained = True
        
        results = {
            'strength_model': {
                'training': strength_results,
                'test_metrics': strength_test_metrics
            },
            'cost_model': {
                'training': cost_results,
                'test_metrics': cost_test_metrics
            }
        }
        
        print("All models trained successfully!")
        print(f"Strength prediction R² (test): {strength_test_metrics['r2']:.4f}")
        print(f"Cost prediction R² (test): {cost_test_metrics['r2']:.4f}")
        
        return results
    
    def predict_material_properties(self, material_features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict both strength and cost for given material features.
        
        Args:
            material_features (pd.DataFrame): Material features
            
        Returns:
            Dict: Predictions for strength and cost
        """
        if not self.models_trained:
            raise ValueError("Models must be trained before making predictions.")
        
        # Ensure features match training columns
        if list(material_features.columns) != self.feature_columns:
            # Reorder or filter columns to match training
            material_features = material_features[self.feature_columns]
        
        strength_pred = self.strength_model.predict(material_features)
        cost_pred = self.cost_model.predict(material_features)
        
        return {
            'predicted_strength': strength_pred,
            'predicted_cost': cost_pred
        }
    
    def save_models(self, model_dir: str):
        """
        Save both trained models.
        
        Args:
            model_dir (str): Directory to save models
        """
        if not self.models_trained:
            raise ValueError("Models must be trained before saving.")
        
        os.makedirs(model_dir, exist_ok=True)
        
        strength_path = os.path.join(model_dir, 'strength_model.pkl')
        cost_path = os.path.join(model_dir, 'cost_model.pkl')
        
        self.strength_model.save_model(strength_path)
        self.cost_model.save_model(cost_path)
        
        # Save feature columns
        feature_path = os.path.join(model_dir, 'feature_columns.pkl')
        joblib.dump(self.feature_columns, feature_path)
        
        print(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir: str):
        """
        Load both trained models.
        
        Args:
            model_dir (str): Directory containing saved models
        """
        strength_path = os.path.join(model_dir, 'strength_model.pkl')
        cost_path = os.path.join(model_dir, 'cost_model.pkl')
        feature_path = os.path.join(model_dir, 'feature_columns.pkl')
        
        self.strength_model = MaterialStrengthPredictor()
        self.strength_model.load_model(strength_path)
        
        self.cost_model = MaterialCostPredictor()
        self.cost_model.load_model(cost_path)
        
        if os.path.exists(feature_path):
            self.feature_columns = joblib.load(feature_path)
        
        self.models_trained = True
        print(f"Models loaded from {model_dir}")

if __name__ == "__main__":
    # Example usage
    from data_processor import MaterialDataProcessor
    
    # Process data
    processor = MaterialDataProcessor()
    data_path = "../data/materials.csv"
    
    if os.path.exists(data_path):
        processed_data = processor.process_pipeline(data_path)
        
        # Train models
        model_manager = MaterialModelManager()
        results = model_manager.train_models(processed_data)
        
        # Save models
        model_manager.save_models("../models")
        
        print("Model training and saving completed!")
    else:
        print(f"Data file not found: {data_path}")
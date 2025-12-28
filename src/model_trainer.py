"""
Regression Model Training & Evaluation Module

This module provides a production-grade regression suite with comprehensive
model evaluation metrics and performance comparison capabilities.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict, Any
import logging


class RegressorSuite:
    """
    Production-grade regression model training and evaluation framework.
    
    Supports multiple regression algorithms with standardized evaluation metrics.
    
    Attributes:
        logger (logging.Logger): Logger instance for operation tracking.
        models (Dict[str, Any]): Dictionary of initialized regression models.
        results (Dict[str, float]): Performance metrics storage.
    """
    
    def __init__(self):
        """Initialize RegressorSuite with multiple regression models."""
        self.logger = logging.getLogger(__name__)
        
        self.models: Dict[str, Any] = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(
                n_estimators=100, 
                random_state=42,
                n_jobs=-1  # Parallel processing
            )
        }
        
        self.results: Dict[str, Dict[str, float]] = {}
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        target: str = 'Total Spent',
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into features (X) and target (y), then train/test sets.
        
        Args:
            df (pd.DataFrame): Complete feature-engineered dataframe.
            target (str): Target variable column name.
            test_size (float): Proportion of data for testing (0.0-1.0).
            random_state (int): Random seed for reproducibility.
            
        Returns:
            Tuple containing:
                - X_train (pd.DataFrame): Training features
                - X_test (pd.DataFrame): Test features
                - y_train (pd.Series): Training target
                - y_test (pd.Series): Test target
                
        Raises:
            KeyError: If target column is missing.
            ValueError: If test_size is invalid.
        """
        if target not in df.columns:
            self.logger.error(f"Target column '{target}' not found!")
            raise KeyError(f"Missing target column: {target}")
        
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        
        self.logger.info(f"Defining target variable: {target}")
        self.logger.info(f"Defining features (X): All columns except {target}")
        
        X = df.drop(columns=[target])
        y = df[target]
        
        self.logger.info(
            f"Splitting data: {int((1-test_size)*100)}% train, "
            f"{int(test_size*100)}% test"
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
        
        self.logger.info(
            f"Split complete. Train: {X_train.shape}, Test: {X_test.shape}"
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> None:
        """
        Train all regression models in the suite.
        
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target values.
            
        Raises:
            ValueError: If training data is empty or invalid.
        """
        if X_train.empty or y_train.empty:
            raise ValueError("Training data cannot be empty!")
        
        self.logger.info("Training regression models...")
        
        for name, model in self.models.items():
            try:
                self.logger.info(f"Training {name}...")
                model.fit(X_train, y_train)
                self.logger.info(f"{name} training complete.")
            except Exception as e:
                self.logger.error(f"{name} training failed: {e}")
                raise
    
    def evaluate(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Evaluate all trained models and return performance metrics.
        
        Metrics computed:
        - RMSE (Root Mean Squared Error): Lower is better
        - MAE (Mean Absolute Error): Lower is better
        - R² Score: Higher is better (max 1.0)
        
        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target values.
            
        Returns:
            pd.DataFrame: Performance comparison table.
            
        Example:
            >>> metrics_df = suite.evaluate(X_test, y_test)
            >>> print(metrics_df.to_markdown(index=False))
        """
        self.logger.info("Evaluating model performance on test set.")
        
        metrics_list = []
        
        for name, model in self.models.items():
            try:
                self.logger.info(f"Evaluating {name}...")
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                self.logger.info(
                    f"{name} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}"
                )
                
                metrics_list.append({
                    "Model": name,
                    "RMSE": round(rmse, 4),
                    "MAE": round(mae, 4),
                    "R2_Score": round(r2, 4)
                })
                
                # Store for potential later use
                self.results[name] = {
                    "RMSE": rmse,
                    "MAE": mae,
                    "R2": r2
                }
                
            except Exception as e:
                self.logger.error(f"{name} evaluation failed: {e}")
                raise
        
        return pd.DataFrame(metrics_list)
    
    def get_best_model(self) -> Tuple[str, Dict[str, float]]:
        """
        Identify the best performing model based on lowest RMSE.
        
        Returns:
            Tuple[str, Dict[str, float]]: Model name and its metrics.
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate() first.")
        
        best_model = min(self.results.items(), key=lambda x: x[1]['RMSE'])
        self.logger.info(f"Best model: {best_model[0]} (RMSE: {best_model[1]['RMSE']:.4f})")
        
        return best_model

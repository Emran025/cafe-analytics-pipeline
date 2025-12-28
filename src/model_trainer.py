"""
Model Training & Evaluation Module

This module provides a production-grade ML training suite supporting both
regression and classification tasks with comprehensive evaluation metrics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score
)
from typing import Tuple, Dict, Any, List, Optional
import logging


class ModelTrainer:
    """
    Production-grade ML training and evaluation framework.
    
    Supports both regression and classification tasks with standardized 
    evaluation metrics and model comparison capabilities.
    
    Attributes:
        logger (logging.Logger): Logger instance for operation tracking.
        task_type (str): Either 'regression' or 'classification'.
        models (Dict[str, Any]): Dictionary of initialized models.
        results (Dict[str, Dict[str, float]]): Performance metrics storage.
    """
    
    def __init__(self, task_type: str = 'regression'):
        """
        Initialize ModelTrainer with task-specific models.
        
        Args:
            task_type (str): Either 'regression' or 'classification'.
            
        Raises:
            ValueError: If task_type is not 'regression' or 'classification'.
        """
        self.logger = logging.getLogger(__name__)
        
        if task_type not in ['regression', 'classification']:
            raise ValueError(
                f"Invalid task_type: '{task_type}'. "
                "Must be 'regression' or 'classification'."
            )
        
        self.task_type = task_type
        self.models: Dict[str, Any] = self.initialize_models(task_type)
        self.results: Dict[str, Dict[str, float]] = {}
        
        self.logger.info(f"ModelTrainer initialized for {task_type} task.")
    
    def initialize_models(self, task_type: str) -> Dict[str, Any]:
        """
        Initialize models based on task type.
        
        Args:
            task_type (str): Either 'regression' or 'classification'.
            
        Returns:
            Dict[str, Any]: Dictionary of initialized models.
        """
        if task_type == 'regression':
            self.logger.info("Loading regression models...")
            return {
                "LinearRegression": LinearRegression(),
                "RandomForest": RandomForestRegressor(
                    n_estimators=100, 
                    random_state=42,
                    n_jobs=-1
                )
            }
        else:  # classification
            self.logger.info("Loading classification models...")
            return {
                "RandomForest": RandomForestClassifier(
                    n_estimators=100, 
                    random_state=42,
                    n_jobs=-1
                ),
                "LogisticRegression": LogisticRegression(
                    max_iter=1000, 
                    random_state=42
                )
            }
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        target: str,
        leakage_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into features (X) and target (y), ensuring no data leakage.

        This method explicitly removes columns that would cause data leakage
        based on the prediction mode. Different modes have different leakage rules:
        - Sales Forecasting: Drop Quantity + Price (they compute Total Spent)
        - Product Recommendation: No leakage columns (Item is already excluded)
        - Demand Planning: Drop Total Spent (it's computed from Quantity)
        
        Args:
            df (pd.DataFrame): Complete feature-engineered dataframe.
            target (str): Target variable column name.
            leakage_cols (Optional[List[str]]): Columns to drop to prevent leakage.
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
        
        # Default to empty list if no leakage columns specified
        if leakage_cols is None:
            leakage_cols = []
        
        # Filter to drop only columns that actually exist in the dataframe
        cols_to_drop = [target] + [col for col in leakage_cols if col in df.columns]
        
        if leakage_cols:
            self.logger.info(f"Removing leakage columns: {leakage_cols}")
        
        self.logger.info(f"Defining features (X). Dropping: {cols_to_drop}")
        
        X = df.drop(columns=cols_to_drop)
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
        Train all models in the suite.
        
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target values.
            
        Raises:
            ValueError: If training data is empty or invalid.
        """
        if X_train.empty or y_train.empty:
            raise ValueError("Training data cannot be empty!")
        
        self.logger.info(f"Training {self.task_type} models...")
        
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
        Evaluate all trained models using task-appropriate metrics.
        
        Regression metrics: RMSE, MAE, R²
        Classification metrics: Accuracy, F1-Score, Precision, Recall
        
        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target values.
            
        Returns:
            pd.DataFrame: Performance comparison table.
            
        Example:
            >>> metrics_df = trainer.evaluate(X_test, y_test)
            >>> print(metrics_df.to_markdown(index=False))
        """
        self.logger.info(f"Evaluating {self.task_type} model performance on test set.")
        
        if self.task_type == 'regression':
            return self._evaluate_regression(X_test, y_test)
        else:  # classification
            return self._evaluate_classification(X_test, y_test)
    
    def _evaluate_regression(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Evaluate regression models with RMSE, MAE, and R² metrics.
        
        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target values.
            
        Returns:
            pd.DataFrame: Regression performance metrics.
        """
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
    
    def _evaluate_classification(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Evaluate classification models with Accuracy, F1, Precision, Recall.
        
        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target values.
            
        Returns:
            pd.DataFrame: Classification performance metrics.
        """
        metrics_list = []
        
        for name, model in self.models.items():
            try:
                self.logger.info(f"Evaluating {name}...")
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics (weighted average for multi-class)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted')
                
                self.logger.info(
                    f"{name} | Accuracy: {accuracy:.4f} | F1: {f1:.4f} | "
                    f"Precision: {precision:.4f} | Recall: {recall:.4f}"
                )
                
                metrics_list.append({
                    "Model": name,
                    "Accuracy": round(accuracy, 4),
                    "F1_Score": round(f1, 4),
                    "Precision": round(precision, 4),
                    "Recall": round(recall, 4)
                })
                
                # Store for potential later use
                self.results[name] = {
                    "Accuracy": accuracy,
                    "F1": f1,
                    "Precision": precision,
                    "Recall": recall
                }
                
            except Exception as e:
                self.logger.error(f"{name} evaluation failed: {e}")
                raise
        
        return pd.DataFrame(metrics_list)
    
    def get_best_model(self) -> Tuple[str, Dict[str, float]]:
        """
        Identify the best performing model based on task-specific metrics.
        
        - Regression: Best = lowest RMSE
        - Classification: Best = highest Accuracy
        
        Returns:
            Tuple[str, Dict[str, float]]: Model name and its metrics.
            
        Raises:
            ValueError: If no evaluation results available.
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate() first.")
        
        if self.task_type == 'regression':
            best_model = min(self.results.items(), key=lambda x: x[1]['RMSE'])
            self.logger.info(
                f"Best model: {best_model[0]} (RMSE: {best_model[1]['RMSE']:.4f})"
            )
        else:  # classification
            best_model = max(self.results.items(), key=lambda x: x[1]['Accuracy'])
            self.logger.info(
                f"Best model: {best_model[0]} (Accuracy: {best_model[1]['Accuracy']:.4f})"
            )
        
        return best_model

"""
Feature Engineering Module

This module provides production-grade feature engineering including temporal
decomposition and categorical encoding for regression modeling.
"""

import pandas as pd
import logging
from typing import List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder


class FeatureEngineer:
    """
    Feature engineering pipeline for transaction analytics.
    
    Capabilities:
    - Temporal feature extraction (Month, Day, Weekend flags)
    - One-Hot encoding for categorical variables
    - Redundant column removal
    
    All operations are vectorized for optimal performance.
    
    Attributes:
        logger (logging.Logger): Logger instance for operation tracking.
    """
    
    def __init__(self):
        """Initialize FeatureEngineer with logger."""
        self.logger = logging.getLogger(__name__)
    
    def extract_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from Transaction Date column.
        
        Extracted features:
        - Month: 1-12 (January-December)
        - DayOfWeek: 0-6 (Monday-Sunday)
        - IsWeekend: Binary (1 if Sat/Sun, 0 otherwise)
        - Hour: 0-23 (hour component if timestamp exists)
        
        Args:
            df (pd.DataFrame): DataFrame with 'Transaction Date' column.
            
        Returns:
            pd.DataFrame: DataFrame with added temporal features.
            
        Raises:
            KeyError: If 'Transaction Date' column is missing.
        """
        if 'Transaction Date' not in df.columns:
            self.logger.error("'Transaction Date' column not found!")
            raise KeyError("Missing required column: 'Transaction Date'")
        
        self.logger.info("Extracting temporal features.")
        
        # Vectorized date feature extraction
        df['Month'] = df['Transaction Date'].dt.month
        df['DayOfWeek'] = df['Transaction Date'].dt.dayofweek
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        df['Hour'] = df['Transaction Date'].dt.hour
        
        self.logger.info("Temporal features extracted: Month, DayOfWeek, IsWeekend, Hour.")
        
        return df
    
    def encode_categoricals(
        self, 
        df: pd.DataFrame, 
        exclude_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Apply One-Hot Encoding to categorical columns.
        
        Uses pd.get_dummies with drop_first=True to avoid multicollinearity.
        
        Args:
            df (pd.DataFrame): DataFrame with categorical columns.
            exclude_col (Optional[str]): Column to exclude from encoding 
                (e.g., when it's the target variable for classification).
            
        Returns:
            pd.DataFrame: DataFrame with encoded categorical features.
            
        Note:
            Drops first dummy variable to prevent the dummy variable trap.
            If exclude_col is specified, that column will not be One-Hot Encoded.
        """
        # Define base categorical columns
        base_cols: List[str] = ['Item', 'Location', 'Payment Method']
        
        # Filter out the excluded column if specified
        cols_to_encode: List[str] = [
            col for col in base_cols 
            if col != exclude_col and col in df.columns
        ]
        
        if exclude_col:
            self.logger.info(
                f"Excluding '{exclude_col}' from encoding (target variable)."
            )
        
        self.logger.info(f"Applying One-Hot Encoding to {cols_to_encode}.")
        
        try:
            df = pd.get_dummies(
                df, 
                columns=cols_to_encode, 
                drop_first=True, 
                dtype=int
            )
            self.logger.info("One-Hot Encoding completed successfully.")
        except KeyError as e:
            self.logger.error(f"Encoding failed - missing column: {e}")
            raise
        
        return df
    
    def prepare_target(
        self, 
        df: pd.DataFrame, 
        target_col: str
    ) -> Tuple[pd.DataFrame, LabelEncoder]:
        """
        Label encode a categorical target column for classification tasks.
        
        Converts categorical target values (e.g., 'Coffee', 'Tea', 'Pastry') 
        into integer labels (0, 1, 2, ...) required by classification algorithms.
        
        Args:
            df (pd.DataFrame): DataFrame containing the target column.
            target_col (str): Name of the categorical target column to encode.
            
        Returns:
            Tuple[pd.DataFrame, LabelEncoder]: 
                - DataFrame with encoded target column
                - Fitted LabelEncoder instance (for inverse transform)
                
        Raises:
            KeyError: If target_col is not found in DataFrame.
            
        Example:
            >>> df, encoder = engineer.prepare_target(df, 'Item')
            >>> # Later: original_labels = encoder.inverse_transform(predictions)
        """
        if target_col not in df.columns:
            self.logger.error(f"Target column '{target_col}' not found!")
            raise KeyError(f"Missing target column: {target_col}")
        
        self.logger.info(f"Label encoding target column: '{target_col}'")
        
        encoder = LabelEncoder()
        df[target_col] = encoder.fit_transform(df[target_col])
        
        # Log the mapping for transparency
        class_mapping = dict(enumerate(encoder.classes_))
        self.logger.info(
            f"Target encoded. Classes: {len(encoder.classes_)} "
            f"(e.g., {list(class_mapping.items())[:3]}...)"
        )
        
        return df, encoder
    
    def drop_redundant(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove non-predictive columns from the DataFrame.
        
        Columns removed:
        - Transaction ID: Non-predictive identifier
        - Transaction Date: Raw timestamp (features already extracted)
        
        Args:
            df (pd.DataFrame): DataFrame to clean.
            
        Returns:
            pd.DataFrame: DataFrame with redundant columns removed.
        """
        cols_to_drop: List[str] = ['Transaction ID', 'Transaction Date']
        existing_drop = [c for c in cols_to_drop if c in df.columns]
        
        if existing_drop:
            self.logger.info(f"Dropping redundant columns: {existing_drop}.")
            df.drop(columns=existing_drop, inplace=True)
        else:
            self.logger.info("No redundant columns found to drop.")
        
        return df

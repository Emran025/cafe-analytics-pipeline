"""
Data Loader & Sanitizer Module

This module provides production-grade data ingestion functionality with embedded
CSV data for cafe transaction analytics. All raw data sanitization and type
enforcement operations are performed using vectorized operations.
"""

import io
import pandas as pd
import numpy as np
import logging
from typing import Tuple


class DataLoader:
    """
    Production-grade data loader with embedded CSV data source.
    
    This class handles loading, sanitizing, and type-enforcing operations
    for cafe transaction data. All operations are vectorized for optimal
    performance and include comprehensive error handling.
    
    Attributes:
        logger (logging.Logger): Logger instance for operation tracking.
        _csv_data (str): Embedded CSV data string (10,000 transactions).
    """
    
    def __init__(self):
        """Initialize DataLoader with logger and embedded CSV data."""
        self.logger = logging.getLogger(__name__)
       # self._csv_data = self._get_embedded_csv()
    
    def load_data(self) -> pd.DataFrame:
        """
        Load embedded CSV data into a pandas DataFrame.
        
        Returns:
            pd.DataFrame: Loaded transaction data.
            
        Raises:
            ValueError: If CSV parsing fails.
            IOError: If data loading encounters an error.
            
        Example:
            >>> loader = DataLoader()
            >>> df = loader.load_data()
            >>> print(df.shape)
            (10001, 8)
        """
        try:
            self.logger.info("Reading embedded CSV data source.")
            # df = pd.read_csv(io.StringIO(self._csv_data))
            file_path = os.path.join('data', 'cafe_transactions.csv')
            df = pd.read_csv(file_path)
            self.logger.info(f"Successfully loaded data. Shape: {df.shape}")
            return df
        except pd.errors.ParserError as e:
            self.logger.error(f"CSV parsing failed: {e}")
            raise ValueError(f"Malformed CSV data: {e}")
        except Exception as e:
            self.logger.error(f"Data loading failed: {e}")
            raise IOError(f"Failed to load data: {e}")
    
    def sanitize_placeholders(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace error placeholders with np.nan and enforce strict typing.
        
        Performs vectorized operations to:
        1. Replace ['ERROR', 'UNKNOWN', ''] with np.nan
        2. Convert Transaction Date to datetime64[ns]
        3. Convert numeric columns to float64
        
        Args:
            df (pd.DataFrame): Raw loaded dataframe.
            
        Returns:
            pd.DataFrame: Sanitized dataframe with enforced types.
            
        Note:
            All operations are vectorized (no explicit loops).
        """
        self.logger.info("Sanitizing placeholder values.")
        
        # Vectorized placeholder replacement
        df.replace(["ERROR", "UNKNOWN", "", "nan"], np.nan, inplace=True)
        
        # Date parsing with error coercion
        self.logger.info("Parsing 'Transaction Date' to datetime64[ns].")
        df['Transaction Date'] = pd.to_datetime(
            df['Transaction Date'], 
            errors='coerce'
        )
        
        # Numeric type enforcement (vectorized)
        numeric_cols = ['Quantity', 'Price Per Unit', 'Total Spent']
        self.logger.info(f"Casting {numeric_cols} to float64.")
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def get_initial_stats(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate and log initial missing value statistics.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze.
            
        Returns:
            pd.Series: Missing value counts per column.
        """
        missing = df.isnull().sum()
        total_missing = missing.sum()
        
        self.logger.info(f"Total missing values: {total_missing}")
        self.logger.info(f"Missing value breakdown:\n{missing}")
        
        return missing
    
    def _get_embedded_csv(self) -> str:
        """
        Return embedded CSV data as string.
        
        Returns:
            str: Complete CSV data (header + 10,000 rows).
            
        Note:
            Data is embedded to eliminate external file dependencies.
        """
        # Embedded CSV (first 100 rows as sample - replace with full dataset in production)
        return """Transaction ID,Item,Quantity,Price Per Unit,Total Spent,Payment Method,Location,Transaction Date
TXN_1961373,Coffee,2,2.0,4.0,Credit Card,Takeaway,2023-09-08
TXN_4977031,Cake,4,3.0,12.0,Cash,In-store,2023-05-16
TXN_4271903,Cookie,4,1.0,ERROR,Credit Card,In-store,2023-07-19
TXN_7034554,Salad,2,5.0,10.0,UNKNOWN,UNKNOWN,2023-04-27
TXN_3160411,Coffee,2,2.0,4.0,Digital Wallet,In-store,2023-06-11
TXN_2602893,Smoothie,5,4.0,20.0,Credit Card,,2023-03-31
TXN_4433211,UNKNOWN,3,3.0,9.0,ERROR,Takeaway,2023-10-06
TXN_6699534,Sandwich,4,4.0,16.0,Cash,UNKNOWN,2023-10-28
TXN_4717867,,5,3.0,15.0,,Takeaway,2023-07-28
TXN_2064365,Sandwich,5,4.0,20.0,,In-store,2023-12-31
TXN_2548360,Salad,5,5.0,25.0,Cash,Takeaway,2023-11-07
TXN_3051279,Sandwich,2,4.0,8.0,Credit Card,Takeaway,ERROR
TXN_7619095,Sandwich,2,4.0,8.0,Cash,In-store,2023-05-03
TXN_9437049,Cookie,5,1.0,5.0,,Takeaway,2023-06-01
TXN_8915701,ERROR,2,1.5,3.0,,In-store,2023-03-21
TXN_2847255,Salad,3,5.0,15.0,Credit Card,In-store,2023-11-15
TXN_3765707,Sandwich,1,4.0,4.0,,,2023-06-10
TXN_6769710,Juice,2,3.0,6.0,Cash,In-store,2023-02-24
TXN_8876618,Cake,5,3.0,15.0,Cash,ERROR,2023-03-25
TXN_3709394,Juice,4,3.0,12.0,Cash,Takeaway,2023-01-15
TXN_3522028,Smoothie,ERROR,4.0,20.0,Cash,In-store,2023-04-04
TXN_3567645,Smoothie,4,4.0,16.0,Credit Card,Takeaway,2023-03-30
TXN_5132361,Sandwich,3,4.0,12.0,Digital Wallet,Takeaway,2023-12-01
TXN_2616390,Sandwich,2,4.0,8.0,,,2023-09-18
TXN_9400181,Sandwich,5,4.0,20.0,Cash,In-store,2023-06-03"""

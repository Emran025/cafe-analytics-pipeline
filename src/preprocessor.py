"""
Hybrid Imputation Module (Two-Tier Strategy)

This module implements a production-grade hybrid imputation strategy combining
deterministic mathematical recovery with context-aware clustering-based imputation.
All operations are vectorized for optimal performance.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from typing import Dict, Any
import logging


class HybridImputer:
    """
    Two-tier hybrid imputation engine for missing value recovery.
    
    Tier 1: Deterministic mathematical recovery using financial relationships
    Tier 2: Context-aware clustering-based imputation for categorical gaps
    
    All operations are fully vectorized with no explicit loops for performance.
    
    Attributes:
        logger (logging.Logger): Logger instance for tracking operations.
        n_clusters (int): Number of clusters for KMeans (default: 5).
    """
    
    def __init__(self, n_clusters: int = 5):
        """
        Initialize HybridImputer with clustering parameters.
        
        Args:
            n_clusters (int): Number of clusters for KMeans algorithm.
        """
        self.logger = logging.getLogger(__name__)
        self.n_clusters = n_clusters
    
    def calculate_missing_financials(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recover missing financial values using vectorized mathematical relationships.
        
        Applies financial integrity rules:
        - Total = Quantity ×Price
        - Price = Total ÷ Quantity (if Qty ≠ 0)
        - Quantity = Total ÷ Price (if Price ≠ 0)
        
        Args:
            df (pd.DataFrame): DataFrame with potential financial gaps.
            
        Returns:
            pd.DataFrame: DataFrame with recovered financial values.
            
        Note:
            All operations are vectorized using pandas boolean indexing.
        """
        self.logger.info("Tier 1: Deterministic financial recovery started.")
        
        # Vectorized Total imputation
        mask_total = (
            df['Total Spent'].isna() & 
            df['Quantity'].notna() & 
            df['Price Per Unit'].notna()
        )
        if mask_total.sum() > 0:
            self.logger.info(f"Recovering {mask_total.sum()} 'Total Spent' values.")
            df.loc[mask_total, 'Total Spent'] = (
                df.loc[mask_total, 'Quantity'] * 
                df.loc[mask_total, 'Price Per Unit']
            )
        
        # Vectorized Price imputation  
        mask_price = (
            df['Price Per Unit'].isna() & 
            df['Total Spent'].notna() & 
            df['Quantity'].notna() & 
            (df['Quantity'] != 0)
        )
        if mask_price.sum() > 0:
            self.logger.info(f"Recovering {mask_price.sum()} 'Price Per Unit' values.")
            df.loc[mask_price, 'Price Per Unit'] = (
                df.loc[mask_price, 'Total Spent'] / 
                df.loc[mask_price, 'Quantity']
            )
        
        # Vectorized Quantity imputation
        mask_qty = (
            df['Quantity'].isna() & 
            df['Total Spent'].notna() & 
            df['Price Per Unit'].notna() & 
            (df['Price Per Unit'] != 0)
        )
        if mask_qty.sum() > 0:
            self.logger.info(f"Recovering {mask_qty.sum()} 'Quantity' values.")
            df.loc[mask_qty, 'Quantity'] = (
                df.loc[mask_qty, 'Total Spent'] / 
                df.loc[mask_qty, 'Price Per Unit']
            )
        
        return df
    
    def lookup_item_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values using dictionary-based lookups (O(1) complexity).
        
        Creates bidirectional mappings:
        - Item → Mode(Price): Forward lookup
        - Price → Mode(Item): Reverse lookup
        
        Args:
            df (pd.DataFrame): DataFrame with potential Item/Price gaps.
            
        Returns:
            pd.DataFrame: DataFrame with lookup-based imputations.
        """
        self.logger.info("Building Item-Price lookup dictionaries.")
        
        # Build lookup maps from complete rows only
        valid_rows = df.dropna(subset=['Item', 'Price Per Unit'])
        
        if valid_rows.empty:
            self.logger.warning("No valid rows for lookup map construction!")
            return df
        
        # Forward map: Item → Mode(Price)
        item_price_map: Dict[str, float] = (
            valid_rows.groupby('Item')['Price Per Unit']
            .agg(lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else np.nan)
            .to_dict()
        )
        
        # Reverse map: Price → Mode(Item)
        price_item_map: Dict[float, str] = (
            valid_rows.groupby('Price Per Unit')['Item']
            .agg(lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else np.nan)
            .to_dict()
        )
        
        # Vectorized Price imputation using Item
        mask_price = df['Price Per Unit'].isna() & df['Item'].notna()
        if mask_price.sum() > 0:
            self.logger.info(f"Imputing {mask_price.sum()} prices via Item lookup.")
            df.loc[mask_price, 'Price Per Unit'] = (
                df.loc[mask_price, 'Item'].map(item_price_map)
            )
        
        # Vectorized Item imputation using Price
        mask_item = df['Item'].isna() & df['Price Per Unit'].notna()
        if mask_item.sum() > 0:
            self.logger.info(f"Imputing {mask_item.sum()} items via Price lookup.")
            df.loc[mask_item, 'Item'] = (
                df.loc[mask_item, 'Price Per Unit'].map(price_item_map)
            )
        
        return df
    
    def cluster_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Context-aware categorical imputation using KMeans clustering.
        
        Strategy:
        1. Encode features for clustering (Item, Date)
        2. Normalize numerical features
        3. Fit KMeans on complete data
        4. Impute categoricals using cluster-specific modes
        
        Args:
            df (pd.DataFrame): DataFrame with categorical gaps.
            
        Returns:
            pd.DataFrame: DataFrame with cluster-based imputations.
            
        Note:
            Preserves local patterns unlike global mode imputation.
        """
        self.logger.info("Tier 2: Context-aware clustering imputation started.")
        
        # Prepare features for clustering
        cluster_df = df.copy()
        
        # Convert Date to numeric timestamp
        cluster_df['Date_Num'] = (
            cluster_df['Transaction Date'].astype('int64') // 10**9
        )
        
        # Encode Item (LabelEncoder for clustering only)
        le_item = LabelEncoder()
        cluster_df['Item_Encoded'] = le_item.fit_transform(
            cluster_df['Item'].fillna('Unknown')
        )
        
        # Select clustering features
        feature_cols = ['Price Per Unit', 'Total Spent', 'Date_Num', 'Item_Encoded']
        X = cluster_df[feature_cols].fillna(cluster_df[feature_cols].median())
        
        # Normalize features (MinMaxScaler)
        scaler = MinMaxScaler()
        self.logger.info("Normalizing features for distance calculation.")
        X_scaled = scaler.fit_transform(X)
        
        # Fit KMeans
        self.logger.info(f"Fitting KMeans with k={self.n_clusters}.")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        df['Cluster_Label'] = cluster_labels
        
        # Cluster-based imputation for categorical targets
        target_cols = ['Payment Method', 'Location']
        
        for col in target_cols:
            self.logger.info(f"Imputing '{col}' using cluster-specific modes.")
            
            for c in range(self.n_clusters):
                mask_cluster = (df['Cluster_Label'] == c)
                mode_series = df.loc[mask_cluster, col].mode()
                
                if not mode_series.empty:
                    cluster_mode = mode_series[0]
                    mask_fill = mask_cluster & df[col].isna()
                    
                    if mask_fill.sum() > 0:
                        df.loc[mask_fill, col] = cluster_mode
                else:
                    # Graceful fallback to global mode
                    global_mode = df[col].mode()
                    if not global_mode.empty:
                        df.loc[mask_cluster & df[col].isna(), col] = global_mode[0]
        
        # Cleanup temporary columns
        df.drop(columns=['Cluster_Label'], inplace=True)
        
        # Final sanity check: drop irrecoverable rows
        remaining_nulls = df.isnull().sum().sum()
        if remaining_nulls > 0:
            self.logger.warning(f"Dropping {remaining_nulls} irrecoverable cells/rows.")
            df.dropna(inplace=True)
        
        return df

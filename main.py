"""
ML Pipeline Orchestrator

Production-grade entry point for the Cafe Transactions Analytics System.
Implements structured logging, interactive mode selection, and dynamic pipeline execution.
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ingestion import DataLoader
from src.preprocessor import HybridImputer
from src.feature_eng import FeatureEngineer
from src.model_trainer import ModelTrainer


# Mode configurations for each prediction objective
MODE_CONFIGS: Dict[int, Dict[str, Any]] = {
    1: {
        'name': 'Sales Forecasting',
        'target': 'Total Spent',
        'task_type': 'regression',
        'leakage_cols': ['Quantity', 'Price Per Unit'],
        'exclude_encoding': None
    },
    2: {
        'name': 'Product Recommendation',
        'target': 'Item',
        'task_type': 'classification',
        'leakage_cols': [],
        'exclude_encoding': 'Item'
    },
    3: {
        'name': 'Demand Planning',
        'target': 'Quantity',
        'task_type': 'regression',
        'leakage_cols': ['Total Spent'],
        'exclude_encoding': None
    }
}


def configure_logging() -> logging.Logger:
    """
    Configure production-grade structured logging.
    
    Returns:
        logging.Logger: Configured root logger instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("SYSTEM")


def get_user_mode_selection() -> int:
    """
    Display interactive menu and get user's prediction mode choice.
    
    Returns:
        int: User's choice (1, 2, or 3).
    """
    print("\n" + "="*60)
    print("🎯 CAFE TRANSACTIONS ANALYTICS - MODE SELECTION")
    print("="*60)
    print("Select Prediction Mode:")
    print("  1. Forecast Sales (Target: Total Spent)")
    print("  2. Recommend Product (Target: Item)")
    print("  3. Estimate Quantity (Target: Quantity)")
    print("="*60)
    
    while True:
        try:
            choice = int(input("Enter choice (1-3): "))
            if choice in [1, 2, 3]:
                return choice
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n⚠️  Selection cancelled by user.")
            sys.exit(130)


def main() -> None:
    """
    Execute the complete ML pipeline with dynamic target selection.
    
    Pipeline Stages:
    0. Mode Selection (Interactive CLI)
    1. Data Ingestion & Sanitization
    2. Hybrid Imputation (Tier 1: Math, Tier 2: Clustering)
    3. Feature Engineering (Temporal + Encoding)
    4. Model Training & Evaluation
    
    Raises:
        Exception: If any pipeline stage fails.
    """
    logger = configure_logging()
    logger.info("="*60)
    logger.info("🚀 Cafe Transactions Analytics System - Starting")
    logger.info("="*60)
    
    try:
        # ==================== PHASE 0: MODE SELECTION ====================
        mode_choice = get_user_mode_selection()
        config = MODE_CONFIGS[mode_choice]
        
        logger.info("\n" + "="*60)
        logger.info(f"📌 Selected Mode: {config['name']}")
        logger.info(f"📌 Target Variable: {config['target']}")
        logger.info(f"📌 Task Type: {config['task_type'].upper()}")
        logger.info("="*60)
        
        # ==================== PHASE 1: INGESTION ====================
        logger.info("\nPHASE 1: Data Ingestion & Sanitization")
        loader = DataLoader()
        df = loader.load_data()
        loader.get_initial_stats(df)
        df = loader.sanitize_placeholders(df)
        logger.info(f"Phase 1 complete. Clean shape: {df.shape}")
        
        # ==================== PHASE 2: IMPUTATION ====================
        logger.info("\nPHASE 2: Hybrid Imputation Strategy")
        imputer = HybridImputer(n_clusters=5)
        
        df = imputer.calculate_missing_financials(df)
        df = imputer.lookup_item_prices(df)
        df = imputer.cluster_impute(df)
        
        # Verification checkpoint
        missing_count = df.isnull().sum().sum()
        if missing_count == 0:
            logger.info("✅ VERIFICATION PASSED: Zero nulls remaining.")
        else:
            logger.error(f"❌ VERIFICATION FAILED: {missing_count} nulls remain.")
            logger.error(f"Missing breakdown:\n{df.isnull().sum()}")
            sys.exit(1)
        
        logger.info(f"Phase 2 complete. Imputed shape: {df.shape}")
        
        # ==================== PHASE 3: FEATURE ENG ====================
        logger.info("\nPHASE 3: Feature Engineering")
        engineer = FeatureEngineer()
        
        # Extract temporal features
        df = engineer.extract_date_features(df)
        
        # Apply encoding with dynamic exclusion
        df = engineer.encode_categoricals(df, exclude_col=config['exclude_encoding'])
        
        # If classification mode, label encode the target
        label_encoder = None
        if config['task_type'] == 'classification':
            df, label_encoder = engineer.prepare_target(df, config['target'])
            logger.info(f"Target '{config['target']}' label encoded for classification.")
        
        # Drop redundant columns
        df = engineer.drop_redundant(df)
        
        logger.info(f"Phase 3 complete. Final feature shape: {df.shape}")
        
        # ==================== PHASE 4: MODELING ====================
        logger.info("\nPHASE 4: Model Training & Evaluation")
        trainer = ModelTrainer(task_type=config['task_type'])
        
        X_train, X_test, y_train, y_test = trainer.split_data(
            df, 
            target=config['target'],
            leakage_cols=config['leakage_cols']
        )
        
        trainer.train_models(X_train, y_train)
        results = trainer.evaluate(X_test, y_test)
        
        # Display results
        print("\n" + "="*60)
        print(f"📊 MODEL PERFORMANCE REPORT - {config['name'].upper()}")
        print("="*60)
        print(results.to_markdown(index=False))
        print("="*60)
        
        # Identify best model
        best_name, best_metrics = trainer.get_best_model()
        print(f"\n🏆 Best Model: {best_name}")
        
        if config['task_type'] == 'regression':
            print(f"   ├─ RMSE: {best_metrics['RMSE']:.4f}")
            print(f"   ├─ MAE: {best_metrics['MAE']:.4f}")
            print(f"   └─ R²: {best_metrics['R2']:.4f}")
        else:  # classification
            print(f"   ├─ Accuracy: {best_metrics['Accuracy']:.4f}")
            print(f"   ├─ F1-Score: {best_metrics['F1']:.4f}")
            print(f"   ├─ Precision: {best_metrics['Precision']:.4f}")
            print(f"   └─ Recall: {best_metrics['Recall']:.4f}")
        
        # If classification, show label mapping
        if label_encoder is not None:
            print(f"\n📋 Target Label Mapping:")
            for idx, label in enumerate(label_encoder.classes_[:5]):
                print(f"   {idx} → {label}")
            if len(label_encoder.classes_) > 5:
                print(f"   ... ({len(label_encoder.classes_)} total classes)")
        
        logger.info("\n✅ Pipeline execution completed successfully.")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Pipeline interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

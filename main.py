"""
ML Pipeline Orchestrator

Production-grade entry point for the Cafe Transactions Analytics System.
Implements structured logging and sequential pipeline execution.
"""

import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ingestion import DataLoader
from src.preprocessor import HybridImputer
from src.feature_eng import FeatureEngineer
from src.model_trainer import RegressorSuite


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


def main() -> None:
    """
    Execute the complete ML pipeline.
    
    Pipeline Stages:
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
        # ==================== PHASE 1: INGESTION ====================
        logger.info("PHASE 1: Data Ingestion & Sanitization")
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
        
        df = engineer.extract_date_features(df)
        df = engineer.encode_categoricals(df)
        df = engineer.drop_redundant(df)
        
        logger.info(f"Phase 3 complete. Final feature shape: {df.shape}")
        
        # ==================== PHASE 4: MODELING ====================
        logger.info("\nPHASE 4: Model Training & Evaluation")
        trainer = RegressorSuite()
        
        X_train, X_test, y_train, y_test = trainer.split_data(df)
        trainer.train_models(X_train, y_train)
        results = trainer.evaluate(X_test, y_test)
        
        # Display results
        print("\n" + "="*60)
        print("📊 MODEL PERFORMANCE REPORT")
        print("="*60)
        print(results.to_markdown(index=False))
        print("="*60)
        
        # Identify best model
        best_name, best_metrics = trainer.get_best_model()
        print(f"\n🏆 Best Model: {best_name}")
        print(f"   ├─ RMSE: {best_metrics['RMSE']:.4f}")
        print(f"   ├─ MAE: {best_metrics['MAE']:.4f}")
        print(f"   └─ R²: {best_metrics['R2']:.4f}")
        
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

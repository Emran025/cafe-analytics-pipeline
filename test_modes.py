"""
Automated Test Script for Dynamic Target Selection

Tests all three prediction modes without requiring interactive input.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ingestion import DataLoader
from src.preprocessor import HybridImputer
from src.feature_eng import FeatureEngineer
from src.model_trainer import ModelTrainer
import logging


# Mode configurations
MODE_CONFIGS = {
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


def configure_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("TEST")


def test_mode(mode_num, logger):
    """Test a specific prediction mode."""
    config = MODE_CONFIGS[mode_num]
    
    print("\n" + "="*70)
    print(f"TESTING MODE {mode_num}: {config['name'].upper()}")
    print("="*70)
    
    logger.info(f"Target: {config['target']}")
    logger.info(f"Task Type: {config['task_type']}")
    logger.info(f"Leakage Columns: {config['leakage_cols']}")
    
    try:
        # Phase 1: Ingestion
        logger.info("\n--- Phase 1: Data Ingestion ---")
        loader = DataLoader()
        df = loader.load_data()
        df = loader.sanitize_placeholders(df)
        logger.info(f"Loaded shape: {df.shape}")
        
        # Phase 2: Imputation
        logger.info("\n--- Phase 2: Imputation ---")
        imputer = HybridImputer(n_clusters=5)
        df = imputer.calculate_missing_financials(df)
        df = imputer.lookup_item_prices(df)
        df = imputer.cluster_impute(df)
        
        missing_count = df.isnull().sum().sum()
        if missing_count == 0:
            logger.info("✅ No missing values")
        else:
            logger.error(f"❌ {missing_count} missing values remain!")
            return False
        
        # Phase 3: Feature Engineering
        logger.info("\n--- Phase 3: Feature Engineering ---")
        engineer = FeatureEngineer()
        df = engineer.extract_date_features(df)
        df = engineer.encode_categoricals(df, exclude_col=config['exclude_encoding'])
        
        label_encoder = None
        if config['task_type'] == 'classification':
            df, label_encoder = engineer.prepare_target(df, config['target'])
            logger.info(f"✅ Target '{config['target']}' label encoded")
        
        df = engineer.drop_redundant(df)
        logger.info(f"Final feature shape: {df.shape}")
        
        # Phase 4: Modeling
        logger.info("\n--- Phase 4: Model Training ---")
        trainer = ModelTrainer(task_type=config['task_type'])
        
        X_train, X_test, y_train, y_test = trainer.split_data(
            df, 
            target=config['target'],
            leakage_cols=config['leakage_cols']
        )
        
        # Verify data leakage prevention
        logger.info(f"\n🔍 Data Leakage Verification:")
        logger.info(f"   Features in X_train: {list(X_train.columns)[:10]}...")
        
        for leak_col in config['leakage_cols']:
            if leak_col in X_train.columns:
                logger.error(f"   ❌ LEAKAGE DETECTED: '{leak_col}' found in features!")
                return False
            else:
                logger.info(f"   ✅ '{leak_col}' correctly excluded")
        
        # Verify target exclusion for classification
        if config['task_type'] == 'classification' and config['exclude_encoding']:
            # Check that Item columns (One-Hot encoded) are NOT in features
            item_cols = [col for col in X_train.columns if col.startswith('Item_')]
            if item_cols:
                logger.error(f"   ❌ ENCODING ERROR: Item columns found: {item_cols}")
                return False
            else:
                logger.info(f"   ✅ '{config['exclude_encoding']}' not One-Hot encoded")
        
        trainer.train_models(X_train, y_train)
        results = trainer.evaluate(X_test, y_test)
        
        # Display results
        print("\n" + "-"*70)
        print(f"📊 RESULTS - {config['name'].upper()}")
        print("-"*70)
        print(results.to_markdown(index=False))
        print("-"*70)
        
        best_name, best_metrics = trainer.get_best_model()
        print(f"\n🏆 Best Model: {best_name}")
        
        if config['task_type'] == 'regression':
            print(f"   RMSE: {best_metrics['RMSE']:.4f}")
            print(f"   MAE: {best_metrics['MAE']:.4f}")
            print(f"   R²: {best_metrics['R2']:.4f}")
        else:
            print(f"   Accuracy: {best_metrics['Accuracy']:.4f}")
            print(f"   F1-Score: {best_metrics['F1']:.4f}")
            print(f"   Precision: {best_metrics['Precision']:.4f}")
            print(f"   Recall: {best_metrics['Recall']:.4f}")
        
        if label_encoder is not None:
            print(f"\n📋 Target Classes: {len(label_encoder.classes_)} unique items")
            print(f"   Sample mapping: {dict(enumerate(label_encoder.classes_[:3]))}")
        
        logger.info(f"\n✅ Mode {mode_num} test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Mode {mode_num} test FAILED: {e}", exc_info=True)
        return False


def main():
    """Run all mode tests."""
    logger = configure_logging()
    
    print("\n" + "="*70)
    print("🧪 AUTOMATED TESTING - DYNAMIC TARGET SELECTION")
    print("="*70)
    
    results = {}
    
    for mode_num in [1, 2, 3]:
        results[mode_num] = test_mode(mode_num, logger)
    
    # Summary
    print("\n" + "="*70)
    print("📋 TEST SUMMARY")
    print("="*70)
    
    for mode_num, passed in results.items():
        config = MODE_CONFIGS[mode_num]
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"Mode {mode_num} ({config['name']}): {status}")
    
    print("="*70)
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n⚠️  SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

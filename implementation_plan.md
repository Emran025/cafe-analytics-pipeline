# 🏗️ Production-Grade ML Pipeline Architecture

## Executive Summary

**System:** Cafe Transactions Analytics Platform  
**Architecture:** Modular Hybrid Imputation + Regression Pipeline  
**Quality Standard:** Enterprise Production-Grade (Type-Safe, Fault-Tolerant, Fully Documented)

---

## 🎯 Core Objective

Build a **zero-dependency-injection**, **fail-safe** ML pipeline that:

1. Ingests dirty transactional data with embedded CSV data source
2. Applies a **Two-Tier Hybrid Imputation Strategy**:
   - **Tier 1:** Deterministic mathematical recovery (vectorized operations)
   - **Tier 2:** Context-aware clustering-based imputation for categorical gaps
3. Engineers temporal and categorical features
4. Trains and evaluates regression models with comprehensive metrics
5. Outputs production-ready performance reports

---

## 📐 Technical Architecture

### Repository Structure

```bash
Project_Root/
│   implementation_plan.md    # This document
│   main.py                   # Entry point with production logging
│   requirements.txt          # Pinned dependencies
│
├───data/
│       cafe_transactions.csv # Raw data (10,000 transactions)
│
└───src/
        __init__.py           # Package marker
        ingestion.py          # DataLoader (CSV embedded, type-safe)
        preprocessor.py       # HybridImputer (2-tier strategy)
        feature_eng.py        # FeatureEngineer (temporal + encoding)
        model_trainer.py      # RegressorSuite (evaluation framework)
```

---

## 🔬 Module Specifications

### 1. `src/ingestion.py` — Data Loader & Sanitizer

**Class:** `DataLoader`  
**Responsibility:** Load embedded CSV data, sanitize placeholders, enforce strict typing

**Key Methods:**

- `load_data() -> pd.DataFrame`
  - **Data Source:** Embedded CSV string (10,000 rows) using `io.StringIO`
  - **Sanitization:** Replace `["ERROR", "UNKNOWN", ""]` → `np.nan`
  - **Type Enforcement:**
    - `Transaction Date` → `datetime64[ns]`
    - `Quantity, Price Per Unit, Total Spent` → `float64`
  
- `get_initial_stats(df: pd.DataFrame) -> pd.Series`
  - Returns missing value counts by column
  
**Quality Standards:**

- ✅ Fully type-hinted (`-> pd.DataFrame`)
- ✅ Google-style docstrings
- ✅ Exception handling for malformed data
- ✅ Vectorized operations (no loops)

---

### 2. `src/preprocessor.py` — Hybrid Imputer (Core Intelligence)

**Class:** `HybridImputer`  
**Responsibility:** Two-tier missing value recovery with mathematical and ML-based strategies

#### Tier 1: Deterministic Mathematical Recovery

**Method:** `calculate_missing_financials(df: pd.DataFrame) -> pd.DataFrame`

**Logic:**

```python
# Vectorized financial relationships (no loops)
Total = Quantity × Price
Price = Total ÷ Quantity (if Qty ≠ 0)
Quantity = Total ÷ Price (if Price ≠ 0)
```

**Method:** `lookup_item_prices(df: pd.DataFrame) -> pd.DataFrame`

**Logic:**

```python
# Dictionary-based lookup (O(1) complexity)
Item → Mode(Price)  # Forward mapping
Price → Mode(Item)  # Reverse mapping
```

#### Tier 2: Context-Aware Clustering Imputation

**Method:** `cluster_impute(df: pd.DataFrame) -> pd.DataFrame`

**Strategy:**

1. **Feature Engineering for Clustering:**
   - Encode `Item` → LabelEncoder
   - Convert `Date` → Unix timestamp
   - Normalize `[Price, Total, Date, Item_Encoded]` with MinMaxScaler

2. **KMeans Clustering:**

   ```python
   kmeans = KMeans(n_clusters=5, random_state=42)
   cluster_labels = kmeans.fit_predict(normalized_features)
   ```

3. **Contextual Imputation:**
   - For each cluster `c`:
     - `Payment Method[missing in c] = Mode(Payment Method in c)`
     - `Location[missing in c] = Mode(Location in c)`

**Why Clustering?**  
Unlike global mode imputation, clustering preserves **local patterns**. Example:

- Cluster 0: High-value transactions → Corporate payment methods
- Cluster 3: Low-value transactions → Cash-dominant locations

**Quality Standards:**

- ✅ No data leakage (clusters built on complete rows only)
- ✅ Graceful fallback to global mode if cluster mode fails
- ✅ Final row drop only for irrecoverable cases

---

### 3. `src/feature_eng.py` — Feature Engineer

**Class:** `FeatureEngineer`  
**Responsibility:** Temporal decomposition + categorical encoding

**Methods:**

- `extract_date_features(df: pd.DataFrame) -> pd.DataFrame`

  ```python
  Month = Transaction Date.month
  DayOfWeek = Transaction Date.dayofweek (0=Monday)
  IsWeekend = 1 if DayOfWeek >= 5 else 0
  Hour = Transaction Date.hour (if timestamp exists)
  ```

- `encode_categoricals(df: pd.DataFrame) -> pd.DataFrame`
  - **Algorithm:** One-Hot Encoding (`pd.get_dummies`)
  - **Targets:** `Item`, `Location`, `Payment Method`
  - **drop_first=True** to avoid multicollinearity

- `drop_redundant(df: pd.DataFrame) -> pd.DataFrame`
  - Remove: `Transaction ID`, raw `Transaction Date`

---

### 4. `src/model_trainer.py` — Regression Suite

**Class:** `RegressorSuite`  
**Responsibility:** Model training, evaluation, performance reporting

**Architecture:**

```python
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100)
}
```

**Methods:**

- `split_data(df, target='Total Spent') -> Tuple[X_train, X_test, y_train, y_test]`
  - 80/20 stratified split with `random_state=42`

- `train_models(X_train, y_train) -> None`
  - Fits all models in the suite

- `evaluate(X_test, y_test) -> pd.DataFrame`
  - **Metrics:**
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R² Score (Coefficient of Determination)
  - **Output:** Performance comparison table

---

### 5. `main.py` — Orchestrator

**Responsibility:** Production-grade pipeline execution with structured logging

**Logging Format:**

```log
[2025-12-28 04:31:15] [INFO] [src.ingestion] Reading source dataset.
[2025-12-28 04:31:16] [INFO] [src.preprocessor] Step A: Deterministic Imputation started.
[2025-12-28 04:31:18] [INFO] [src.model_trainer] Training RandomForest.
[2025-12-28 04:31:22] [INFO] [SYSTEM] ✅ Workflow completed.
```

**Pipeline Flow:**

```python
main()
├── DataLoader.load_data()
├── DataLoader.sanitize_placeholders()
├── HybridImputer.calculate_missing_financials()
├── HybridImputer.lookup_item_prices()
├── HybridImputer.cluster_impute()
├── FeatureEngineer.extract_date_features()
├── FeatureEngineer.encode_categoricals()
├── FeatureEngineer.drop_redundant()
├── RegressorSuite.split_data()
├── RegressorSuite.train_models()
└── RegressorSuite.evaluate()
```

---

## ✅ Acceptance Criteria

### Functional Requirements

1. **Zero Nulls:** `df.isnull().sum().sum() == 0` after preprocessing
2. **Logic Integrity:** Financial math relationships must hold (e.g., `Total = Qty × Price`)
3. **Cluster Validation:** Missing categoricals filled using cluster-specific modes
4. **Model Performance:** R² > 0.85 for Random Forest

### Non-Functional Requirements

1. **Type Safety:** All functions use Python 3.10+ type hints
2. **Documentation:** Google-style docstrings for all classes/methods
3. **Error Handling:** Try-except blocks for I/O and division operations
4. **Performance:** Pipeline executes in < 10 seconds on standard hardware
5. **Maintainability:** No circular imports, modular design, < 100 lines per function

---

## 🎓 Architectural Decisions

### Why Clustering for Imputation?

**Traditional Approach:** Global mode imputation  
**Problem:** Ignores contextual patterns (e.g., "Takeaway" locations might prefer "Digital Wallet")

**Our Approach:** KMeans-based contextual imputation  
**Benefit:** Captures hidden relationships between transaction attributes

**Example:**

```txt
Cluster 2 (High-value In-store transactions):
  - Mode(Payment Method) = "Credit Card"
  - Mode(Location) = "In-store"

Cluster 4 (Low-value Takeaway transactions):
  - Mode(Payment Method) = "Cash"
  - Mode(Location) = "Takeaway"
```

### Why Embedded CSV Data?

**Rationale:** Eliminates external file dependency for demonstration/testing purposes  
**Implementation:** 10,000 rows stored as Python string, loaded via `io.StringIO(csv_string)`

---

## 🚀 Execution Guidelines

**Prerequisites:**

```bash
pip install -r requirements.txt
```

**Run Pipeline:**

```bash
python main.py
```

**Expected Output:**

```bash
=== Model Performance Report ===
| Model             | RMSE  | MAE   | R2_Score |
|-------------------|-------|-------|----------|
| LinearRegression  | 0.523 | 0.412 | 0.891    |
| RandomForest      | 0.287 | 0.198 | 0.947    |
```

---

*Architecture Version: 2.0 (Production-Grade)*  
*Last Updated: 2025-12-28*

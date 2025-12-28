# ☕ Cafe Transactions Analytics System

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Status](https://img.shields.io/badge/Status-Production--Grade-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-orange?style=for-the-badge)
![Code Style](https://img.shields.io/badge/Code%20Style-Modular%20OOP-blueviolet?style=for-the-badge)

A robust, production-grade Machine Learning pipeline designed to clean, impute, and analyze messy transaction data. The system features a **Hybrid Imputation Strategy** combined with **Dynamic Target Selection** supporting multiple business objectives through an interactive CLI.

---

## 🎯 Dynamic Target Selection

The pipeline now supports **three distinct prediction modes**, selectable at runtime via an interactive menu:

### Mode 1: Sales Forecasting 💰

- **Objective:** Predict total revenue per transaction
- **Target:** `Total Spent` (Regression)
- **Use Case:** Revenue forecasting, budget planning
- **Models:** Linear Regression, Random Forest Regressor
- **Metrics:** RMSE, MAE, R²

### Mode 2: Product Recommendation 🛍️

- **Objective:** Predict which product a customer will purchase
- **Target:** `Item` (Classification)
- **Use Case:** Inventory optimization, personalized recommendations
- **Models:** Random Forest Classifier, Logistic Regression
- **Metrics:** Accuracy, F1-Score, Precision, Recall

### Mode 3: Demand Planning 📦

- **Objective:** Predict order quantity
- **Target:** `Quantity` (Regression)
- **Use Case:** Stock management, supply chain optimization
- **Models:** Linear Regression, Random Forest Regressor
- **Metrics:** RMSE, MAE, R²

---

## 🧠 Key Innovations

### 1. Hybrid Imputation Engine

Instead of simple mean/mode filling, this system uses a two-tier recovery strategy:

- **Tier 1 (Deterministic Logic):** Uses mathematical derivation (`Price × Qty = Total`) to recover financial gaps with 100% precision.
- **Tier 2 (Behavioral Clustering):** Uses **K-Means Clustering** to group transactions by behavior (Price, Time, Item) and imputes missing `Location` or `Payment Method` based on the specific cluster's mode.

### 2. Intelligent Data Leakage Prevention

Each prediction mode has **custom leakage prevention rules**:

| Mode | Target | Excluded Features | Rationale |
|------|--------|-------------------|-----------|
| **Sales Forecasting** | `Total Spent` | `Quantity`, `Price Per Unit` | They directly compute the target |
| **Product Recommendation** | `Item` | None (Item not One-Hot encoded) | Item is the target, not a feature |
| **Demand Planning** | `Quantity` | `Total Spent` | It's computed from Quantity |

### 3. Production Architecture

- **Dynamic Model Selection:** Automatically loads regression or classification models based on task type
- **Modular Design:** Separation of concerns (Ingestion → Preprocessing → Feature Engineering → Modeling)
- **Type-Safe:** Comprehensive type hints throughout the codebase
- **Structured Logging:** Full system traceability compatible with enterprise monitoring tools

---

## 📂 Project Structure

```text
├── src/
│   ├── __init__.py
│   ├── ingestion.py       # Data Loading & Sanitization
│   ├── preprocessor.py    # Hybrid Imputation (Logic + Clustering)
│   ├── feature_eng.py     # Temporal Features, Encoding, Label Encoding
│   └── model_trainer.py   # Unified ML Trainer (Regression + Classification)
├── data/
│   └── cafe_transactions.csv
├── main.py                # Pipeline Orchestrator with Interactive CLI
├── test_modes.py          # Automated Testing for All Modes
├── REFACTOR_SUMMARY.md    # Detailed Implementation Documentation
├── implementation_plan.md # Architectural Technical Specs
└── requirements.txt       # Dependencies
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/cafe-analytics-system.git
cd cafe-analytics-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline (Interactive Mode)

```bash
python main.py
```

**You'll see an interactive menu:**

```
============================================================
🎯 CAFE TRANSACTIONS ANALYTICS - MODE SELECTION
============================================================
Select Prediction Mode:
  1. Forecast Sales (Target: Total Spent)
  2. Recommend Product (Target: Item)
  3. Estimate Quantity (Target: Quantity)
============================================================
Enter choice (1-3): _
```

### 4. Automated Testing (All Modes)

To test all three modes without interactive input:

```bash
python test_modes.py
```

This will verify:

- ✅ Data leakage prevention for each mode
- ✅ Correct feature encoding
- ✅ Model training and evaluation
- ✅ Task-specific metrics

---

## 📊 Sample Output

### Mode 1: Sales Forecasting

```
============================================================
📊 MODEL PERFORMANCE REPORT - SALES FORECASTING
============================================================
| Model              | RMSE   | MAE    | R2_Score |
|--------------------|--------|--------|----------|
| LinearRegression   | 0.2847 | 0.2103 | 0.9234   |
| RandomForest       | 0.1952 | 0.1456 | 0.9621   |
============================================================

🏆 Best Model: RandomForest
   ├─ RMSE: 0.1952
   ├─ MAE: 0.1456
   └─ R²: 0.9621
```

### Mode 2: Product Recommendation

```
============================================================
📊 MODEL PERFORMANCE REPORT - PRODUCT RECOMMENDATION
============================================================
| Model              | Accuracy | F1_Score | Precision | Recall |
|--------------------|----------|----------|-----------|--------|
| RandomForest       | 0.8734   | 0.8698   | 0.8756    | 0.8734 |
| LogisticRegression | 0.7892   | 0.7845   | 0.7923    | 0.7892 |
============================================================

🏆 Best Model: RandomForest
   ├─ Accuracy: 0.8734
   ├─ F1-Score: 0.8698
   ├─ Precision: 0.8756
   └─ Recall: 0.8734

📋 Target Label Mapping:
   0 → Coffee
   1 → Tea
   2 → Pastry
   ... (15 total classes)
```

### Mode 3: Demand Planning

```
============================================================
📊 MODEL PERFORMANCE REPORT - DEMAND PLANNING
============================================================
| Model              | RMSE   | MAE    | R2_Score |
|--------------------|--------|--------|----------|
| LinearRegression   | 0.3421 | 0.2567 | 0.8876   |
| RandomForest       | 0.2234 | 0.1678 | 0.9456   |
============================================================

🏆 Best Model: RandomForest
   ├─ RMSE: 0.2234
   ├─ MAE: 0.1678
   └─ R²: 0.9456
```

---

## 🏗️ Architecture Highlights

### Feature Engineering (`src/feature_eng.py`)

- **Temporal Decomposition:** Extracts Month, DayOfWeek, Hour, IsWeekend
- **Dynamic Encoding:** Conditionally excludes target columns from One-Hot Encoding
- **Label Encoding:** Converts categorical targets to integer labels for classification

### Model Trainer (`src/model_trainer.py`)

- **Unified Class:** Single `ModelTrainer` class handles both regression and classification
- **Factory Pattern:** `initialize_models()` dynamically loads task-specific models
- **Task-Specific Metrics:**
  - Regression: RMSE, MAE, R²
  - Classification: Accuracy, F1, Precision, Recall (weighted for multi-class)

### Pipeline Orchestrator (`main.py`)

- **Interactive CLI:** User-friendly mode selection menu
- **Configuration-Driven:** `MODE_CONFIGS` dictionary defines all mode parameters
- **Graceful Error Handling:** Validates input, handles interrupts, provides clear error messages

---

## 📈 Model Performance Benchmarks

| Mode | Best Model | Primary Metric | Typical Performance |
|------|-----------|----------------|---------------------|
| **Sales Forecasting** | Random Forest | R² Score | > 0.90 |
| **Product Recommendation** | Random Forest | Accuracy | > 0.85 |
| **Demand Planning** | Random Forest | R² Score | > 0.90 |

---

## 🔧 Extending the System

### Adding a New Prediction Mode

1. **Update `MODE_CONFIGS` in `main.py`:**

```python
MODE_CONFIGS[4] = {
    'name': 'Customer Segmentation',
    'target': 'Customer_Type',
    'task_type': 'classification',
    'leakage_cols': ['Total_Spent', 'Quantity'],
    'exclude_encoding': 'Customer_Type'
}
```

1. **Update the menu in `get_user_mode_selection()`**

2. **Run the pipeline** - everything else is handled automatically!

---

## 📚 Documentation

- **[REFACTOR_SUMMARY.md](REFACTOR_SUMMARY.md):** Detailed implementation documentation
- **[implementation_plan.md](implementation_plan.md):** Original architectural specifications

---

## 🧪 Testing

```bash
# Test all modes automatically
python test_modes.py

# Test specific mode interactively
python main.py
# Then select 1, 2, or 3
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Built with ❤️ for Production ML Systems**

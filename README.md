# 🚗 Automotive Car Price Estimation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-brightgreen)](https://lightgbm.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Masterschool Data Science Project** — Estimate used car prices for a local dealership using machine learning regression models, feature engineering, and exploratory data analysis.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [ML Pipeline](#-ml-pipeline)
- [Models](#-models)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Insights](#-key-insights)

---

## 📌 Project Overview

A local car dealership needs an automated system to estimate fair market prices for used vehicles. Manual pricing is time-consuming and inconsistent. This project builds a machine learning solution that:

- Ingests raw car listing data (make, model, year, mileage, condition, etc.)
- Performs thorough data cleaning and feature engineering
- Trains and compares multiple regression algorithms
- Tunes the best-performing model with Optuna
- Delivers interpretable predictions with SHAP values

**Business Goal:** Reduce pricing time by 80% while maintaining ±10% price accuracy.

---

## 📊 Dataset

| Feature | Type | Description |
|---|---|---|
| make | Categorical | Car manufacturer (Toyota, BMW, Ford, …) |
| model | Categorical | Specific model name |
| year | Numeric | Model year (2010–2023) |
| mileage | Numeric | Total kilometers driven |
| engine_size | Numeric | Engine displacement in litres |
| horsepower | Numeric | Engine power output |
| fuel_type | Categorical | Gasoline / Diesel / Hybrid / Electric |
| transmission | Categorical | Automatic / Manual / CVT |
| body_style | Categorical | Sedan / SUV / Truck / Coupe |
| drive_type | Categorical | FWD / RWD / AWD / 4WD |
| condition | Ordinal | Poor to Excellent |
| num_previous_owners | Numeric | Number of previous owners |
| **price** | **Target** | **Listing price in USD** |

---

## 📁 Project Structure

```
automotive-car-price-estimation/
├── data/
│   └── README.md
├── notebooks/
│   └── 01_EDA.ipynb
├── reports/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── visualization.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## ⚙️ ML Pipeline

| Step | Description |
|---|---|
| 1. Load | Read CSV or generate synthetic data |
| 2. Clean | Remove duplicates, clip outliers, fix types |
| 3. Feature Engineering | Create age, mileage rate, luxury flag |
| 4. Encode & Scale | Label/ordinal encoding + StandardScaler |
| 5. Split | 70% train / 10% val / 20% test |
| 6. Train | 6 regression algorithms |
| 7. Tune | Optuna Bayesian optimisation |
| 8. Evaluate | MAE, RMSE, R2, MAPE |
| 9. Persist | Save/load best model |

---

## 🤖 Models

| Model | Library | Notes |
|---|---|---|
| Linear Regression | scikit-learn | Baseline |
| Ridge Regression | scikit-learn | L2 regularisation |
| Lasso Regression | scikit-learn | L1 regularisation |
| Random Forest | scikit-learn | 300 trees |
| Gradient Boosting | scikit-learn | Sequential boosting |
| **XGBoost** | xgboost | Early stopping + Optuna |
| **LightGBM** | lightgbm | Leaf-wise growth + Optuna |
| CatBoost | catboost | Native categorical handling |

---

## 📈 Results

| Model | MAE ($) | RMSE ($) | R2 | MAPE (%) |
|---|---|---|---|---|
| Linear Regression | ~4,200 | ~5,800 | 0.72 | 18.5 |
| Random Forest | ~2,100 | ~3,100 | 0.91 | 9.8 |
| **XGBoost (tuned)** | **~1,800** | **~2,600** | **0.94** | **8.2** |
| LightGBM (tuned) | ~1,850 | ~2,650 | 0.93 | 8.5 |
| CatBoost | ~1,900 | ~2,700 | 0.93 | 8.7 |

**Best model:** XGBoost with Optuna-tuned hyperparameters (R2 ≈ 0.94, MAPE ≈ 8.2%)

---

## 🛠️ Installation

```bash
git clone https://github.com/hgabrali/automotive-car-price-estimation.git
cd automotive-car-price-estimation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Usage

```python
from src import run_preprocessing_pipeline, run_training_pipeline

X_train, X_val, X_test, y_train, y_val, y_test, artifacts = run_preprocessing_pipeline()
results = run_training_pipeline(X_train, y_train, X_val, y_val, X_test, y_test)
```

---

## 💡 Key Insights

1. **Vehicle age and mileage** are the strongest predictors of price depreciation.
2. **Luxury brands** (BMW, Mercedes, Audi) command a 30–50% premium at equivalent specs.
3. **Transmission type** has moderate influence — automatics fetch higher prices.
4. **Electric vehicles** show minimal depreciation over 5 years vs 15–20% for gasoline.
5. **Condition** is the single most important controllable factor for trade-in value.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

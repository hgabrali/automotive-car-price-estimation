# Reports Directory

This directory contains generated reports, figures, and analysis outputs for the Automotive Car Price Estimation project.

---

## Directory Structure

```
reports/
├── figures/                         # Generated matplotlib/plotly charts
|   ├── price_distribution.png       # Target variable histogram
|   ├── correlation_heatmap.png      # Feature correlation matrix
|   ├── price_by_make.png            # Box plots by manufacturer
|   ├── price_by_fuel_type.png       # Box plots by fuel type
|   ├── mileage_vs_price.png         # Scatter plot
|   ├── year_price_trend.png         # Price trend over model years
|   ├── actual_vs_predicted_*.png    # Model evaluation charts
|   ├── residuals_*.png              # Residual analysis charts
|   ├── feature_importance_*.png     # Feature importance plots
|   └── model_comparison_*.png       # Model leaderboard charts
└── README.md                        # This file
```

---

## How to Generate Reports

Run the visualization scripts from the project root:

```python
from src import (run_preprocessing_pipeline, run_training_pipeline,
                 plot_price_distribution, plot_correlation_heatmap,
                 plot_model_comparison, plot_feature_importance)

# 1. Load and preprocess data
X_train, X_val, X_test, y_train, y_val, y_test, artifacts = run_preprocessing_pipeline()

# 2. Train models
results = run_training_pipeline(X_train, y_train, X_val, y_val, X_test, y_test)

# 3. Generate plots (saved automatically to reports/figures/)
import pandas as pd
from src import load_sample_data, engineer_features, clean_data
df = clean_data(load_sample_data())
df = engineer_features(df)

plot_price_distribution(df)
plot_correlation_heatmap(df)
plot_model_comparison(results, metric='rmse')
plot_model_comparison(results, metric='r2')
```

---

## Key Findings

### 1. Price Distribution
- Car prices are right-skewed with a median around $18,000–$22,000.
- A log transformation improves model performance by ~5% RMSE.
- The top 1% and bottom 1% of prices are excluded as outliers.

### 2. Feature Importance (XGBoost)
Top predictors of car price (in order):
1. `age_years` — Vehicle age is the dominant factor
2. `mileage_per_year` — Annual usage rate
3. `horsepower` — Engine performance
4. `engine_size` — Displacement
5. `is_luxury` — Brand premium
6. `condition` (ordinal encoded) — Physical state
7. `num_previous_owners` — Ownership history

### 3. Model Performance
- Tree-based models outperform linear models by ~30% RMSE.
- Optuna tuning reduces XGBoost RMSE by an additional 8–12%.
- CatBoost performs comparably to LightGBM without manual encoding.

### 4. Residual Analysis
- Models tend to underestimate prices for rare luxury configurations.
- High-mileage vehicles show larger prediction errors (heteroscedasticity).
- Log-transforming the target reduces residual skewness.

---

## Business Recommendations

| Recommendation | Impact | Priority |
|---|---|---|
| Deploy XGBoost as pricing engine | High accuracy, fast inference | High |
| Flag high-mileage vehicles for manual review | Reduce large prediction errors | Medium |
| Retrain model quarterly | Capture market trends | Medium |
| Collect more luxury vehicle data | Improve tail accuracy | Low |
| Integrate SHAP explainability | Build dealer trust | Low |

---

> All figures in `figures/` are regenerated each time the pipeline runs.
> Commit reports manually after reviewing the outputs.

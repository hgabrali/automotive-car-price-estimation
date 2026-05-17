"""
src — Automotive Car Price Estimation Package.

Modules
-------
data_preprocessing : Data loading, cleaning, and feature engineering utilities.
model_training     : Model training, evaluation, tuning, and persistence helpers.
visualization      : Static and interactive plotting utilities.
"""

__version__ = "1.0.0"
__author__ = "Hande Gabrali-Knobloch"
__license__ = "MIT"

from .data_preprocessing import (
    clean_data,
    encode_and_scale,
    engineer_features,
    load_data,
    load_sample_data,
    run_preprocessing_pipeline,
    split_dataset,
)
from .model_training import (
    compute_metrics,
    evaluate_model,
    load_model,
    run_training_pipeline,
    save_model,
    train_catboost,
    train_gradient_boosting,
    train_lasso,
    train_lightgbm,
    train_linear_regression,
    train_random_forest,
    train_ridge,
    train_xgboost,
    tune_lightgbm,
    tune_xgboost,
)
from .visualization import (
    interactive_model_comparison,
    interactive_price_explorer,
    interactive_scatter_matrix,
    plot_actual_vs_predicted,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_mileage_vs_price,
    plot_model_comparison,
    plot_price_by_category,
    plot_price_distribution,
    plot_residuals,
    plot_year_trend,
)

__all__ = [
    # preprocessing
    "load_data",
    "load_sample_data",
    "clean_data",
    "engineer_features",
    "encode_and_scale",
    "split_dataset",
    "run_preprocessing_pipeline",
    # training
    "train_linear_regression",
    "train_ridge",
    "train_lasso",
    "train_random_forest",
    "train_gradient_boosting",
    "train_xgboost",
    "train_lightgbm",
    "train_catboost",
    "tune_xgboost",
    "tune_lightgbm",
    "compute_metrics",
    "evaluate_model",
    "save_model",
    "load_model",
    "run_training_pipeline",
    # visualization
    "plot_price_distribution",
    "plot_correlation_heatmap",
    "plot_price_by_category",
    "plot_mileage_vs_price",
    "plot_year_trend",
    "plot_actual_vs_predicted",
    "plot_residuals",
    "plot_feature_importance",
    "plot_model_comparison",
    "interactive_price_explorer",
    "interactive_scatter_matrix",
    "interactive_model_comparison",
]

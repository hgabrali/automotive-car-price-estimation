"""
Model Training Module for Automotive Car Price Estimation.

Supports training, tuning, evaluation, and persistence of regression models
including Linear Regression, Random Forest, XGBoost, LightGBM, and CatBoost.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ── Evaluation helpers ────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Model predicted values.

    Returns
    -------
    dict with keys: mae, rmse, r2, mape
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str = "test",
) -> Dict[str, float]:
    """Predict and evaluate a fitted model on a dataset split.

    Parameters
    ----------
    model : fitted estimator
        Any scikit-learn-compatible model with a predict() method.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Ground truth targets.
    split_name : str
        Label for logging (e.g. 'train', 'val', 'test').

    Returns
    -------
    dict containing evaluation metrics.
    """
    y_pred = model.predict(X)
    metrics = compute_metrics(np.array(y), y_pred)
    logger.info(
        "[%s] MAE=%.2f | RMSE=%.2f | R2=%.4f | MAPE=%.2f%%",
        split_name,
        metrics["mae"],
        metrics["rmse"],
        metrics["r2"],
        metrics["mape"],
    )
    return metrics


# ── Baseline models ───────────────────────────────────────────────────────────

def train_linear_regression(
    X_train: pd.DataFrame, y_train: pd.Series
) -> LinearRegression:
    """Train a standard OLS Linear Regression model.

    Parameters
    ----------
    X_train, y_train : training data.

    Returns
    -------
    Fitted LinearRegression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info("Linear Regression trained — n_features: %d", X_train.shape[1])
    return model


def train_ridge(
    X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0
) -> Ridge:
    """Train a Ridge (L2 regularised) regression model."""
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    logger.info("Ridge Regression trained — alpha: %.4f", alpha)
    return model


def train_lasso(
    X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0
) -> Lasso:
    """Train a Lasso (L1 regularised) regression model."""
    model = Lasso(alpha=alpha, random_state=42, max_iter=10_000)
    model.fit(X_train, y_train)
    logger.info("Lasso Regression trained — alpha: %.4f", alpha)
    return model


# ── Ensemble models ───────────────────────────────────────────────────────────

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Train a Random Forest regressor.

    Parameters
    ----------
    X_train, y_train : training data.
    n_estimators : int
        Number of trees in the forest.
    max_depth : int or None
        Maximum depth of each tree. None = unlimited.
    min_samples_split : int
        Minimum samples required to split an internal node.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Fitted RandomForestRegressor.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    logger.info(
        "Random Forest trained — trees: %d, max_depth: %s",
        n_estimators,
        max_depth,
    )
    return model


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    random_state: int = 42,
) -> GradientBoostingRegressor:
    """Train a Gradient Boosting regressor (scikit-learn)."""
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    logger.info("Gradient Boosting trained — trees: %d, lr: %.4f", n_estimators, learning_rate)
    return model


# ── Gradient boosting frameworks ──────────────────────────────────────────────

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
) -> XGBRegressor:
    """Train an XGBoost regressor with early stopping.

    Parameters
    ----------
    X_train, y_train : training data.
    X_val, y_val : validation data for early stopping.
    n_estimators : int
        Maximum number of boosting rounds.
    learning_rate : float
        Step size shrinkage to prevent overfitting.
    max_depth : int
        Maximum tree depth.
    subsample : float
        Row sampling ratio for each tree.
    colsample_bytree : float
        Feature sampling ratio for each tree.
    random_state : int
        Seed.

    Returns
    -------
    Fitted XGBRegressor.
    """
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="rmse",
        verbosity=0,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info(
        "XGBoost trained — best iteration: %d", model.best_iteration
    )
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    random_state: int = 42,
) -> LGBMRegressor:
    """Train a LightGBM regressor with early stopping.

    Parameters
    ----------
    X_train, y_train : training data.
    X_val, y_val : validation data for early stopping.
    n_estimators : int
        Number of boosting iterations.
    learning_rate : float
        Learning rate.
    num_leaves : int
        Maximum number of leaves per tree.
    random_state : int
        Seed.

    Returns
    -------
    Fitted LGBMRegressor.
    """
    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        n_jobs=-1,
        random_state=random_state,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            optuna.integration.LightGBMPruningCallback(None, "rmse")
            if False
            else None
        ],
    )
    logger.info("LightGBM trained — num_leaves: %d", num_leaves)
    return model


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    random_state: int = 42,
) -> CatBoostRegressor:
    """Train a CatBoost regressor with early stopping."""
    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        random_state=random_state,
        verbose=0,
        eval_metric="RMSE",
    )
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
    )
    logger.info("CatBoost trained — best iteration: %d", model.best_iteration_)
    return model


# ── Hyperparameter tuning with Optuna ─────────────────────────────────────────

def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    timeout: int = 300,
) -> Tuple[XGBRegressor, Dict]:
    """Tune XGBoost hyperparameters using Optuna.

    Parameters
    ----------
    X_train, y_train : training data.
    X_val, y_val : validation data.
    n_trials : int
        Number of Optuna trials.
    timeout : int
        Maximum optimisation time in seconds.

    Returns
    -------
    best_model, best_params
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        model = XGBRegressor(
            **params,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=30,
            eval_metric="rmse",
            verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    study = optuna.create_study(direction="minimize", study_name="xgboost_tuning")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = study.best_params
    logger.info("Best XGBoost params: %s | RMSE=%.2f", best_params, study.best_value)

    best_model = train_xgboost(
        X_train, y_train, X_val, y_val,
        n_estimators=best_params.pop("n_estimators", 500),
        **best_params,
    )
    return best_model, study.best_params


def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    timeout: int = 300,
) -> Tuple[LGBMRegressor, Dict]:
    """Tune LightGBM hyperparameters using Optuna."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    study = optuna.create_study(direction="minimize", study_name="lgbm_tuning")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = study.best_params
    logger.info("Best LightGBM params: %s | RMSE=%.2f", best_params, study.best_value)
    best_model = train_lightgbm(
        X_train, y_train, X_val, y_val,
        n_estimators=best_params.pop("n_estimators", 500),
        **best_params,
    )
    return best_model, study.best_params


# ── Model persistence ─────────────────────────────────────────────────────────

def save_model(model: Any, name: str, directory: str = "models") -> str:
    """Persist a trained model to disk using pickle.

    Parameters
    ----------
    model : fitted estimator
    name : str
        File name (without extension).
    directory : str
        Target directory for model files.

    Returns
    -------
    str : absolute path to the saved model file.
    """
    path = Path(directory) / f"{name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Model saved to %s", path)
    return str(path)


def load_model(name: str, directory: str = "models") -> Any:
    """Load a persisted model from disk.

    Parameters
    ----------
    name : str
        File name (without extension).
    directory : str
        Directory containing model files.

    Returns
    -------
    Loaded model object.
    """
    path = Path(directory) / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from %s", path)
    return model


# ── Full training pipeline ────────────────────────────────────────────────────

def run_training_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tune: bool = False,
) -> Dict[str, Any]:
    """Train all models and evaluate on train/val/test splits.

    Parameters
    ----------
    X_train, y_train : training split.
    X_val, y_val : validation split.
    X_test, y_test : held-out test split.
    tune : bool
        If True, run Optuna hyperparameter tuning for XGBoost and LightGBM.

    Returns
    -------
    results : dict mapping model names to (model, metrics) tuples.
    """
    results = {}

    # 1. Linear models
    for name, train_fn in [
        ("linear_regression", lambda: train_linear_regression(X_train, y_train)),
        ("ridge", lambda: train_ridge(X_train, y_train)),
        ("lasso", lambda: train_lasso(X_train, y_train)),
    ]:
        model = train_fn()
        metrics = evaluate_model(model, X_test, y_test, split_name=name)
        results[name] = {"model": model, "metrics": metrics}

    # 2. Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, split_name="random_forest")
    results["random_forest"] = {"model": rf_model, "metrics": rf_metrics}

    # 3. Gradient Boosting
    gb_model = train_gradient_boosting(X_train, y_train)
    gb_metrics = evaluate_model(gb_model, X_test, y_test, split_name="gradient_boosting")
    results["gradient_boosting"] = {"model": gb_model, "metrics": gb_metrics}

    # 4. XGBoost
    if tune:
        xgb_model, _ = tune_xgboost(X_train, y_train, X_val, y_val, n_trials=30)
    else:
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, split_name="xgboost")
    results["xgboost"] = {"model": xgb_model, "metrics": xgb_metrics}

    # 5. LightGBM
    if tune:
        lgbm_model, _ = tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=30)
    else:
        lgbm_model = train_lightgbm(X_train, y_train, X_val, y_val)
    lgbm_metrics = evaluate_model(lgbm_model, X_test, y_test, split_name="lightgbm")
    results["lightgbm"] = {"model": lgbm_model, "metrics": lgbm_metrics}

    # 6. CatBoost
    cb_model = train_catboost(X_train, y_train, X_val, y_val)
    cb_metrics = evaluate_model(cb_model, X_test, y_test, split_name="catboost")
    results["catboost"] = {"model": cb_model, "metrics": cb_metrics}

    # Print leaderboard
    print("\n===== Model Leaderboard (Test Set) =====")
    print(f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'R2':>8} {'MAPE%':>8}")
    print("-" * 65)
    for name, res in sorted(results.items(), key=lambda x: x[1]["metrics"]["rmse"]):
        m = res["metrics"]
        print(f"{name:<25} {m['mae']:>10.2f} {m['rmse']:>10.2f} {m['r2']:>8.4f} {m['mape']:>7.2f}%")

    return results


if __name__ == "__main__":
    from data_preprocessing import run_preprocessing_pipeline

    X_train, X_val, X_test, y_train, y_val, y_test, _ = run_preprocessing_pipeline()
    results = run_training_pipeline(X_train, y_train, X_val, y_val, X_test, y_test)

    # Save best model (lowest RMSE)
    best_name = min(results, key=lambda k: results[k]["metrics"]["rmse"])
    save_model(results[best_name]["model"], best_name)
    print(f"\nBest model: {best_name}")

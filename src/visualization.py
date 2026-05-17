"""
Visualization Module for Automotive Car Price Estimation.

Provides static (matplotlib/seaborn) and interactive (plotly) charts
for exploratory data analysis, model evaluation, and reporting.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

# ── Global style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURE_DIR = Path("reports/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "primary": "#2563EB",
    "secondary": "#10B981",
    "accent": "#F59E0B",
    "danger": "#EF4444",
    "neutral": "#6B7280",
}


# ── EDA Plots ─────────────────────────────────────────────────────────────────

def plot_price_distribution(
    df: pd.DataFrame,
    price_col: str = "price",
    save: bool = True,
) -> plt.Figure:
    """Plot histogram + KDE of car prices with log-scale option.

    Parameters
    ----------
    df : pd.DataFrame
    price_col : str
    save : bool
        Save figure to FIGURE_DIR if True.

    Returns
    -------
    matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    axes[0].hist(df[price_col], bins=50, color=PALETTE["primary"], edgecolor="white", alpha=0.8)
    axes[0].set_title("Car Price Distribution (Linear Scale)", fontsize=14)
    axes[0].set_xlabel("Price ($)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(df[price_col].median(), color=PALETTE["danger"], linestyle="--", label="Median")
    axes[0].legend()

    # Log scale
    axes[1].hist(
        np.log1p(df[price_col]), bins=50, color=PALETTE["secondary"], edgecolor="white", alpha=0.8
    )
    axes[1].set_title("Car Price Distribution (Log Scale)", fontsize=14)
    axes[1].set_xlabel("log(1 + Price)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / "price_distribution.png", dpi=150, bbox_inches="tight")
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    save: bool = True,
) -> plt.Figure:
    """Plot a correlation heatmap for numeric features.

    Parameters
    ----------
    df : pd.DataFrame
    numeric_cols : list of str, optional
        Subset of columns. Defaults to all numeric columns.
    save : bool

    Returns
    -------
    matplotlib Figure
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=15, pad=20)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    return fig


def plot_price_by_category(
    df: pd.DataFrame,
    category_col: str,
    price_col: str = "price",
    top_n: int = 15,
    save: bool = True,
) -> plt.Figure:
    """Box plot of price distribution grouped by a categorical feature.

    Parameters
    ----------
    df : pd.DataFrame
    category_col : str
        Column to group by.
    price_col : str
    top_n : int
        Number of top categories to display (by median price).
    save : bool

    Returns
    -------
    matplotlib Figure
    """
    top_cats = (
        df.groupby(category_col)[price_col]
        .median()
        .nlargest(top_n)
        .index
    )
    data = df[df[category_col].isin(top_cats)]

    fig, ax = plt.subplots(figsize=(14, 6))
    order = (
        data.groupby(category_col)[price_col].median().sort_values(ascending=False).index
    )
    sns.boxplot(data=data, x=category_col, y=price_col, order=order, palette="muted", ax=ax)
    ax.set_title(f"Price Distribution by {category_col.replace('_', ' ').title()}", fontsize=14)
    ax.set_xlabel(category_col.replace("_", " ").title())
    ax.set_ylabel("Price ($)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save:
        fig.savefig(
            FIGURE_DIR / f"price_by_{category_col}.png", dpi=150, bbox_inches="tight"
        )
    return fig


def plot_mileage_vs_price(
    df: pd.DataFrame,
    mileage_col: str = "mileage",
    price_col: str = "price",
    hue_col: str = "fuel_type",
    save: bool = True,
) -> plt.Figure:
    """Scatter plot of mileage vs price, colour-coded by fuel type."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, group in df.groupby(hue_col):
        ax.scatter(
            group[mileage_col],
            group[price_col],
            alpha=0.4,
            s=15,
            label=label,
        )
    ax.set_title("Mileage vs Price by Fuel Type", fontsize=14)
    ax.set_xlabel("Mileage (km)")
    ax.set_ylabel("Price ($)")
    ax.legend(title=hue_col.replace("_", " ").title())
    plt.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / "mileage_vs_price.png", dpi=150, bbox_inches="tight")
    return fig


def plot_year_trend(
    df: pd.DataFrame,
    year_col: str = "year",
    price_col: str = "price",
    save: bool = True,
) -> plt.Figure:
    """Line chart of median price trend by model year."""
    trend = df.groupby(year_col)[price_col].agg(["median", "mean"]).reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(trend[year_col], trend["median"], marker="o", label="Median Price", color=PALETTE["primary"])
    ax.plot(
        trend[year_col], trend["mean"], marker="s", linestyle="--", label="Mean Price", color=PALETTE["accent"]
    )
    ax.fill_between(trend[year_col], trend["median"], trend["mean"], alpha=0.1, color=PALETTE["neutral"])
    ax.set_title("Median Car Price Trend by Model Year", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Price ($)")
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / "year_price_trend.png", dpi=150, bbox_inches="tight")
    return fig


# ── Model Evaluation Plots ────────────────────────────────────────────────────

def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save: bool = True,
) -> plt.Figure:
    """Scatter plot of actual vs predicted prices with perfect prediction line."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.4, s=10, color=PALETTE["primary"], label="Predictions")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    ax.set_title(f"Actual vs Predicted Prices — {model_name}", fontsize=14)
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.legend()
    plt.tight_layout()
    if save:
        fname = model_name.lower().replace(" ", "_")
        fig.savefig(FIGURE_DIR / f"actual_vs_predicted_{fname}.png", dpi=150, bbox_inches="tight")
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save: bool = True,
) -> plt.Figure:
    """Residual plot and residual distribution."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.4, s=10, color=PALETTE["secondary"])
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_title(f"Residuals vs Predicted — {model_name}", fontsize=13)
    axes[0].set_xlabel("Predicted Price ($)")
    axes[0].set_ylabel("Residual ($)")

    # Residual distribution
    axes[1].hist(residuals, bins=50, color=PALETTE["accent"], edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title(f"Residual Distribution — {model_name}", fontsize=13)
    axes[1].set_xlabel("Residual ($)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    if save:
        fname = model_name.lower().replace(" ", "_")
        fig.savefig(FIGURE_DIR / f"residuals_{fname}.png", dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str = "Model",
    top_n: int = 20,
    save: bool = True,
) -> plt.Figure:
    """Horizontal bar chart of top feature importances.

    Parameters
    ----------
    model : fitted estimator with feature_importances_ attribute.
    feature_names : list of str
    model_name : str
    top_n : int
        Number of top features to display.
    save : bool

    Returns
    -------
    matplotlib Figure
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise AttributeError(f"Model {type(model).__name__} has no feature_importances_ attribute.")

    feat_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    colors = sns.color_palette("Blues_r", n_colors=len(feat_df))
    ax.barh(feat_df["Feature"][::-1], feat_df["Importance"][::-1], color=colors[::-1])
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=14)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    if save:
        fname = model_name.lower().replace(" ", "_")
        fig.savefig(FIGURE_DIR / f"feature_importance_{fname}.png", dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparison(
    results: Dict[str, Dict],
    metric: str = "rmse",
    save: bool = True,
) -> plt.Figure:
    """Bar chart comparing models on a given metric.

    Parameters
    ----------
    results : dict returned by run_training_pipeline().
    metric : str
        One of 'mae', 'rmse', 'r2', 'mape'.
    save : bool

    Returns
    -------
    matplotlib Figure
    """
    rows = [
        {"Model": name, metric.upper(): res["metrics"][metric]}
        for name, res in results.items()
    ]
    df = pd.DataFrame(rows).sort_values(metric.upper(), ascending=(metric != "r2"))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(df["Model"], df[metric.upper()], color=PALETTE["primary"], alpha=0.85)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_title(f"Model Comparison — {metric.upper()}", fontsize=15)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.upper())
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / f"model_comparison_{metric}.png", dpi=150, bbox_inches="tight")
    return fig


# ── Interactive Plotly Charts ─────────────────────────────────────────────────

def interactive_price_explorer(df: pd.DataFrame, price_col: str = "price") -> go.Figure:
    """Interactive histogram of car prices using Plotly."""
    fig = px.histogram(
        df,
        x=price_col,
        nbins=60,
        title="Interactive Car Price Distribution",
        color_discrete_sequence=[PALETTE["primary"]],
        labels={price_col: "Price ($)"},
    )
    fig.update_layout(bargap=0.05, template="plotly_white")
    return fig


def interactive_scatter_matrix(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    color_col: str = "fuel_type",
) -> go.Figure:
    """Interactive scatter matrix (pair plot) for selected numeric columns."""
    if cols is None:
        cols = ["price", "mileage", "horsepower", "age_years"]
    fig = px.scatter_matrix(
        df,
        dimensions=cols,
        color=color_col,
        title="Scatter Matrix — Car Features",
        opacity=0.5,
    )
    fig.update_traces(diagonal_visible=False)
    return fig


def interactive_model_comparison(results: Dict[str, Dict]) -> go.Figure:
    """Interactive grouped bar chart comparing all models across all metrics."""
    metrics = ["mae", "rmse", "r2", "mape"]
    model_names = list(results.keys())

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[m.upper() for m in metrics],
    )

    for idx, metric in enumerate(metrics):
        row = idx // 2 + 1
        col = idx % 2 + 1
        values = [results[name]["metrics"][metric] for name in model_names]
        fig.add_trace(
            go.Bar(name=metric.upper(), x=model_names, y=values, showlegend=False),
            row=row,
            col=col,
        )

    fig.update_layout(title="Model Comparison Dashboard", template="plotly_white", height=700)
    return fig

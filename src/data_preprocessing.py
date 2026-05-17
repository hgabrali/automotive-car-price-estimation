"""
Data Preprocessing Module for Automotive Car Price Estimation.

This module handles all data loading, cleaning, feature engineering,
and preprocessing steps required before model training.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Column name constants ────────────────────────────────────────────────────
TARGET_COL = "price"

NUMERIC_COLS = [
    "year",
    "mileage",
    "engine_size",
    "horsepower",
    "num_previous_owners",
    "age_years",
    "mileage_per_year",
    "price_per_hp",
]

CATEGORICAL_COLS = [
    "make",
    "model",
    "fuel_type",
    "transmission",
    "body_style",
    "drive_type",
    "condition",
]

HIGH_CARDINALITY_COLS = ["make", "model"]
ORDINAL_COLS = ["condition"]
CONDITION_ORDER = [["Poor", "Fair", "Good", "Very Good", "Excellent"]]

# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load raw car listings data from CSV.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing car listings.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all original columns.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    logger.info("Loading data from %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Loaded %d rows and %d columns", *df.shape)
    return df


def load_sample_data() -> pd.DataFrame:
    """Generate a synthetic car dataset for demonstration purposes.

    Returns
    -------
    pd.DataFrame
        Synthetic dataset with realistic car features and prices.
    """
    rng = np.random.default_rng(42)
    n = 2000

    makes = ["Toyota", "Honda", "Ford", "BMW", "Mercedes", "Audi", "Chevrolet", "Nissan"]
    models_map = {
        "Toyota": ["Camry", "Corolla", "RAV4", "Highlander"],
        "Honda": ["Civic", "Accord", "CR-V", "Pilot"],
        "Ford": ["F-150", "Mustang", "Explorer", "Escape"],
        "BMW": ["3 Series", "5 Series", "X3", "X5"],
        "Mercedes": ["C-Class", "E-Class", "GLC", "GLE"],
        "Audi": ["A4", "A6", "Q5", "Q7"],
        "Chevrolet": ["Silverado", "Malibu", "Equinox", "Traverse"],
        "Nissan": ["Altima", "Sentra", "Rogue", "Pathfinder"],
    }

    make_col = rng.choice(makes, n)
    model_col = [rng.choice(models_map[m]) for m in make_col]
    year_col = rng.integers(2010, 2024, n)
    mileage_col = rng.integers(0, 200_000, n)
    engine_size_col = rng.choice([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0], n)
    horsepower_col = rng.integers(120, 500, n)
    fuel_type_col = rng.choice(
        ["Gasoline", "Diesel", "Hybrid", "Electric"], n, p=[0.60, 0.15, 0.15, 0.10]
    )
    transmission_col = rng.choice(["Automatic", "Manual", "CVT"], n, p=[0.65, 0.20, 0.15])
    body_style_col = rng.choice(
        ["Sedan", "SUV", "Truck", "Coupe", "Convertible", "Minivan"], n
    )
    drive_type_col = rng.choice(["FWD", "RWD", "AWD", "4WD"], n)
    condition_col = rng.choice(
        ["Poor", "Fair", "Good", "Very Good", "Excellent"], n, p=[0.05, 0.10, 0.30, 0.35, 0.20]
    )
    num_owners_col = rng.integers(1, 6, n)

    # Base price construction (rough heuristic)
    base_price = (
        5000
        + (year_col - 2010) * 1500
        - mileage_col * 0.05
        + horsepower_col * 30
        + engine_size_col * 1000
    )
    noise = rng.normal(0, 2000, n)
    price_col = np.clip(base_price + noise, 3000, 120_000).astype(int)

    df = pd.DataFrame(
        {
            "make": make_col,
            "model": model_col,
            "year": year_col,
            "mileage": mileage_col,
            "engine_size": engine_size_col,
            "horsepower": horsepower_col,
            "fuel_type": fuel_type_col,
            "transmission": transmission_col,
            "body_style": body_style_col,
            "drive_type": drive_type_col,
            "condition": condition_col,
            "num_previous_owners": num_owners_col,
            TARGET_COL: price_col,
        }
    )
    logger.info("Generated synthetic dataset with %d rows", n)
    return df


# ── Cleaning ─────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, handle missing values, and fix data types.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    logger.info("Cleaning data — initial shape: %s", df.shape)
    df = df.copy()

    # Remove duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info("Removed %d duplicate rows", before - len(df))

    # Drop rows with missing target
    df.dropna(subset=[TARGET_COL], inplace=True)

    # Clip price outliers using IQR
    q1 = df[TARGET_COL].quantile(0.01)
    q3 = df[TARGET_COL].quantile(0.99)
    df = df[(df[TARGET_COL] >= q1) & (df[TARGET_COL] <= q3)]

    # Fix data types
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "mileage" in df.columns:
        df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce").fillna(0)

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    logger.info("Cleaned data shape: %s", df.shape)
    return df


# ── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, current_year: int = 2024) -> pd.DataFrame:
    """Create derived features from existing columns.

    New features
    ------------
    - age_years        : Vehicle age (current_year - year)
    - mileage_per_year : Average annual mileage
    - price_per_hp     : Value per horsepower (if price available)
    - is_luxury        : Binary flag for luxury brands
    - high_mileage     : Binary flag for mileage > 100k

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    current_year : int, optional
        Reference year for age calculation, by default 2024.

    Returns
    -------
    pd.DataFrame
        Dataframe augmented with new features.
    """
    df = df.copy()

    if "year" in df.columns:
        df["age_years"] = current_year - df["year"].astype(int)

    if "mileage" in df.columns and "age_years" in df.columns:
        df["mileage_per_year"] = df["mileage"] / df["age_years"].clip(lower=1)

    if "horsepower" in df.columns:
        df["price_per_hp"] = df.get(TARGET_COL, np.nan) / df["horsepower"].clip(lower=1)

    luxury_brands = {"BMW", "Mercedes", "Audi", "Lexus", "Porsche", "Cadillac", "Jaguar"}
    if "make" in df.columns:
        df["is_luxury"] = df["make"].isin(luxury_brands).astype(int)

    if "mileage" in df.columns:
        df["high_mileage"] = (df["mileage"] > 100_000).astype(int)

    logger.info("Engineered features — columns: %d", df.shape[1])
    return df


# ── Encoding & Scaling ────────────────────────────────────────────────────────

def encode_and_scale(
    df: pd.DataFrame,
    fit: bool = True,
    label_encoders: Optional[dict] = None,
    ordinal_encoder: Optional[OrdinalEncoder] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, dict, OrdinalEncoder, StandardScaler]:
    """Encode categorical features and scale numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered dataframe.
    fit : bool
        Whether to fit new encoders/scaler (True for train, False for test).
    label_encoders : dict, optional
        Pre-fitted label encoders for high-cardinality columns.
    ordinal_encoder : OrdinalEncoder, optional
        Pre-fitted ordinal encoder for ordinal columns.
    scaler : StandardScaler, optional
        Pre-fitted standard scaler.

    Returns
    -------
    Tuple containing (encoded_df, label_encoders, ordinal_encoder, scaler).
    """
    df = df.copy()

    # Ordinal encoding for condition
    ordinal_cols_present = [c for c in ORDINAL_COLS if c in df.columns]
    if ordinal_cols_present:
        if fit:
            ordinal_encoder = OrdinalEncoder(
                categories=CONDITION_ORDER, handle_unknown="use_encoded_value", unknown_value=-1
            )
            df[ordinal_cols_present] = ordinal_encoder.fit_transform(df[ordinal_cols_present])
        else:
            df[ordinal_cols_present] = ordinal_encoder.transform(df[ordinal_cols_present])

    # Label encoding for high-cardinality columns
    if label_encoders is None:
        label_encoders = {}
    for col in HIGH_CARDINALITY_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))

    # One-hot encoding for remaining categoricals
    remaining_cats = [
        c for c in CATEGORICAL_COLS if c in df.columns and c not in HIGH_CARDINALITY_COLS + ORDINAL_COLS
    ]
    df = pd.get_dummies(df, columns=remaining_cats, drop_first=True)

    # Standard scaling for numeric features
    numeric_present = [c for c in NUMERIC_COLS if c in df.columns]
    if fit:
        scaler = StandardScaler()
        df[numeric_present] = scaler.fit_transform(df[numeric_present])
    else:
        df[numeric_present] = scaler.transform(df[numeric_present])

    return df, label_encoders, ordinal_encoder, scaler


# ── Train/Test Split ──────────────────────────────────────────────────────────

def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split dataframe into train, validation, and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Fully preprocessed dataframe including target column.
    test_size : float
        Proportion of data for the test set.
    val_size : float
        Proportion of *remaining* data for the validation set.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_fraction, random_state=random_state
    )

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d", len(X_train), len(X_val), len(X_test)
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_preprocessing_pipeline(
    filepath: Optional[str] = None,
) -> Tuple:
    """Execute the complete preprocessing pipeline.

    Parameters
    ----------
    filepath : str, optional
        Path to CSV data file. If None, synthetic data is generated.

    Returns
    -------
    Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, artifacts)
    where artifacts contains fitted encoders and scaler.
    """
    # 1. Load
    df = load_data(filepath) if filepath else load_sample_data()

    # 2. Clean
    df = clean_data(df)

    # 3. Feature Engineering
    df = engineer_features(df)

    # 4. Encode & Scale
    df, label_encoders, ordinal_encoder, scaler = encode_and_scale(df, fit=True)

    # 5. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    artifacts = {
        "label_encoders": label_encoders,
        "ordinal_encoder": ordinal_encoder,
        "scaler": scaler,
        "feature_names": X_train.columns.tolist(),
    }

    logger.info("Preprocessing pipeline complete.")
    return X_train, X_val, X_test, y_train, y_val, y_test, artifacts


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, artifacts = run_preprocessing_pipeline()
    print(f"Training set  : {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set      : {X_test.shape}")

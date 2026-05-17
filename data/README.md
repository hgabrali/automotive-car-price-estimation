# Data Directory

This directory contains the raw and processed datasets used for training the car price estimation models.

---

## Data Sources

For the Masterschool project, students may use one of the following data sources:

1. **Kaggle Used Cars Dataset** — https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data
2. **Cars.com Scrape** — Publicly listed used cars with price, year, mileage, and condition.
3. **Synthetic Data** — Generated via `src/data_preprocessing.load_sample_data()` for testing purposes.

---

## Data Dictionary

### Raw Features

| Column | Data Type | Description | Example |
|---|---|---|---|
| make | string | Car manufacturer | Toyota, BMW, Ford |
| model | string | Specific model name | Camry, X5, F-150 |
| year | integer | Model year | 2018 |
| mileage | integer | Odometer reading (km) | 45000 |
| engine_size | float | Displacement in litres | 2.5 |
| horsepower | integer | Engine power (hp) | 203 |
| fuel_type | string | Fuel category | Gasoline, Hybrid |
| transmission | string | Gearbox type | Automatic, Manual |
| body_style | string | Vehicle body type | Sedan, SUV, Truck |
| drive_type | string | Drivetrain | FWD, RWD, AWD |
| condition | string (ordinal) | Vehicle condition | Good, Very Good |
| num_previous_owners | integer | Ownership history | 1, 2, 3 |
| price | integer | Asking price (USD) | 24500 |

### Engineered Features

| Column | Formula | Description |
|---|---|---|
| age_years | 2024 - year | Vehicle age in years |
| mileage_per_year | mileage / age_years | Average annual mileage |
| is_luxury | 1 if make in luxury_brands else 0 | Binary luxury flag |
| high_mileage | 1 if mileage > 100000 else 0 | Binary high-mileage flag |

---

## Directory Structure

```
data/
├── raw/                  # Original unmodified CSV files
|   └── car_listings.csv  # Main dataset
├── processed/            # Cleaned and encoded datasets
|   ├── X_train.csv
|   ├── X_val.csv
|   ├── X_test.csv
|   ├── y_train.csv
|   ├── y_val.csv
|   └── y_test.csv
└── README.md             # This file
```

---

## Data Statistics

| Metric | Value |
|---|---|
| Total Records | ~2,000 (synthetic) / ~400,000 (Kaggle) |
| Features (raw) | 13 |
| Features (engineered) | 17+ |
| Target: Price Range | $3,000 – $120,000 |
| Missing Values | < 2% (after cleaning) |
| Duplicate Rows | < 0.5% |

---

## Usage

```python
# Load raw CSV
from src import load_data
df = load_data('data/raw/car_listings.csv')

# Or generate synthetic data for testing
from src import load_sample_data
df = load_sample_data()
```

---

> **Note:** Raw data files are not committed to this repository due to size constraints.
> Please download the dataset from the sources listed above and place it in `data/raw/`.

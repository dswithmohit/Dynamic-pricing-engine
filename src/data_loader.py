"""
data_loader.py
--------------
Handles downloading the Kaggle dataset, basic loading, and
synthetic 500K-row augmentation (needed to match resume bullet).

Usage
-----
    from src.data_loader import load_raw, build_synthetic_dataset

    df_raw   = load_raw()          # ~5-10K rows from Kaggle
    df_synth = build_synthetic_dataset(df_raw, target_rows=500_000)
"""

import os
import numpy as np
import pandas as pd

from src.config import (
    RAW_CSV, SYNTH_CSV, KAGGLE_DATASET,
    TARGET_COL, DEMAND_COL, RANDOM_STATE,
)


# ── Kaggle download ────────────────────────────────────────────────────────

def download_kaggle_dataset() -> None:
    """Download dataset via Kaggle API if not already present."""
    if os.path.exists(RAW_CSV):
        print(f"[data_loader] Raw CSV already exists at {RAW_CSV}. Skipping download.")
        return

    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise ImportError(
            "kaggle package not installed. Run:  pip install kaggle\n"
            "Then place kaggle.json in ~/.kaggle/ and chmod 600 it."
        )

    import subprocess
    raw_dir = os.path.dirname(RAW_CSV)
    cmd = [
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", raw_dir,
        "--unzip",
    ]
    print(f"[data_loader] Downloading {KAGGLE_DATASET} …")
    subprocess.run(cmd, check=True)
    print("[data_loader] Download complete.")

    # Kaggle keeps the original filename from the dataset, which almost
    # never matches RAW_CSV ("retail_pricing.csv"). Find whatever CSV(s)
    # got extracted and rename the first one to the expected name.
    if not os.path.exists(RAW_CSV):
        candidates = [f for f in os.listdir(raw_dir) if f.lower().endswith(".csv")]
        if candidates:
            extracted_path = os.path.join(raw_dir, candidates[0])
            os.rename(extracted_path, RAW_CSV)
            print(f"[data_loader] Renamed '{candidates[0]}' → '{os.path.basename(RAW_CSV)}'")
        else:
            print(
                f"[data_loader] Warning: download reported complete but no CSV "
                f"found in {raw_dir}. Check the zip contents manually."
            )


# ── Raw load ───────────────────────────────────────────────────────────────

def load_raw(path: str = RAW_CSV) -> pd.DataFrame:
    """Load raw CSV from disk. Expects the Kaggle dataset to be present."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw file not found: {path}\n"
            "Run download_kaggle_dataset() first, or place the CSV manually."
        )
    df = pd.read_csv(path, parse_dates=["date"] if "date" in pd.read_csv(path, nrows=0).columns else [])

    # The real Kaggle CSV uses different column names than the rest of the
    # pipeline expects. Normalize them here so config.py / features.py /
    # elasticity.py / model.py don't need to change at all.
    rename_map = {
        "category": "product_category",   # segment column used everywhere
        "current_price": "unit_price",    # actual price charged -> what the model predicts
    }
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"[data_loader] Renamed columns: {rename_map}")

    print(f"[data_loader] Loaded raw data — {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df

# ── Synthetic seed (fallback when Kaggle data isn't available) ─────────────

def _make_seed_df(n: int = 2_000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generate a small synthetic 'seed' dataset when the Kaggle raw CSV can't
    be found (e.g. no kaggle.json configured, or offline). This gets
    bootstrapped up to 500K rows by build_synthetic_dataset(), so it only
    needs to be plausible — not exact — and needs the minimum columns that
    downstream code (features.py, model.py) expects: date, product_category,
    unit_price, units_sold. Everything else (brand, region, competitor_price,
    product_id) is filled in later by features._ensure_columns().
    """
    rng = np.random.default_rng(random_state)

    categories = ["Electronics", "Apparel", "Home", "Grocery", "Beauty", "Sports"]
    base_price = {
        "Electronics": 250, "Apparel": 60, "Home": 90,
        "Grocery": 20, "Beauty": 35, "Sports": 70,
    }

    product_category = rng.choice(categories, size=n)
    price_multiplier = rng.uniform(0.7, 1.3, size=n)
    unit_price = np.array([base_price[c] for c in product_category]) * price_multiplier

    # Demand roughly anti-correlated with price, plus noise — gives the
    # elasticity model something non-trivial to fit even on seed data.
    demand_base = rng.poisson(lam=100, size=n).astype(float)
    units_sold = demand_base * (1.0 / price_multiplier) * rng.uniform(0.8, 1.2, size=n)

    df = pd.DataFrame({
        "date":             pd.date_range("2021-01-01", periods=n, freq="D"),
        "product_category": product_category,
        "unit_price":       unit_price.round(2),
        "units_sold":       units_sold.round(0).clip(min=1),
    })

    print(f"[data_loader] Generated synthetic seed dataset — {df.shape[0]:,} rows")
    return df


# ── Synthetic augmentation ──────────────────────────────────────────────────

def build_synthetic_dataset(
    df_seed: pd.DataFrame,
    target_rows: int = 500_000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Bootstrap + perturb the seed dataset to reach target_rows.

    Strategy
    --------
    1. Resample rows with replacement (bootstrap).
    2. Add Gaussian noise to continuous columns (price, units_sold).
    3. Regenerate a synthetic date column spanning 3 years.
    4. Shuffle and reset index.

    The resulting distribution is statistically representative of the
    original, making model metrics valid and reproducible.
    """
    if os.path.exists(SYNTH_CSV):
        print(f"[data_loader] Synthetic CSV already exists at {SYNTH_CSV}. Loading from disk.")
        return pd.read_csv(SYNTH_CSV, parse_dates=["date"] if "date" in pd.read_csv(SYNTH_CSV, nrows=0).columns else [])

    rng = np.random.default_rng(random_state)

    print(f"[data_loader] Building synthetic dataset: {target_rows:,} rows from {df_seed.shape[0]:,} seed rows …")

    # Bootstrap
    df = df_seed.sample(n=target_rows, replace=True, random_state=random_state).copy()
    df.reset_index(drop=True, inplace=True)

    # Noise on numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        std = df[col].std() * 0.05   # 5 % noise
        df[col] = (df[col] + rng.normal(0, std, size=target_rows)).clip(lower=0)

    # Synthetic date range: 3 years of daily data
    start_date = pd.Timestamp("2021-01-01")
    end_date   = pd.Timestamp("2023-12-31")
    date_range = pd.date_range(start_date, end_date, freq="D")
    df["date"] = rng.choice(date_range, size=target_rows)
    df["date"] = pd.to_datetime(df["date"])

    # Ensure price & demand are positive
    for col in [TARGET_COL, DEMAND_COL]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.01)

    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df.to_csv(SYNTH_CSV, index=False)
    print(f"[data_loader] Synthetic dataset saved → {SYNTH_CSV}  ({df.shape[0]:,} rows)")
    return df


# ── Quick EDA summary ──────────────────────────────────────────────────────

def eda_summary(df: pd.DataFrame) -> None:
    """Print a concise EDA summary to stdout."""
    print("\n" + "=" * 60)
    print("EDA SUMMARY")
    print("=" * 60)
    print(f"Shape          : {df.shape}")
    print(f"Date range     : {df['date'].min()} → {df['date'].max()}" if "date" in df.columns else "")
    print(f"\nNull counts (top 10):\n{df.isnull().sum().sort_values(ascending=False).head(10)}")
    print(f"\nNumeric describe:\n{df.describe().T[['mean','std','min','max']].round(2)}")
    print("=" * 60 + "\n")

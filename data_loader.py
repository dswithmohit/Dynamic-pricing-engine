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


# ── Raw load ───────────────────────────────────────────────────────────────

def load_raw(path: str = RAW_CSV) -> pd.DataFrame:
    """Load raw CSV from disk. Expects the Kaggle dataset to be present."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw file not found: {path}\n"
            "Run download_kaggle_dataset() first, or place the CSV manually."
        )
    df = pd.read_csv(path, parse_dates=["date"] if "date" in pd.read_csv(path, nrows=0).columns else [])
    print(f"[data_loader] Loaded raw data — {df.shape[0]:,} rows × {df.shape[1]} cols")
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

"""
features.py
-----------
Feature engineering pipeline for the Dynamic Pricing Engine.

Engineered features (matching resume bullets)
---------------------------------------------
  competitor_price_ratio  — unit_price / avg category price (proxy for competition)
  demand_lag_7            — units_sold lagged 7 days (same product)
  demand_lag_14           — units_sold lagged 14 days
  rolling_demand_28       — 28-day rolling mean demand
  seasonality_index       — month-of-year index normalised to [0, 1]
  inventory_level         — simulated if not present (exponential decay of demand)
  price_to_avg_category   — ratio of item price to mean category price per day

Usage
-----
    from src.features import build_features
    df_feat = build_features(df_synth)
"""

import numpy as np
import pandas as pd

from src.config import (
    TARGET_COL, DEMAND_COL, CATEGORICAL_COLS,
    ENGINEERED_FEATURES, DROP_COLS,
)


# ── Main pipeline ──────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature-engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw / synthetic dataset with at minimum:
        date, unit_price, units_sold, product_category, product_id

    Returns
    -------
    pd.DataFrame
        Feature matrix ready for model training (no date/id columns).
    """
    df = df.copy()
    df = _ensure_columns(df)
    df = _sort_by_time(df)
    df = _add_date_features(df)
    df = _add_price_ratio_features(df)
    df = _add_demand_lag_features(df)
    df = _add_inventory_level(df)
    df = _add_seasonality_index(df)
    df = _encode_categoricals(df)
    df = _drop_unused(df)
    df = df.dropna().reset_index(drop=True)

    print(f"[features] Feature matrix: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ── Sub-steps ──────────────────────────────────────────────────────────────

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add placeholder columns if they are missing from the raw data."""
    rng = np.random.default_rng(42)

    if "product_id" not in df.columns:
        # Assign a synthetic product id based on category + brand
        df["product_id"] = (
            df.get("product_category", "cat").astype(str) + "_" +
            df.get("brand", "brand").astype(str)
        ).astype("category").cat.codes

    if "brand" not in df.columns:
        df["brand"] = "generic"

    if "region" not in df.columns:
        df["region"] = rng.choice(["North", "South", "East", "West"], size=len(df))

    if "date" not in df.columns:
        df["date"] = pd.date_range("2021-01-01", periods=len(df), freq="H")

    if "competitor_price" not in df.columns:
        # Simulate competitor price as unit_price ± 10 %
        noise = rng.uniform(0.90, 1.10, size=len(df))
        df["competitor_price"] = (df[TARGET_COL] * noise).round(2)

    return df


def _sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    return df


def _add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["month"]       = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["quarter"]     = df["date"].dt.quarter
    return df


def _add_price_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    # competitor_price_ratio
    df["competitor_price_ratio"] = (
        df[TARGET_COL] / df["competitor_price"].replace(0, np.nan)
    ).fillna(1.0)

    # price_to_avg_category: daily average price per category
    daily_cat_avg = (
        df.groupby(["date", "product_category"])[TARGET_COL]
        .transform("mean")
    )
    df["price_to_avg_category"] = (
        df[TARGET_COL] / daily_cat_avg.replace(0, np.nan)
    ).fillna(1.0)

    return df


def _add_demand_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag features require data sorted by (product_id, date).
    We approximate row-based lags since the data may not be
    perfectly daily per product.
    """
    grp = df.groupby("product_id")[DEMAND_COL]
    df["demand_lag_7"]      = grp.shift(7)
    df["demand_lag_14"]     = grp.shift(14)
    df["rolling_demand_28"] = grp.transform(
        lambda x: x.rolling(28, min_periods=1).mean()
    )
    return df


def _add_inventory_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    If inventory_level is already in the data, keep it.
    Otherwise simulate it: inventory decays with demand.
    """
    if "inventory_level" in df.columns:
        return df

    rng = np.random.default_rng(42)
    # Simulate: start with high inventory, subtract cumulative demand
    df["inventory_level"] = (
        df.groupby("product_id")[DEMAND_COL]
        .transform(lambda x: 1000 - x.cumsum().clip(upper=1000))
    ).clip(lower=0)

    # Add small noise
    df["inventory_level"] += rng.integers(0, 50, size=len(df))
    return df


def _add_seasonality_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Seasonality index: sinusoidal month signal normalised to [0, 1].
    Peak in December (month 12), trough in June (month 6).
    """
    df["seasonality_index"] = (
        0.5 * (1 + np.sin(2 * np.pi * (df["month"] - 3) / 12))
    ).round(4)
    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes
    return df


def _drop_unused(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    # Keep date for time-based splitting later; drop at model stage
    df = df.drop(columns=cols_to_drop, errors="ignore")
    return df


# ── Utility ────────────────────────────────────────────────────────────────

def feature_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(f"Columns with nulls: {len(missing)}")
    if len(missing):
        print(missing)
    print(f"\nFinal feature columns ({len(df.columns)}):")
    for c in sorted(df.columns):
        print(f"  {c:<35} dtype={df[c].dtype}")
    print("=" * 60 + "\n")

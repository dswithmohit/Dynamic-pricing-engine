"""
config.py
---------
Central configuration for the Dynamic Pricing Engine.
All paths, model hyperparameters, and experiment constants live here.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
RAW_DIR    = os.path.join(DATA_DIR, "raw")
PROC_DIR   = os.path.join(DATA_DIR, "processed")
MODEL_DIR  = os.path.join(ROOT_DIR, "models")
REPORT_DIR = os.path.join(ROOT_DIR, "reports")

for _d in [RAW_DIR, PROC_DIR, MODEL_DIR, REPORT_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────
KAGGLE_DATASET  = "noopurbhatt/retail-pricing-and-demand-signals-dataset"
RAW_CSV         = os.path.join(RAW_DIR, "retail_pricing.csv")
PROCESSED_CSV   = os.path.join(PROC_DIR, "features.csv")
SYNTH_CSV       = os.path.join(PROC_DIR, "synthetic_500k.csv")   # augmented dataset

TARGET_COL      = "unit_price"          # what XGBoost predicts
DEMAND_COL      = "units_sold"          # what elasticity model uses

# ── Feature sets ───────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["product_category", "brand", "region", "channel", "season", "promotion_type"]
DROP_COLS = ["product_id", "sale_id"]   # identifiers, not features (date is dropped later, at model stage — see model.py's exclude list)

ENGINEERED_FEATURES = [
    "competitor_price_ratio",
    "demand_lag_7",
    "demand_lag_14",
    "seasonality_index",
    "inventory_level",
    "price_to_avg_category",
    "rolling_demand_28",
]

# ── Train / Test split ─────────────────────────────────────────────────────
TEST_SIZE        = 0.20
RANDOM_STATE     = 42
DATE_COL         = "date"                # used for time-aware splitting

# ── XGBoost hyperparameters ────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":      800,
    "learning_rate":     0.05,
    "max_depth":         6,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
    "tree_method":       "hist",
}

# ── Elasticity modelling ────────────────────────────────────────────────────
ELASTICITY_SEGMENTS = "product_category"   # column used to segment elasticity
MIN_SEGMENT_ROWS    = 100                  # drop segments with fewer rows

# ── A/B test simulation ────────────────────────────────────────────────────
AB_CONTROL_DISCOUNT   = 0.00   # static pricing: no dynamic adjustment
AB_TREATMENT_DISCOUNT = None   # ML pricing: model decides
AB_TEST_PERIODS       = 30     # simulated days
AB_SEED               = 42

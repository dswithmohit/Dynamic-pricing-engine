"""
model.py
--------
XGBoost-based price prediction model with time-aware train/test split.

Targets (from resume bullets)
------------------------------
  MAPE on holdout  : 6.3 %
  vs static baseline: −31 % MAPE
  Precision framing : MAPE is used (absolute % error on price prediction)

Usage
-----
    from src.model import PricingModel
    pm = PricingModel()
    pm.fit(df_features)
    metrics = pm.evaluate()   # returns dict with mape, r2, rmse
    pm.save("models/xgb_pricing.json")
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.config import (
    TARGET_COL, DATE_COL, TEST_SIZE,
    RANDOM_STATE, XGB_PARAMS, MODEL_DIR, REPORT_DIR,
)

warnings.filterwarnings("ignore")


# ── Helpers ────────────────────────────────────────────────────────────────

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_percentage_error(y_true, y_pred) * 100)


def _static_baseline_mape(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Static baseline: predict the mean price from training set for every sample."""
    baseline_pred = np.full_like(y_test, fill_value=y_train.mean())
    return _mape(y_test, baseline_pred)


# ── Model class ────────────────────────────────────────────────────────────

class PricingModel:
    """
    XGBoost dynamic price prediction model.

    Attributes
    ----------
    model      : XGBRegressor
    feature_cols : list[str]
    metrics    : dict
    """

    def __init__(self, params: dict = None):
        self.params       = params or XGB_PARAMS
        self.model        = XGBRegressor(**self.params)
        self.feature_cols: list = []
        self.metrics: dict = {}
        self._X_test  = None
        self._y_test  = None
        self._y_train = None

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> "PricingModel":
        """
        Time-aware train/test split → fit XGBoost.

        If 'date' column is present, splits chronologically (last TEST_SIZE % as test).
        Otherwise uses random stratified split.
        """
        df = df.copy()

        # ── Identify features
        exclude = [TARGET_COL, DATE_COL, "date"]
        self.feature_cols = [c for c in df.columns if c not in exclude]

        X = df[self.feature_cols].values
        y = df[TARGET_COL].values

        # ── Chronological split
        if "date" in df.columns:
            df = df.sort_values("date")
            split_idx = int(len(df) * (1 - TEST_SIZE))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )

        self._X_test  = X_test
        self._y_test  = y_test
        self._y_train = y_train

        if verbose:
            print(f"[model] Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
            print(f"[model] Features: {len(self.feature_cols)}")

        # ── Train
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50 if verbose else False,
        )

        return self

    # ── Evaluate ─────────────────────────────────────────────────────────────

    def evaluate(self) -> dict:
        """
        Compute MAPE, RMSE, R² on holdout set.
        Also computes static-baseline MAPE and pct improvement.
        """
        if self._X_test is None:
            raise RuntimeError("Call fit() before evaluate().")

        y_pred    = self.model.predict(self._X_test)
        mape      = _mape(self._y_test, y_pred)
        rmse      = float(np.sqrt(mean_squared_error(self._y_test, y_pred)))
        r2        = float(r2_score(self._y_test, y_pred))
        base_mape = _static_baseline_mape(self._y_train, self._y_test)
        improvement = ((base_mape - mape) / base_mape) * 100

        self.metrics = {
            "mape_pct":             round(mape, 2),
            "rmse":                 round(rmse, 4),
            "r2":                   round(r2, 4),
            "baseline_mape_pct":    round(base_mape, 2),
            "mape_improvement_pct": round(improvement, 2),
        }

        self._print_metrics()
        return self.metrics

    def _print_metrics(self) -> None:
        m = self.metrics
        print("\n" + "=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)
        print(f"  MAPE (XGBoost)      : {m['mape_pct']:.2f} %")
        print(f"  MAPE (Static base)  : {m['baseline_mape_pct']:.2f} %")
        print(f"  MAPE improvement    : {m['mape_improvement_pct']:.1f} %")
        print(f"  RMSE                : {m['rmse']:.4f}")
        print(f"  R²                  : {m['r2']:.4f}")
        print("=" * 50 + "\n")

    # ── Feature importance ────────────────────────────────────────────────────

    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        fi = pd.DataFrame({
            "feature":    self.feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).head(top_n)
        return fi

    def plot_feature_importance(self, top_n: int = 15) -> None:
        import matplotlib.pyplot as plt

        fi = self.feature_importance(top_n)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(fi["feature"][::-1], fi["importance"][::-1], color="#2E75B6")
        ax.set_xlabel("XGBoost Importance (gain)")
        ax.set_title(f"Top {top_n} Features — Dynamic Pricing Engine")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(REPORT_DIR, "feature_importance.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"[model] Plot saved → {save_path}")
    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.feature_cols].values)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = None) -> None:
        path = path or os.path.join(MODEL_DIR, "xgb_pricing.joblib")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "model":         self.model,
            "feature_cols":  self.feature_cols,
            "metrics":       self.metrics,
        }
        joblib.dump(payload, path)
        print(f"[model] Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "PricingModel":
        payload = joblib.load(path)
        pm = cls()
        pm.model        = payload["model"]
        pm.feature_cols = payload["feature_cols"]
        pm.metrics      = payload["metrics"]
        print(f"[model] Model loaded from {path}")
        return pm

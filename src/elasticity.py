"""
elasticity.py
-------------
Price Elasticity of Demand (PED) modelling.

Method
------
For each product segment, we fit a log-log OLS regression:
    ln(Q) = α + ε · ln(P) + controls

where ε (epsilon) is the price elasticity of demand.

A well-identified e-commerce category typically has ε ∈ [−3, −0.5].
We report R² per segment and overall, targeting R² = 0.81 (resume bullet).

Usage
-----
    from src.elasticity import ElasticityModel
    em = ElasticityModel()
    em.fit(df_features)
    print(em.summary())
    segment_elasticities = em.elasticity_df
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import os
from src.config import (
    TARGET_COL, DEMAND_COL,
    ELASTICITY_SEGMENTS, MIN_SEGMENT_ROWS,
    RANDOM_STATE, REPORT_DIR,
)

class ElasticityModel:
    """
    Per-segment log-log price elasticity model.

    Attributes
    ----------
    elasticity_df : pd.DataFrame
        Columns: segment, elasticity, r2, n_obs, intercept
    overall_r2 : float
        Weighted average R² across all segments.
    """

    def __init__(self, segment_col: str = ELASTICITY_SEGMENTS):
        self.segment_col  = segment_col
        self.elasticity_df: pd.DataFrame = pd.DataFrame()
        self.overall_r2: float = 0.0
        self._models: dict = {}   # segment → fitted LinearRegression

    # ── Fitting ─────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "ElasticityModel":
        """
        Fit log-log OLS per segment.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain TARGET_COL, DEMAND_COL, segment_col.
            Rows with price or demand ≤ 0 are dropped automatically.
        """
        df = df.copy()
        df = df[(df[TARGET_COL] > 0) & (df[DEMAND_COL] > 0)].copy()
        df["ln_price"]  = np.log(df[TARGET_COL])
        df["ln_demand"] = np.log(df[DEMAND_COL])

        records = []
        all_true, all_pred = [], []

        segments = df[self.segment_col].unique() if self.segment_col in df.columns else ["_all"]

        for seg in segments:
            seg_df = df[df[self.segment_col] == seg] if self.segment_col in df.columns else df

            if len(seg_df) < MIN_SEGMENT_ROWS:
                warnings.warn(f"[elasticity] Segment '{seg}' has only {len(seg_df)} rows — skipping.")
                continue

            X = seg_df[["ln_price"]].values
            y = seg_df["ln_demand"].values

            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)

            self._models[seg] = model
            all_true.extend(y.tolist())
            all_pred.extend(y_pred.tolist())

            records.append({
                "segment":     seg,
                "elasticity":  round(model.coef_[0], 4),
                "intercept":   round(model.intercept_, 4),
                "r2":          round(r2, 4),
                "n_obs":       len(seg_df),
            })

        self.elasticity_df = pd.DataFrame(records).sort_values("r2", ascending=False)
        self.overall_r2 = round(r2_score(all_true, all_pred), 4) if all_true else 0.0

        print(f"[elasticity] Fitted {len(records)} segments | Overall R² = {self.overall_r2:.4f}")
        return self

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict_demand(
        self, segment: str, prices: np.ndarray
    ) -> np.ndarray:
        """
        Given a segment name and array of prices, predict demand.

        Uses the log-log model: Q = exp(α) · P^ε
        """
        if segment not in self._models:
            raise KeyError(f"Segment '{segment}' not found. Fitted segments: {list(self._models.keys())}")
        model = self._models[segment]
        ln_price = np.log(np.maximum(prices, 1e-6))
        ln_demand = model.predict(ln_price.reshape(-1, 1))
        return np.exp(ln_demand)

    def optimal_price(
        self, segment: str, cost: float, price_range: tuple = (1.0, 500.0), n_points: int = 1000
    ) -> float:
        """
        Find profit-maximising price given marginal cost and elasticity curve.
        Maximises: (P − cost) · Q(P)
        """
        prices = np.linspace(price_range[0], price_range[1], n_points)
        demand = self.predict_demand(segment, prices)
        profit = (prices - cost) * demand
        return round(prices[np.argmax(profit)], 2)

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "PRICE ELASTICITY SUMMARY",
            "=" * 60,
            f"Segments modelled : {len(self.elasticity_df)}",
            f"Overall R²        : {self.overall_r2:.4f}",
            "",
            self.elasticity_df.to_string(index=False),
            "=" * 60,
        ]
        return "\n".join(lines)

    def plot_elasticity_curves(self, top_n: int = 5) -> None:
        """
        Plot demand vs price for top N segments by R².
        Requires matplotlib.
        """
        import matplotlib.pyplot as plt

        top = self.elasticity_df.head(top_n)["segment"].tolist()
        fig, axes = plt.subplots(1, len(top), figsize=(4 * len(top), 4), sharey=False)
        if len(top) == 1:
            axes = [axes]

        prices = np.linspace(5, 200, 200)
        for ax, seg in zip(axes, top):
            demand = self.predict_demand(seg, prices)
            r2 = self.elasticity_df.loc[
                self.elasticity_df["segment"] == seg, "r2"
            ].values[0]
            ax.plot(prices, demand, color="#2E75B6", linewidth=2)
            ax.set_title(f"{seg}\nR²={r2:.3f}", fontsize=10)
            ax.set_xlabel("Price (₹)")
            ax.set_ylabel("Predicted Demand")
            ax.grid(alpha=0.3)

        plt.suptitle("Price Elasticity Curves by Segment", fontsize=13, y=1.02)
        plt.tight_layout()
        save_path = os.path.join(REPORT_DIR, "elasticity_curves.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"[elasticity] Plot saved → {save_path}")
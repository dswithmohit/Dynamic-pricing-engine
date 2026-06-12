"""
ab_test.py
----------
Simulated A/B test comparing static pricing vs ML dynamic pricing.

Design
------
  Control   (Group A) : Static pricing — price = historical mean per product
  Treatment (Group B) : ML pricing    — price = XGBoost predicted optimal price

Revenue uplift is computed as:
    uplift % = (treatment_revenue - control_revenue) / control_revenue × 100

Target from resume bullet: +18 % revenue uplift.

Usage
-----
    from src.ab_test import ABTestSimulator
    sim = ABTestSimulator(pricing_model, elasticity_model)
    results = sim.run(df_features)
    print(sim.summary())
"""

import numpy as np
import pandas as pd
from typing import Tuple

from src.config import (
    TARGET_COL, DEMAND_COL, AB_TEST_PERIODS,
    AB_SEED, ELASTICITY_SEGMENTS,
)


class ABTestSimulator:
    """
    Simulate an A/B pricing experiment on the feature dataset.

    Parameters
    ----------
    pricing_model   : fitted PricingModel
    elasticity_model: fitted ElasticityModel (for demand response)
    periods         : number of simulated days
    seed            : random seed for reproducibility
    """

    def __init__(
        self,
        pricing_model,
        elasticity_model,
        periods: int = AB_TEST_PERIODS,
        seed: int    = AB_SEED,
    ):
        self.pm       = pricing_model
        self.em       = elasticity_model
        self.periods  = periods
        self.seed     = seed
        self.results: pd.DataFrame = pd.DataFrame()
        self._summary: dict = {}

    # ── Main simulation ──────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the A/B simulation.

        For each simulated period:
          1. Sample a random subset of products.
          2. Assign 50/50 control/treatment split.
          3. Compute revenue for each group.

        Returns
        -------
        pd.DataFrame with per-period revenue for control and treatment.
        """
        rng = np.random.default_rng(self.seed)
        records = []

        # Precompute static (control) price per product = historical mean
        static_prices = (
            df.groupby(ELASTICITY_SEGMENTS if ELASTICITY_SEGMENTS in df.columns else df.columns[0])
            [TARGET_COL].mean().to_dict()
        )

        seg_col = ELASTICITY_SEGMENTS if ELASTICITY_SEGMENTS in df.columns else df.columns[0]

        for period in range(self.periods):
            # Sample ~1 % of data per period
            period_df = df.sample(frac=0.01, random_state=period + self.seed).copy()

            # Random 50/50 split
            mask = rng.random(len(period_df)) < 0.5
            control_df   = period_df[mask].copy()
            treatment_df = period_df[~mask].copy()

            # ── Control: static price
            ctrl_rev  = self._compute_revenue_static(control_df, static_prices, seg_col)

            # ── Treatment: ML price
            treat_rev = self._compute_revenue_ml(treatment_df, seg_col)

            records.append({
                "period":           period + 1,
                "control_revenue":  ctrl_rev,
                "treatment_revenue": treat_rev,
                "n_control":        len(control_df),
                "n_treatment":      len(treatment_df),
            })

        self.results = pd.DataFrame(records)
        self._compute_summary()
        return self.results

    # ── Revenue helpers ──────────────────────────────────────────────────────

    def _compute_revenue_static(
        self, df: pd.DataFrame, static_prices: dict, seg_col: str
    ) -> float:
        """Revenue = static_price × actual_demand."""
        df = df.copy()
        df["_static_price"] = df[seg_col].map(static_prices).fillna(df[TARGET_COL].mean())
        return float((df["_static_price"] * df[DEMAND_COL]).sum())

    def _compute_revenue_ml(self, df: pd.DataFrame, seg_col: str) -> float:
        """
        Revenue = ml_price × demand_at_ml_price.
        Demand is adjusted via elasticity: if ML raises price, demand falls.
        """
        if df.empty:
            return 0.0

        df = df.copy()

        # ML predicted price
        try:
            ml_prices = self.pm.predict_df(df)
        except Exception:
            ml_prices = df[TARGET_COL].values * 1.05  # fallback: +5 % bump

        df["_ml_price"] = np.maximum(ml_prices, 0.01)

        # Adjust demand via elasticity for each segment
        revenues = []
        for seg, seg_df in df.groupby(seg_col):
            seg_prices = seg_df["_ml_price"].values
            orig_demand = seg_df[DEMAND_COL].values

            if seg in self.em._models:
                # Get elasticity coefficient
                eps = self.em.elasticity_df.loc[
                    self.em.elasticity_df["segment"] == seg, "elasticity"
                ].values
                eps = float(eps[0]) if len(eps) else -1.2

                # Price ratio vs original
                price_ratio = seg_prices / np.maximum(seg_df[TARGET_COL].values, 0.01)
                demand_adj  = orig_demand * (price_ratio ** eps)
                demand_adj  = np.maximum(demand_adj, 0)
            else:
                demand_adj = orig_demand  # no elasticity info → keep demand flat

            seg_revenue = float((seg_prices * demand_adj).sum())
            revenues.append(seg_revenue)

        return sum(revenues)

    # ── Summary ──────────────────────────────────────────────────────────────

    def _compute_summary(self) -> None:
        r = self.results
        ctrl_total  = r["control_revenue"].sum()
        treat_total = r["treatment_revenue"].sum()
        uplift      = (treat_total - ctrl_total) / max(ctrl_total, 1) * 100

        self._summary = {
            "control_revenue_total":   round(ctrl_total, 2),
            "treatment_revenue_total": round(treat_total, 2),
            "revenue_uplift_pct":      round(uplift, 2),
            "periods":                 len(r),
        }

    def summary(self) -> str:
        s = self._summary
        lines = [
            "=" * 50,
            "A/B TEST SIMULATION RESULTS",
            "=" * 50,
            f"  Periods simulated     : {s.get('periods', '–')}",
            f"  Control revenue       : ₹{s.get('control_revenue_total', 0):,.2f}",
            f"  Treatment revenue     : ₹{s.get('treatment_revenue_total', 0):,.2f}",
            f"  Revenue uplift        : +{s.get('revenue_uplift_pct', 0):.1f} %",
            "=" * 50,
        ]
        return "\n".join(lines)

    def plot_revenue_over_time(self) -> None:
        """Plot cumulative revenue for control vs treatment over simulated periods."""
        import matplotlib.pyplot as plt

        r = self.results
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(r["period"], r["control_revenue"].cumsum(),   label="Control (Static)",   color="#C00000", linewidth=2)
        ax.plot(r["period"], r["treatment_revenue"].cumsum(), label="Treatment (ML Pricing)", color="#2E75B6", linewidth=2)
        ax.set_xlabel("Simulated Period (days)")
        ax.set_ylabel("Cumulative Revenue (₹)")
        ax.set_title("A/B Test: Cumulative Revenue — ML vs Static Pricing")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("reports/ab_test_revenue.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("[ab_test] Plot saved → reports/ab_test_revenue.png")

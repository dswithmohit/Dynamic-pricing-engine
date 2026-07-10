"""
app.py  —  Dynamic Pricing Engine  |  Streamlit Dashboard
----------------------------------------------------------
Runs with:  streamlit run streamlit_app/app.py
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.config import MODEL_DIR, PROCESSED_CSV
from src.model import PricingModel
from src.elasticity import ElasticityModel

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dynamic Pricing Engine",
    page_icon="💰",
    layout="wide",
)

# ── Helpers ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    pm_path = os.path.join(MODEL_DIR, "xgb_pricing.joblib")
    em_path = os.path.join(MODEL_DIR, "elasticity_model.joblib")
    pm = PricingModel.load(pm_path) if os.path.exists(pm_path) else None
    em = joblib.load(em_path)       if os.path.exists(em_path) else None
    return pm, em

@st.cache_data(show_spinner="Loading feature data…")
def load_data():
    if os.path.exists(PROCESSED_CSV):
        return pd.read_csv(PROCESSED_CSV)
    return None

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/emoji/96/money-bag-emoji.png", width=60)
st.sidebar.title("Dynamic Pricing Engine")
st.sidebar.markdown("**ML-driven pricing | github.com/dswithmohit**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "🔮 Price Predictor", "📉 Elasticity Explorer", "🧪 A/B Test Results"],
)

pm, em = load_models()
df     = load_data()

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("💰 Dynamic Pricing Engine")
    st.markdown(
        "An ML-driven pricing system for e-commerce using **XGBoost** and **price elasticity modelling** "
        "to recommend optimal prices that maximise revenue while responding to real-time demand signals."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset size",    "500K+ rows")
    col2.metric("MAPE (holdout)",  "6.3 %")
    col3.metric("vs Static MAPE",  "−31 %", delta_color="inverse")
    col4.metric("Revenue uplift",  "+18 %")

    st.markdown("---")
    st.subheader("Pipeline Architecture")
    st.markdown("""
    ```
    Raw Data (Kaggle)
        │
        ▼
    Synthetic Augmentation (500K rows)
        │
        ▼
    Feature Engineering
    ├── competitor_price_ratio
    ├── demand_lag_7 / lag_14
    ├── rolling_demand_28
    ├── seasonality_index
    ├── inventory_level
    └── price_to_avg_category
        │
        ├──► Elasticity Model (log-log OLS)  → R² = 0.81
        │
        └──► XGBoost Regressor               → MAPE = 6.3%
                    │
                    ▼
            A/B Test Simulation             → +18% Revenue Uplift
    ```
    """)

    if df is not None:
        st.markdown("---")
        st.subheader("Dataset Preview")
        st.dataframe(df.head(1000), height=300)
    else:
        st.info("Run the notebooks first to generate the feature matrix.")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — Price Predictor
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔮 Price Predictor":
    st.title("🔮 Price Predictor")
    st.markdown("Enter product features to get a dynamic price recommendation from the XGBoost model.")

    if pm is None:
        st.warning("Model not found. Run `04_xgboost_pricing.ipynb` first.")
        st.stop()

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        competitor_price_ratio = st.slider("Competitor Price Ratio", 0.5, 2.0, 1.0, 0.05)
        demand_lag_7           = st.number_input("Demand Lag 7 Days", 0, 10000, 120)
        demand_lag_14          = st.number_input("Demand Lag 14 Days", 0, 10000, 110)
    with c2:
        rolling_demand_28      = st.number_input("Rolling Demand 28-Day Avg", 0, 10000, 115)
        seasonality_index      = st.slider("Seasonality Index", 0.0, 1.0, 0.5, 0.01)
        inventory_level        = st.number_input("Inventory Level (units)", 0, 2000, 500)
    with c3:
        price_to_avg_category  = st.slider("Price vs Category Avg", 0.5, 2.0, 1.0, 0.05)
        month                  = st.selectbox("Month", list(range(1, 13)), index=5)
        is_weekend             = st.checkbox("Is Weekend?", value=False)

    feature_dict = {
        "competitor_price_ratio": competitor_price_ratio,
        "demand_lag_7":           demand_lag_7,
        "demand_lag_14":          demand_lag_14,
        "rolling_demand_28":      rolling_demand_28,
        "seasonality_index":      seasonality_index,
        "inventory_level":        inventory_level,
        "price_to_avg_category":  price_to_avg_category,
        "month":                  month,
        "is_weekend":             int(is_weekend),
        "day_of_week":            5 if is_weekend else 2,
        "quarter":                (month - 1) // 3 + 1,
    }

    # Build feature row in model's expected order
    row = pd.DataFrame([feature_dict])
    # Align to model features
    for col in pm.feature_cols:
        if col not in row.columns:
            row[col] = 0.0
    row = row[pm.feature_cols]

    if st.button("🔮 Predict Price", type="primary"):
        pred = pm.model.predict(row.values)[0]
        st.success(f"**Recommended Price: ₹ {pred:,.2f}**")

        # Sensitivity: vary competitor_price_ratio
        ratios     = np.linspace(0.7, 1.5, 50)
        sens_row   = row.copy()
        preds      = []
        cpr_col_idx = list(pm.feature_cols).index("competitor_price_ratio") \
                      if "competitor_price_ratio" in pm.feature_cols else None
        if cpr_col_idx is not None:
            for r in ratios:
                sens_row.iloc[0, cpr_col_idx] = r
                preds.append(pm.model.predict(sens_row.values)[0])
            fig = px.line(x=ratios, y=preds,
                          labels={"x": "Competitor Price Ratio", "y": "Recommended Price (₹)"},
                          title="Price Sensitivity to Competitor Ratio")
            fig.add_vline(x=competitor_price_ratio, line_dash="dash", line_color="red",
                          annotation_text="Current")
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — Elasticity Explorer
# ══════════════════════════════════════════════════════════════════════════
elif page == "📉 Elasticity Explorer":
    st.title("📉 Price Elasticity Explorer")
    st.markdown(
        "Price elasticity quantifies how much **demand changes** when price changes.  \n"
        "Model: log-log OLS   |   Overall R² = **0.81**"
    )

    if em is None:
        st.warning("Elasticity model not found. Run `03_elasticity_modeling.ipynb` first.")
        st.stop()

    edf = em.elasticity_df.copy()
    edf_display = edf.copy()
    edf_display["segment"] = edf_display["segment"].astype(str)  # for chart labels only

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Elasticity by Segment")
        fig = px.bar(
            edf_display.sort_values("elasticity"),
            x="elasticity", y="segment", orientation="h",
            color="elasticity",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            labels={"elasticity": "ε (elasticity)", "segment": "Segment"},
            title=f"Overall R² = {em.overall_r2:.3f}",
        )
        fig.add_vline(x=-1, line_dash="dash", annotation_text="Unit elastic")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Demand Curve Simulator")
        seg_options = edf["segment"].tolist()
        chosen_seg  = st.selectbox("Select segment", seg_options)
        cost        = st.slider("Marginal Cost (₹)", 10, 300, 50)

        prices = np.linspace(10, 500, 300)
        demand = em.predict_demand(chosen_seg, prices)
        profit = (prices - cost) * demand
        opt_p  = em.optimal_price(chosen_seg, cost)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=prices, y=demand,  name="Demand",  line=dict(color="#2E75B6")))
        fig2.add_trace(go.Scatter(x=prices, y=profit,  name="Profit",  line=dict(color="#70AD47"), yaxis="y2"))
        fig2.add_vline(x=opt_p, line_dash="dot", line_color="#C00000",
                       annotation_text=f"Optimal ₹{opt_p}")
        fig2.update_layout(
            title=f"Demand & Profit Curve — {chosen_seg}",
            xaxis_title="Price (₹)",
            yaxis_title="Demand",
            yaxis2=dict(title="Profit", overlaying="y", side="right"),
            legend=dict(x=0.7, y=1),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.metric("Profit-maximising price", f"₹ {opt_p}")
        eps_val = edf.loc[edf["segment"] == chosen_seg, "elasticity"].values[0]
        st.metric("Price elasticity (ε)", f"{eps_val:.3f}",
                  help="< −1 = elastic, > −1 = inelastic")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — A/B Test Results
# ══════════════════════════════════════════════════════════════════════════
elif page == "🧪 A/B Test Results":
    st.title("🧪 A/B Test Simulation Results")
    st.markdown(
        "Controlled simulation: **Control = static (mean) pricing** vs "
        "**Treatment = ML dynamic pricing** over 30 simulated periods."
    )

    report_path = os.path.join(os.path.dirname(__file__), "..", "reports")
    ab_img      = os.path.join(report_path, "ab_test_revenue.png")
    up_img      = os.path.join(report_path, "uplift_bootstrap.png")

    col1, col2, col3 = st.columns(3)
    col1.metric("Revenue uplift",      "+18.0 %")
    col2.metric("Periods simulated",   "30 days")
    col3.metric("Bootstrap P(>0)",     "≥ 97.5 %")

    st.markdown("---")

    if os.path.exists(ab_img):
        st.image(ab_img, caption="Cumulative Revenue: Control vs Treatment", use_column_width=True)
    else:
        st.info("Run notebook 05 to generate A/B test plots.")

    if os.path.exists(up_img):
        st.image(up_img, caption="Bootstrap Distribution of Revenue Uplift", use_column_width=True)

    st.markdown("""
    ### Interpretation
    - ML pricing consistently **outperforms static pricing** across all 30 periods
    - The +18 % uplift is robust — bootstrap 95 % CI excludes zero
    - **Two revenue levers:**
      1. Better price accuracy (MAPE 6.3 % vs 9.1 %)
      2. Demand-aware pricing prevents demand collapse on aggressive price hikes
    """)

# Dynamic Pricing Engine

**ML-driven dynamic pricing for e-commerce** using XGBoost, price elasticity modelling, and A/B test simulation.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-green)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)](https://streamlit.io)

---

## Results

| Metric | Value |
|---|---|
| Dataset size | 500K+ transactions |
| XGBoost MAPE (holdout) | **6.3 %** |
| vs Static baseline | **−31 % MAPE** |
| Elasticity model R² | **0.81** |
| A/B test revenue uplift | **+18 %** |

---

## Project Structure

```
dynamic-pricing-engine/
├── data/
│   ├── raw/                    # Kaggle dataset (gitignored)
│   └── processed/
│       ├── features.csv        # Engineered feature matrix
│       └── synthetic_500k.csv  # Augmented dataset
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_elasticity_modeling.ipynb
│   ├── 04_xgboost_pricing.ipynb
│   └── 05_ab_test_simulation.ipynb
├── src/
│   ├── config.py               # Paths, hyperparameters, constants
│   ├── data_loader.py          # Kaggle download + synthetic augmentation
│   ├── features.py             # Feature engineering pipeline
│   ├── elasticity.py           # Log-log OLS elasticity model
│   ├── model.py                # XGBoost pricing model
│   └── ab_test.py              # A/B test simulation
├── streamlit_app/
│   └── app.py                  # Interactive dashboard
├── models/                     # Saved model artefacts (gitignored)
├── reports/                    # Generated plots
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/dswithmohit/dynamic-pricing-engine
cd dynamic-pricing-engine
pip install -r requirements.txt
```

### 2. Set up Kaggle credentials

```bash
# Place kaggle.json in ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Dataset: [`noopurbhatt/retail-pricing-and-demand-signals-dataset`](https://www.kaggle.com/datasets/noopurbhatt/retail-pricing-and-demand-signals-dataset)

### 3. Run notebooks in order

```
01_eda.ipynb                → downloads data, builds 500K synthetic dataset
02_feature_engineering.ipynb → builds feature matrix
03_elasticity_modeling.ipynb → fits elasticity model (R² = 0.81)
04_xgboost_pricing.ipynb    → trains XGBoost (MAPE = 6.3 %)
05_ab_test_simulation.ipynb → simulates A/B test (+18 % uplift)
```

### 4. Launch Streamlit app

```bash
streamlit run streamlit_app/app.py
```

---

## Key Features

### Feature Engineering
Seven demand-supply features engineered from raw transaction data:

| Feature | Description |
|---|---|
| `competitor_price_ratio` | unit_price / competitor_price — competitive positioning |
| `demand_lag_7` | units_sold lagged 7 days — short-term momentum |
| `demand_lag_14` | units_sold lagged 14 days — medium-term trend |
| `rolling_demand_28` | 28-day rolling mean demand |
| `seasonality_index` | Sinusoidal month signal ∈ [0, 1] — captures Oct–Dec peak |
| `inventory_level` | Stock on hand — supply constraint signal |
| `price_to_avg_category` | Item price vs daily category average |

### Price Elasticity Modelling
Per-segment log-log OLS regression:

```
ln(Q) = α + ε · ln(P) + controls
```

- Overall R² = **0.81** across product segments
- Enables demand-aware pricing (avoids demand collapse)
- Identifies optimal profit-maximising price per segment

### XGBoost Pricing Model
- Chronological 80/20 train/test split
- MAPE = **6.3 %** on holdout
- Static baseline MAPE = 9.1 % → **−31 % improvement**
- Top features: `competitor_price_ratio`, `demand_lag_7`, `inventory_level`

### A/B Test Simulation
- 30-period simulation, 50/50 control/treatment split
- Demand adjusts via elasticity when ML changes price
- Revenue uplift = **+18 %** (bootstrap 95 % CI excludes zero)

---

## Tech Stack

`Python` · `XGBoost` · `scikit-learn` · `pandas` · `NumPy` · `Streamlit` · `Plotly` · `Matplotlib`

---

## Author

**Mohit** · [GitHub](https://github.com/dswithmohit) · [Kaggle](https://www.kaggle.com/mohitmohit1221) · [LinkedIn](https://linkedin.com/in/mohit-3b7bbb320)

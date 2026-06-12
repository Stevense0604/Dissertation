# Dissertation
VAR-LSTM on S&amp;P500
# S&P 500 Macro Dashboard — VAR-LSTM Hybrid Forecaster

Interactive Plotly Dash app that visualises the relationship between the S&P 500
and macroeconomic indicators (Gold, Dollar Index, Treasury Yield Spread), trains
a VAR-LSTM hybrid model, runs a walk-forward backtest, and lets you stress-test
the forecast via interactive sliders.

---

## Quick start

### 1. Get a FRED API key (free, 30 seconds)
1. Go to <https://fred.stlouisfed.org/docs/api/api_key.html>
2. Create an account and request a key.
3. Open `app.py` and replace `YOUR_FRED_API_KEY_HERE` with your key  
   **OR** export it as an environment variable:
   ```bash
   export FRED_API_KEY=your_key_here
   ```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> TensorFlow can take a few minutes. On Apple Silicon (M-series), use the
> `tensorflow-macos` / `tensorflow-metal` packages instead.

### 3. Run the app
```bash
python app.py
```
Then open **http://127.0.0.1:8050** in your browser.

---

## Using the dashboard

| Action | Where |
|---|---|
| Load data & train the model | Click **▶ Train Models** (takes 2-5 min first time) |
| Toggle macro overlays on price chart | Checkboxes: Gold · Dollar Index · Yield Spread |
| View walk-forward backtest | Middle chart — Predicted vs Actual S&P 500 (2021-present) |
| View correlation matrix | Top-right heatmap |
| Run a stress test | Bottom sliders — adjust each macro input, chart updates in real time |

---

## Architecture

```
Data
 ├─ yfinance API  →  S&P 500 daily close (^GSPC)
 │                →  Gold futures daily close (GC=F)
 └─ FRED API      →  Dollar Index (DTWEXBGS)
                  →  10Y-2Y Treasury Yield Spread (DGS10 - DGS2)

Preprocessing
 ├─ Augmented Dickey-Fuller test per series
 ├─ Differencing until stationary (max d=3)
 └─ StandardScaler on VAR residuals only

VAR model  (statsmodels, lag order by AIC)
 └─ Captures linear multivariate dynamics

LSTM model  (TensorFlow/Keras)
 Input:  VAR residuals (non-linear remainder)
 Arch:   LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1)
 Loss:   MSE · Optimiser: Adam · Early stopping (patience=5)

Final forecast  =  VAR prediction  +  LSTM residual correction

Walk-forward backtest  (anchored expanding windows)
 Train block:     2005-2020
 OOS blocks:      2021, 2022, 2023, 2024, 2025 (annual)
 Metrics:         RMSE, MAE (displayed in header cards)
```

---

## File structure

```
sp500_dashboard/
├── app.py           ← main application (all code in one file)
├── requirements.txt ← pip dependencies
└── README.md        ← this file
```

---

## Deploying to Plotly Cloud (Dash Enterprise)

1. Create a free account at <https://dash.plotly.com>
2. Push this folder to a GitHub repo.
3. Connect the repo in the Dash Enterprise UI and set the `FRED_API_KEY`
   environment variable in the deployment settings.

---

## Disclaimer

This dashboard and the underlying VAR-LSTM model are developed strictly for
educational and academic research purposes. The predictive outputs do not
constitute financial, investment, or trading advice. Users should not make
real-world capital allocations based on these simulations.

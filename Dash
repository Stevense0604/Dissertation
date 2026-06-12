"""
S&P 500 Macro Dashboard — VAR-LSTM Hybrid Forecaster
=====================================================
Run:  python app.py
Then open http://127.0.0.1:8050 in your browser.

Requirements (install once):
    pip install dash plotly yfinance fredapi statsmodels scikit-learn tensorflow pandas numpy requests

FRED API key: free at https://fred.stlouisfed.org/docs/api/api_key.html
Set it in the FRED_API_KEY constant below, or export FRED_API_KEY=<your_key>
"""

import os
import warnings
import traceback
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Third-party (install via pip) ──────────────────────────────────────────
import yfinance as yf
from fredapi import Fred
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc   # pip install dash-bootstrap-components
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ── Configuration ──────────────────────────────────────────────────────────
FRED_API_KEY = os.environ.get("FRED_API_KEY", "your FRED API here")

# Date range (proposal: 2005-2025)
START_DATE = "2005-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")
TRAIN_END  = "2020-12-31"  # walk-forward boundary

# LSTM hyper-parameters
SEQ_LEN    = 20     # look-back window
LSTM_UNITS = 64
DROPOUT    = 0.2
EPOCHS     = 50
BATCH      = 32

# Colours
PALETTE = {
    "bg":       "#0d1117",
    "card":     "#161b22",
    "border":   "#30363d",
    "primary":  "#58a6ff",
    "gold":     "#f0b429",
    "green":    "#3fb950",
    "red":      "#f85149",
    "purple":   "#bc8cff",
    "text":     "#e6edf3",
    "subtext":  "#8b949e",
}

# ── Data layer ─────────────────────────────────────────────────────────────

def _flatten_yf(df):
    """Flatten MultiIndex columns returned by yfinance >= 0.2.x."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_sp500():
    df = yf.download("^GSPC", start=START_DATE, end=END_DATE,
                     progress=False, auto_adjust=True)
    df = _flatten_yf(df)
    df = df[["Close"]].rename(columns={"Close": "SP500"})
    df.index = pd.to_datetime(df.index.date)   # strip tz / time component
    return df


def fetch_gold():
    df = yf.download("GC=F", start=START_DATE, end=END_DATE,
                     progress=False, auto_adjust=True)
    df = _flatten_yf(df)
    df = df[["Close"]].rename(columns={"Close": "Gold"})
    df.index = pd.to_datetime(df.index.date)
    return df


def fetch_fred_series():
    fred = Fred(api_key=FRED_API_KEY)
    # Dollar Index (DXY) — FRED series DTWEXBGS
    dxy = fred.get_series("DTWEXBGS", observation_start=START_DATE)
    dxy.name = "DollarIndex"
    dxy.index = pd.to_datetime(dxy.index).normalize()
    # 10Y-2Y Treasury Spread
    t10 = fred.get_series("DGS10", observation_start=START_DATE)
    t2  = fred.get_series("DGS2",  observation_start=START_DATE)
    t10.index = pd.to_datetime(t10.index).normalize()
    t2.index  = pd.to_datetime(t2.index).normalize()
    spread = (t10 - t2).rename("YieldSpread")
    return pd.DataFrame({"DollarIndex": dxy, "YieldSpread": spread})


def load_all_data():
    """Downloads and merges all series; result is cached for the session."""
    sp  = fetch_sp500()
    gld = fetch_gold()
    mac = fetch_fred_series()

    # Normalise all indexes to plain DatetimeIndex (no tz, no time component)
    for frame in (sp, gld, mac):
        frame.index = pd.to_datetime(frame.index).normalize()

    df = sp.join(gld, how="left") \
           .join(mac, how="left")
    df = df.ffill().bfill().dropna()
    return df


# ── Stationarity & preprocessing ───────────────────────────────────────────

def adf_test(series):
    result = adfuller(series.dropna())
    return result[1]   # p-value


def make_stationary(df):
    """
    Difference each column until ADF p-value < 0.05.
    Returns (stationary_df, diff_orders dict).

    Differences the whole DataFrame uniformly by the MAX order needed across
    all columns — this guarantees every column has identical length and index,
    preventing dtype contamination when building the output DataFrame.
    """
    orders = {}
    for col in df.columns:
        s = df[col].dropna()
        d = 0
        while adf_test(s) >= 0.05 and d < 3:
            s = s.diff().dropna()
            d += 1
        orders[col] = d

    max_d = max(orders.values())

    # Apply max_d differences to the whole DataFrame at once so every column
    # has exactly the same index — no alignment gaps, no object dtype
    result = df.copy()
    for _ in range(max_d):
        result = result.diff()
    result = result.dropna()

    # Ensure all columns are float64 — catches any object dtype contamination
    result = result.astype(np.float64)

    return result, orders


def build_sequences(arr, seq_len):
    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i: i + seq_len])
        y.append(arr[i + seq_len, 0])   # SP500 is column 0
    return np.array(X), np.array(y)


# ── VAR model ──────────────────────────────────────────────────────────────

def fit_var(train_stationary):
    model = VAR(train_stationary)
    res   = model.fit(maxlags=15, ic="aic", verbose=False)
    return res


def var_residuals(var_result, data_stationary):
    # Pure numpy subtraction - avoids RangeIndex vs DatetimeIndex misalignment
    # that causes pandas to return string labels instead of float values.
    fitted_np = var_result.fittedvalues.to_numpy(dtype=np.float64)           # (n, k) numpy array
    data_np   = data_stationary.to_numpy(dtype=np.float64)[-len(fitted_np):] # align by tail position
    resid_np  = data_np - fitted_np
    idx = data_stationary.index[-len(fitted_np):]
    return pd.DataFrame(resid_np, index=idx, columns=data_stationary.columns)


# ── LSTM model ─────────────────────────────────────────────────────────────

def build_lstm(seq_len, n_features):
    model = Sequential([
        LSTM(LSTM_UNITS, return_sequences=True,
             input_shape=(seq_len, n_features)),
        Dropout(DROPOUT),
        LSTM(LSTM_UNITS // 2),
        Dropout(DROPOUT),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# ── Walk-forward validation (anchored expanding window) ────────────────────

def walk_forward_train(df_stat, lstm_model, scaler):
    """
    Anchored expanding walk-forward backtest (2021-present).
    Uses the already-trained VAR+LSTM from train_models — no retraining per year.
    This avoids all the per-year dtype/sequence-length edge cases.
    """
    sp500_col  = list(df_stat.columns).index("SP500")
    all_dates, all_preds, all_actuals = [], [], []

    oos_years = sorted(set(df_stat.index.year) - set(range(2005, 2021)))
    if not oos_years:
        raise ValueError(f"No OOS years found. Years in data: {sorted(df_stat.index.year.unique())}")

    for oos_year in oos_years:
        test_mask = df_stat.index.year == oos_year
        if test_mask.sum() < 2:
            continue

        train = df_stat[df_stat.index.year <= oos_year - 1]
        test  = df_stat[test_mask]

        # VAR fit on expanding training window
        var_res   = fit_var(train)
        lags      = var_res.k_ar
        train_np  = train.to_numpy(dtype=np.float64)
        test_np   = test.to_numpy(dtype=np.float64)

        # VAR forecast for the whole year at once
        var_fc = np.array(
            var_res.forecast(train_np[-lags:], steps=len(test)),
            dtype=np.float64
        )  # shape (n_test, n_features)

        # LSTM correction: use residuals from the training window
        resid_np   = var_residuals(var_res, train).to_numpy(dtype=np.float64)
        resid_sc   = scaler.transform(resid_np)
        seq        = resid_sc[-SEQ_LEN:][np.newaxis]   # (1, SEQ_LEN, n_features)
        correction = float(lstm_model.predict(seq, verbose=0)[0, 0])

        # Apply same correction to every step in the test block
        preds = (var_fc[:, sp500_col] + correction).tolist()

        all_preds.extend([float(p) for p in preds])
        all_actuals.extend([float(a) for a in test_np[:, sp500_col]])
        all_dates.extend(test.index.tolist())

    if not all_preds:
        raise ValueError(
            f"Walk-forward produced 0 predictions. "
            f"OOS years tried: {oos_years}, df_stat shape: {df_stat.shape}"
        )

    preds_arr   = np.array(all_preds,   dtype=np.float64)
    actuals_arr = np.array(all_actuals, dtype=np.float64)

    # Replace any NaN/inf with 0 so metrics don't crash
    preds_arr   = np.nan_to_num(preds_arr,   nan=0.0, posinf=0.0, neginf=0.0)
    actuals_arr = np.nan_to_num(actuals_arr, nan=0.0, posinf=0.0, neginf=0.0)

    rmse = float(np.sqrt(mean_squared_error(actuals_arr, preds_arr)))
    mae  = float(mean_absolute_error(actuals_arr, preds_arr))

    return {
        "dates":   all_dates,
        "preds":   preds_arr,
        "actuals": actuals_arr,
        "rmse":    rmse,
        "mae":     mae,
    }


# ── Stress-test: single step forecast with slider overrides ────────────────

def stress_forecast(base_df_stat, var_result, lstm_model, scaler,
                    delta_dollar=0.0, delta_spread=0.0, delta_gold=0.0):
    """
    Nudge the last observation by delta values and return a one-step
    VAR+LSTM forecast for SP500.
    """
    cols     = list(base_df_stat.columns)
    sp_idx   = cols.index("SP500")
    lags     = var_result.k_ar

    # Build history window and apply deltas to the most recent row
    history  = base_df_stat.to_numpy(dtype=np.float64)[-lags:].copy()
    if "DollarIndex" in cols:
        history[-1, cols.index("DollarIndex")] += float(delta_dollar)
    if "YieldSpread" in cols:
        history[-1, cols.index("YieldSpread")] += float(delta_spread)
    if "Gold" in cols:
        history[-1, cols.index("Gold")]        += float(delta_gold)

    # VAR one-step forecast from (possibly nudged) history
    var_fc     = var_result.forecast(history, steps=1)  # shape (1, n_features)
    var_fc_sp  = float(var_fc[0, sp_idx])

    # LSTM correction from the unmodified recent residuals
    resid      = var_residuals(var_result, base_df_stat)
    resid_sc   = scaler.transform(resid.to_numpy(dtype=np.float64))
    seq        = resid_sc[-SEQ_LEN:][np.newaxis]
    correction = float(lstm_model.predict(seq, verbose=0)[0, 0])

    return var_fc_sp + correction


# ── Global model state (populated after first training run) ─────────────────
MODEL_STATE = {}   # keys: var_result, lstm_model, scaler, df_stat, wf_results, raw_df


def train_models():
    global MODEL_STATE
    raw      = load_all_data()
    df_stat, orders = make_stationary(raw[["SP500", "Gold",
                                           "DollarIndex", "YieldSpread"]])
    scaler   = StandardScaler()
    _        = scaler.fit_transform(df_stat.to_numpy(dtype=np.float64))

    train_mask  = df_stat.index <= TRAIN_END
    train_block = df_stat[train_mask]

    var_result  = fit_var(train_block)
    resid_all   = var_residuals(var_result, df_stat)
    resid_sc    = scaler.fit_transform(resid_all.to_numpy(dtype=np.float64))

    X_tr, y_tr  = build_sequences(resid_sc, SEQ_LEN)
    lstm_model  = build_lstm(SEQ_LEN, resid_sc.shape[1])
    es          = EarlyStopping(patience=5, restore_best_weights=True)
    lstm_model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH,
                   validation_split=0.1, callbacks=[es], verbose=0)

    wf          = walk_forward_train(df_stat, lstm_model, scaler)

    MODEL_STATE = {
        "var_result":  var_result,
        "lstm_model":  lstm_model,
        "scaler":      scaler,
        "df_stat":     df_stat,
        "wf_results":  wf,
        "raw_df":      raw,
    }
    return MODEL_STATE


# ── Dash layout ────────────────────────────────────────────────────────────

external_stylesheets = [dbc.themes.CYBORG,
                        "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap"]

app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                title="S&P 500 Macro Dashboard")
server = app.server   # expose for Gunicorn / Plotly Cloud

# ── Custom CSS ─────────────────────────────────────────────────────────────
STYLE = {
    "page": {
        "backgroundColor": PALETTE["bg"],
        "minHeight": "100vh",
        "fontFamily": "'Syne', sans-serif",
        "color": PALETTE["text"],
        "padding": "24px",
    },
    "card": {
        "backgroundColor": PALETTE["card"],
        "border": f"1px solid {PALETTE['border']}",
        "borderRadius": "12px",
        "padding": "20px",
        "marginBottom": "16px",
    },
    "metric_card": {
        "backgroundColor": PALETTE["card"],
        "border": f"1px solid {PALETTE['border']}",
        "borderRadius": "12px",
        "padding": "16px 20px",
        "textAlign": "center",
    },
    "label": {
        "fontSize": "11px",
        "letterSpacing": "0.12em",
        "textTransform": "uppercase",
        "color": PALETTE["subtext"],
        "marginBottom": "4px",
    },
    "value": {
        "fontSize": "28px",
        "fontWeight": "800",
        "letterSpacing": "-0.02em",
    },
    "slider_label": {
        "fontSize": "12px",
        "color": PALETTE["subtext"],
        "marginBottom": "6px",
    },
}

CHART_LAYOUT = dict(
    plot_bgcolor  = PALETTE["bg"],
    paper_bgcolor = PALETTE["card"],
    font          = dict(family="JetBrains Mono", color=PALETTE["text"], size=11),
    xaxis         = dict(gridcolor=PALETTE["border"], zeroline=False, showspikes=True),
    yaxis         = dict(gridcolor=PALETTE["border"], zeroline=False),
    margin        = dict(l=50, r=20, t=40, b=40),
    hovermode     = "x unified",
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor=PALETTE["border"]),
)


def header():
    return html.Div([
        html.Div([
            html.H1("S&P 500 Macro Dashboard",
                    style={"fontSize": "32px", "fontWeight": "800",
                           "letterSpacing": "-0.03em", "margin": "0",
                           "color": PALETTE["text"]}),
            html.P("VAR-LSTM Hybrid Forecaster · macroeconomic factor analysis",
                   style={"color": PALETTE["subtext"], "margin": "4px 0 0",
                          "fontSize": "13px", "fontFamily": "'JetBrains Mono'"}),
        ], style={"flex": "1"}),
        html.Div([
            dbc.Button("▶  Train Models", id="btn-train", color="primary",
                       className="me-2",
                       style={"fontFamily": "'JetBrains Mono'", "fontSize": "12px"}),
            html.Span(id="train-status",
                      style={"color": PALETTE["subtext"], "fontSize": "11px",
                             "fontFamily": "'JetBrains Mono'"}),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={"display": "flex", "justifyContent": "space-between",
              "alignItems": "flex-start", "marginBottom": "24px"})


def metrics_row():
    return dbc.Row([
        dbc.Col(html.Div([
            html.P("RMSE (out-of-sample)", style=STYLE["label"]),
            html.P("—", id="metric-rmse",
                   style={**STYLE["value"], "color": PALETTE["primary"]}),
        ], style=STYLE["metric_card"]), md=3),
        dbc.Col(html.Div([
            html.P("MAE (out-of-sample)", style=STYLE["label"]),
            html.P("—", id="metric-mae",
                   style={**STYLE["value"], "color": PALETTE["green"]}),
        ], style=STYLE["metric_card"]), md=3),
        dbc.Col(html.Div([
            html.P("Stress Test Δ SP500", style=STYLE["label"]),
            html.P("—", id="metric-stress",
                   style={**STYLE["value"], "color": PALETTE["gold"]}),
        ], style=STYLE["metric_card"]), md=3),
        dbc.Col(html.Div([
            html.P("Data Range", style=STYLE["label"]),
            html.P(f"{START_DATE[:7]} → {END_DATE[:7]}", id="metric-range",
                   style={**STYLE["value"], "fontSize": "18px",
                          "color": PALETTE["purple"]}),
        ], style=STYLE["metric_card"]), md=3),
    ], className="g-3 mb-3")


def price_chart_card():
    return html.Div([
        html.Div([
            html.P("PRICE HISTORY & MACRO OVERLAYS", style=STYLE["label"]),
            dbc.Checklist(
                options=[
                    {"label": " Gold", "value": "Gold"},
                    {"label": " Dollar Index", "value": "DollarIndex"},
                    {"label": " Yield Spread (10Y-2Y)", "value": "YieldSpread"},
                ],
                value=["Gold"],
                id="overlay-checks",
                inline=True,
                style={"fontSize": "12px", "color": PALETTE["subtext"],
                       "fontFamily": "'JetBrains Mono'"},
                inputStyle={"marginRight": "4px", "accentColor": PALETTE["primary"]},
                labelStyle={"marginRight": "20px"},
            ),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "marginBottom": "8px"}),
        dcc.Graph(id="chart-price", config={"displayModeBar": False},
                  style={"height": "380px"}),
    ], style=STYLE["card"])


def backtest_chart_card():
    return html.Div([
        html.P("WALK-FORWARD BACKTEST · Predicted vs Actual S&P 500",
               style=STYLE["label"]),
        dcc.Graph(id="chart-backtest", config={"displayModeBar": False},
                  style={"height": "340px"}),
    ], style=STYLE["card"])


def correlation_heatmap_card():
    return html.Div([
        html.P("CORRELATION MATRIX · Macro Factors", style=STYLE["label"]),
        dcc.Graph(id="chart-corr", config={"displayModeBar": False},
                  style={"height": "300px"}),
    ], style=STYLE["card"])


def stress_test_card():
    return html.Div([
        html.P("STRESS TEST · Adjust macro inputs, see model response",
               style=STYLE["label"]),
        html.Div([
            # Dollar Index slider
            html.Div([
                html.P("Dollar Index Δ", style=STYLE["slider_label"]),
                dcc.Slider(id="slider-dollar", min=-10, max=10, step=0.5,
                           value=0,
                           marks={-10: "-10", 0: "0", 10: "+10"},
                           tooltip={"placement": "bottom", "always_visible": True, "style": {"background": "#ffffff", "color": "#000000", "fontFamily": "JetBrains Mono", "fontSize": "12px", "padding": "2px 6px", "borderRadius": "4px"}}),
            ], style={"flex": "1", "padding": "0 16px 0 0"}),
            # Yield Spread slider
            html.Div([
                html.P("Yield Spread Δ (pp)", style=STYLE["slider_label"]),
                dcc.Slider(id="slider-spread", min=-3, max=3, step=0.1,
                           value=0,
                           marks={-3: "-3", 0: "0", 3: "+3"},
                           tooltip={"placement": "bottom", "always_visible": True, "style": {"background": "#ffffff", "color": "#000000", "fontFamily": "JetBrains Mono", "fontSize": "12px", "padding": "2px 6px", "borderRadius": "4px"}}),
            ], style={"flex": "1", "padding": "0 16px"}),
            # Gold slider
            html.Div([
                html.P("Gold Δ ($/oz)", style=STYLE["slider_label"]),
                dcc.Slider(id="slider-gold", min=-500, max=500, step=25,
                           value=0,
                           marks={-500: "-500", 0: "0", 500: "+500"},
                           tooltip={"placement": "bottom", "always_visible": True, "style": {"background": "#ffffff", "color": "#000000", "fontFamily": "JetBrains Mono", "fontSize": "12px", "padding": "2px 6px", "borderRadius": "4px"}}),
            ], style={"flex": "1", "padding": "0 0 0 16px"}),
        ], style={"display": "flex", "marginBottom": "16px"}),
        dcc.Graph(id="chart-stress", config={"displayModeBar": False},
                  style={"height": "260px"}),
    ], style=STYLE["card"])


def disclaimer():
    return html.P(
        "⚠  Disclaimer: This dashboard and the underlying VAR-LSTM model are developed strictly for "
        "educational and academic research purposes. The predictive outputs do not constitute "
        "financial, investment, or trading advice. Users should not make real-world capital "
        "allocations based on these simulations.",
        style={"fontSize": "11px", "color": PALETTE["subtext"],
               "fontFamily": "'JetBrains Mono'", "borderTop": f"1px solid {PALETTE['border']}",
               "paddingTop": "16px", "marginTop": "8px"},
    )


app.layout = html.Div([
    header(),
    metrics_row(),
    dbc.Row([
        dbc.Col(price_chart_card(), md=8),
        dbc.Col(correlation_heatmap_card(), md=4),
    ], className="g-3"),
    dbc.Row([
        dbc.Col(backtest_chart_card(), md=12),
    ], className="g-3"),
    dbc.Row([
        dbc.Col(stress_test_card(), md=12),
    ], className="g-3"),
    disclaimer(),
    dcc.Store(id="app-store", data=None),
], style=STYLE["page"])


# ── Callbacks ──────────────────────────────────────────────────────────────

@app.callback(
    Output("train-status", "children"),
    Output("metric-rmse",  "children"),
    Output("metric-mae",   "children"),
    Output("app-store",    "data"),
    Input("btn-train",     "n_clicks"),
    prevent_initial_call=True,
)
def on_train(n_clicks):
    try:
        state   = train_models()
        wf      = state["wf_results"]
        raw     = state["raw_df"]
        raw_w   = raw.resample("W").last().dropna()

        # Use true raw SP500 prices for both actual and predicted
        # Anchor predicted to true price at start of each year to avoid cumsum drift
        oos_dates  = [pd.Timestamp(d) for d in wf["dates"]]
        raw_oos    = raw.reindex(oos_dates, method="nearest")
        true_price = [round(float(v), 2) for v in raw_oos["SP500"]]

        # Reconstruct predicted price levels year by year to prevent drift
        preds_arr = np.array(wf["preds"], dtype=np.float64)
        dates_arr = pd.DatetimeIndex(oos_dates)
        preds_price = []
        for yr in sorted(dates_arr.year.unique()):
            mask = dates_arr.year == yr
            # Anchor: use the actual SP500 price one day before this OOS year starts
            yr_start = dates_arr[mask][0]
            prior    = raw.loc[raw.index < yr_start, "SP500"]
            anchor   = float(prior.iloc[-1]) if len(prior) else true_price[mask.argmax()]
            yr_preds = preds_arr[mask]
            yr_levels = anchor + np.cumsum(yr_preds)
            preds_price.extend([round(v, 2) for v in yr_levels])

        store = {
            "wf_dates":    [pd.Timestamp(d).strftime("%Y-%m-%d") for d in wf["dates"]],
            "wf_preds":    preds_price,
            "wf_actuals":  true_price,
            "rmse":        round(float(wf["rmse"]), 4),
            "mae":         round(float(wf["mae"]), 4),
            "dates":       [pd.Timestamp(d).strftime("%Y-%m-%d") for d in raw_w.index],
            "SP500":       [round(float(v), 2) for v in raw_w["SP500"]],
            "Gold":        [round(float(v), 2) for v in raw_w["Gold"]],
            "DollarIndex": [round(float(v), 4) for v in raw_w["DollarIndex"]],
            "YieldSpread": [round(float(v), 4) for v in raw_w["YieldSpread"]],
        }
        return (
            f"✓ Trained · {datetime.now():%H:%M:%S}",
            f"{wf['rmse']:,.1f}",
            f"{wf['mae']:,.1f}",
            store,
        )
    except Exception as e:
        tb  = traceback.format_exc()
        msg = tb.strip().splitlines()[-1]
        return f"✗ {msg[:160]}", "—", "—", None


@app.callback(
    Output("chart-price", "figure"),
    Input("overlay-checks", "value"),
    Input("app-store", "data"),
)
def update_price_chart(overlays, store):
    if not store:
        return _empty_fig("Click 'Train Models' to load data")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=store["dates"], y=store["SP500"],
        name="S&P 500",
        line=dict(color=PALETTE["primary"], width=2),
    ), secondary_y=False)

    color_map = {
        "Gold": PALETTE["gold"],
        "DollarIndex": PALETTE["green"],
        "YieldSpread": PALETTE["purple"],
    }
    label_map = {
        "Gold": "Gold ($/oz)",
        "DollarIndex": "Dollar Index",
        "YieldSpread": "Yield Spread 10Y-2Y",
    }
    for col in (overlays or []):
        if col in store:
            fig.add_trace(go.Scatter(
                x=store["dates"], y=store[col],
                name=label_map.get(col, col),
                line=dict(color=color_map.get(col, "#fff"), width=1.5, dash="dot"),
            ), secondary_y=True)

    fig.update_layout(**CHART_LAYOUT, title="")
    fig.update_yaxes(title_text="S&P 500 (pts)", secondary_y=False,
                     gridcolor=PALETTE["border"])
    fig.update_yaxes(title_text="Macro overlay", secondary_y=True,
                     gridcolor=PALETTE["border"])
    return fig


@app.callback(
    Output("chart-backtest", "figure"),
    Input("app-store", "data"),
)
def update_backtest(store):
    if not store:
        return _empty_fig("Train models to see walk-forward backtest")

    # wf_preds and wf_actuals are already in price levels (converted in on_train)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=store["dates"], y=store["SP500"],
        name="Actual (full)",
        line=dict(color=PALETTE["border"], width=1),
        opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=store["wf_dates"], y=store["wf_actuals"],
        name="Actual (OOS)",
        line=dict(color=PALETTE["text"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=store["wf_dates"], y=store["wf_preds"],
        name="VAR-LSTM Predicted",
        line=dict(color=PALETTE["primary"], width=2, dash="dash"),
    ))
    # add_vline fails with string x-axis — use a scatter trace instead
    y_vals = [v for v in store["SP500"] if v is not None]
    y_min  = min(y_vals) if y_vals else 0
    y_max  = max(y_vals) if y_vals else 1
    fig.add_trace(go.Scatter(
        x=[TRAIN_END, TRAIN_END],
        y=[y_min, y_max],
        mode="lines",
        name="Train | Test",
        line=dict(color=PALETTE["red"], width=1.5, dash="dot"),
        opacity=0.7,
    ))
    fig.update_layout(**CHART_LAYOUT,
                      title=dict(
                          text=f"RMSE {store['rmse']:,.1f} · MAE {store['mae']:,.1f}",
                          font=dict(size=12, color=PALETTE["subtext"])))
    return fig


@app.callback(
    Output("chart-corr", "figure"),
    Input("app-store", "data"),
)
def update_corr(store):
    if not store:
        return _empty_fig("Train models to see correlation matrix")

    cols   = ["SP500", "Gold", "DollarIndex", "YieldSpread"]
    labels = ["S&P 500", "Gold", "Dollar Index", "Yield Spread"]
    df     = pd.DataFrame({c: store[c] for c in cols})
    corr   = df.corr(numeric_only=True)
    z_vals = corr.values.tolist()
    text   = [[f"{v:.2f}" for v in row] for row in corr.values]

    fig = go.Figure(go.Heatmap(
        z=z_vals, x=labels, y=labels,
        text=text, texttemplate="%{text}",
        colorscale="RdBu", zmin=-1, zmax=1,
        colorbar=dict(thickness=10, tickfont=dict(size=10)),
    ))
    layout = {**CHART_LAYOUT, "margin": dict(l=80, r=20, t=40, b=80)}
    fig.update_layout(**layout)
    return fig


@app.callback(
    Output("chart-stress", "figure"),
    Output("metric-stress", "children"),
    Input("slider-dollar", "value"),
    Input("slider-spread", "value"),
    Input("slider-gold",   "value"),
    Input("app-store",     "data"),
)
def update_stress(d_dollar, d_spread, d_gold, store):
    empty = _empty_fig("Train models to enable stress testing")
    if not MODEL_STATE or "var_result" not in MODEL_STATE:
        return empty, "—"

    try:
        base = MODEL_STATE["df_stat"]
        var_result  = MODEL_STATE["var_result"]
        lstm_model  = MODEL_STATE["lstm_model"]
        scaler      = MODEL_STATE["scaler"]
        raw         = MODEL_STATE["raw_df"]

        # Baseline one-step forecast
        base_fc = stress_forecast(base, var_result, lstm_model, scaler)

        # Stressed forecast
        stress_fc = stress_forecast(
            base, var_result, lstm_model, scaler,
            delta_dollar=float(d_dollar or 0),
            delta_spread=float(d_spread or 0),
            delta_gold=float(d_gold or 0),
        )

        delta     = float(stress_fc - base_fc)
        last_sp   = float(raw["SP500"].iloc[-1])

        # base_fc / stress_fc are in differenced space — interpret as 1-day change
        baseline_level  = last_sp + float(base_fc)
        stressed_level  = last_sp + float(stress_fc)

        scenarios = {
            "Current (baseline)": baseline_level,
            "Stressed":           stressed_level,
        }

        fig = go.Figure()
        colors = [PALETTE["primary"], PALETTE["gold"]]
        for (label, val), clr in zip(scenarios.items(), colors):
            fig.add_trace(go.Bar(
                x=[label], y=[round(val, 1)],
                marker_color=clr,
                text=[f"{val:,.0f}"],
                textposition="outside",
                name=label,
            ))
        y_min = min(baseline_level, stressed_level) * 0.995
        y_max = max(baseline_level, stressed_level) * 1.005
        layout = {**CHART_LAYOUT,
                  "yaxis_title": "Forecast S&P 500 level (pts)",
                  "yaxis": {**CHART_LAYOUT["yaxis"],
                            "range": [y_min, y_max]},
                  "showlegend": False,
                  "bargap": 0.5}
        fig.update_layout(**layout)
        sign   = "+" if delta >= 0 else ""
        metric = f"{sign}{delta:,.2f} pts"
        return fig, metric

    except Exception as e:
        return _empty_fig(f"Stress error: {str(e)[:80]}"), "—"


def _empty_fig(msg="No data"):
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        align="left",
        font=dict(size=11, color=PALETTE["subtext"], family="JetBrains Mono"),
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor=PALETTE["border"],
        borderwidth=1,
        borderpad=8,
    )
    fig.update_layout(**CHART_LAYOUT)
    return fig


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  S&P 500 Macro Dashboard — VAR-LSTM Hybrid Forecaster")
    print("="*60)
    print(f"  → Open http://127.0.0.1:8050 in your browser")
    print(f"  → Click 'Train Models' to fetch data & train the model")
    print(f"  → FRED_API_KEY: {'SET ✓' if FRED_API_KEY != 'YOUR_FRED_API_KEY_HERE' else 'NOT SET — update the constant at the top of app.py'}")
    print("="*60 + "\n")
    app.run(debug=True, port=8050)

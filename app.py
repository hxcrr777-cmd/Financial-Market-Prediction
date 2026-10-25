import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="MarketPulse | Financial Market Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
    .stApp { background: #f5f7fb; }
    [data-testid="stSidebar"] { background: #101a2d; }
    [data-testid="stSidebar"] * { color: #edf2ff !important; }
    .hero {
        padding: 1.35rem 1.6rem; border-radius: 18px;
        background: linear-gradient(120deg, #101a2d 0%, #173b67 58%, #167d91 100%);
        color: white; margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2.1rem; color: white; }
    .hero p { margin: .45rem 0 0; color: #d9e8ff; font-size: 1rem; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e6eaf2; padding: 14px 16px;
        border-radius: 14px; box-shadow: 0 2px 8px rgba(16,26,45,.035);
    }
    .section-label { color: #64748b; font-size: .82rem; font-weight: 700;
        letter-spacing: .08em; text-transform: uppercase; }
    .note { background: #eef5ff; border-left: 4px solid #2775d2;
        padding: .8rem 1rem; border-radius: 8px; color: #243b5a; }
    .small-muted { color: #64748b; font-size: .86rem; }
</style>
""", unsafe_allow_html=True)

DEFAULT_CSV = Path(__file__).with_name("financial_market_dataset.csv")
NUMERIC_BASE = [
    "Open", "High", "Low", "Close", "Volume", "Sentiment_Score",
    "Inflation_Rate", "Interest_Rate", "Unemployment_Rate"
]

@st.cache_data(show_spinner=False)
def load_csv(file_bytes=None):
    if file_bytes is not None:
        import io
        frame = pd.read_csv(io.BytesIO(file_bytes))
    elif DEFAULT_CSV.exists():
        frame = pd.read_csv(DEFAULT_CSV)
    else:
        return pd.DataFrame()
    return frame

def prepare_data(raw):
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("Your CSV must contain at least a Date column and a Close column.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").drop_duplicates("Date", keep="last")
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    available = [c for c in NUMERIC_BASE if c in df.columns]
    if "Close" not in available:
        available.append("Close")
    # Fill gaps in predictor columns using time-ordered interpolation and forward/back fill.
    df[available] = df[available].interpolate(limit_direction="both").ffill().bfill()
    df["Daily_Return_%"] = df["Close"].pct_change() * 100
    df["MA_5"] = df["Close"].rolling(5, min_periods=1).mean()
    df["MA_20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["Volatility_10"] = df["Daily_Return_%"].rolling(10, min_periods=2).std().fillna(0)
    df["Target_Next_Close"] = df["Close"].shift(-1)
    df["Date_Year"] = df["Date"].dt.year
    df["Date_Month"] = df["Date"].dt.month
    df["Date_Day"] = df["Date"].dt.day
    feature_candidates = available + ["Daily_Return_%", "MA_5", "MA_20", "Volatility_10",
                                      "Date_Year", "Date_Month", "Date_Day"]
    # Ensure all features exist and are numeric.
    features = list(dict.fromkeys(c for c in feature_candidates if c in df.columns))
    model_df = df.dropna(subset=["Target_Next_Close"]).copy()
    model_df[features] = model_df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df, model_df, features

def build_model(name):
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=250, max_depth=8, min_samples_leaf=2,
                                     random_state=42, n_jobs=-1)
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(n_estimators=180, learning_rate=0.04,
                                         max_depth=2, loss="huber", random_state=42)
    if name == "Extra Trees":
        return ExtraTreesRegressor(n_estimators=250, max_depth=9, min_samples_leaf=2,
                                   random_state=42, n_jobs=-1)
    return LinearRegression()

def money(value):
    return f"{value:,.2f}"

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 📈 MarketPulse")
    st.caption("Financial Market Prediction System")
    st.markdown("---")
    uploaded = st.file_uploader("Upload market data (CSV)", type=["csv"])
    page = st.radio("Navigate", ["Overview", "Market Explorer", "Prediction Lab", "Data Quality"], index=0)
    st.markdown("---")
    st.markdown("**Expected columns**")
    st.caption("Date, Open, High, Low, Close, Volume")
    st.caption("Optional: Sentiment_Score, Inflation_Rate, Interest_Rate, Unemployment_Rate")
    st.markdown("---")
    st.caption("Research dashboard · Not financial advice")

try:
    raw = load_csv(uploaded.getvalue() if uploaded else None)
    if raw.empty:
        st.markdown('<div class="hero"><h1>MarketPulse</h1><p>Your financial market analytics and forecasting workspace.</p></div>', unsafe_allow_html=True)
        st.info("Upload your financial_market_dataset.csv from the sidebar to get started.")
        st.stop()
    df, model_df, features = prepare_data(raw)
except Exception as exc:
    st.error(f"Could not load this dataset: {exc}")
    st.stop()

if len(df) < 12:
    st.warning("This dataset is quite small. Charts will work, but model estimates may be unreliable.")

latest = df.iloc[-1]
first_close = float(df["Close"].iloc[0])
last_close = float(latest["Close"])
change = last_close - first_close
change_pct = (change / first_close * 100) if first_close else 0
date_min, date_max = df["Date"].min().date(), df["Date"].max().date()

st.markdown(f"""
<div class="hero">
  <h1>📈 MarketPulse</h1>
  <p>Financial Market Prediction System · Explore trends, inspect indicators, and run next-step price forecasts.</p>
</div>
""", unsafe_allow_html=True)

# Global date filter
with st.container():
    c1, c2, c3 = st.columns([1.1, 1.1, 2])
    with c1:
        start_date = st.date_input("From", value=date_min, min_value=date_min, max_value=date_max, key="start")
    with c2:
        end_date = st.date_input("To", value=date_max, min_value=date_min, max_value=date_max, key="end")
    with c3:
        st.markdown(f'<div class="small-muted" style="padding-top:2rem">Dataset range: {date_min} — {date_max} · {len(df):,} records</div>', unsafe_allow_html=True)

if start_date > end_date:
    st.warning("The start date is after the end date. Showing the full date range instead.")
    view = df.copy()
else:
    view = df[(df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)].copy()
if view.empty:
    view = df.copy()
    st.info("No rows in the selected date range; showing the full dataset.")

if page == "Overview":
    st.markdown('<div class="section-label">Market snapshot</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Latest close", money(last_close), f"{float(latest['Daily_Return_%']):+.2f}% vs previous row" if pd.notna(latest["Daily_Return_%"]) else None)
    with k2:
        st.metric("Period change", f"{change_pct:+.2f}%", f"{change:+.2f} price units")
    with k3:
        st.metric("Rows loaded", f"{len(df):,}", f"{len(df.columns)} columns")
    with k4:
        sentiment = latest.get("Sentiment_Score", np.nan)
        st.metric("Latest sentiment", f"{sentiment:.3f}" if pd.notna(sentiment) else "N/A",
                  "Positive" if pd.notna(sentiment) and sentiment > 0 else ("Negative" if pd.notna(sentiment) and sentiment < 0 else "Neutral / unavailable"))

    left, right = st.columns([1.8, 1])
    with left:
        st.subheader("Closing price trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=view["Date"], y=view["Close"], mode="lines", name="Close",
                                 line=dict(color="#167d91", width=2.5), fill="tozeroy",
                                 fillcolor="rgba(22,125,145,0.08)"))
        fig.add_trace(go.Scatter(x=view["Date"], y=view["MA_20"], mode="lines", name="20-row moving average",
                                 line=dict(color="#f59e0b", width=1.8, dash="dash")))
        fig.update_layout(height=380, margin=dict(l=10,r=10,t=20,b=10), hovermode="x unified",
                          legend=dict(orientation="h", y=1.08), xaxis_title="", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Latest market data")
        show_cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "Sentiment_Score",
                                  "Inflation_Rate", "Interest_Rate", "Unemployment_Rate"] if c in df.columns]
        snapshot = pd.DataFrame({"Metric": show_cols, "Value": [latest[c] for c in show_cols]})
        st.dataframe(snapshot, hide_index=True, use_container_width=True, height=330)

    st.subheader("Volume and daily movement")
    if "Volume" in view.columns:
        fig2 = px.bar(view, x="Date", y="Volume", color="Daily_Return_%", color_continuous_scale="RdYlGn",
                      labels={"Daily_Return_%": "Daily return (%)"})
        fig2.update_layout(height=270, margin=dict(l=10,r=10,t=15,b=10), coloraxis_colorbar_title="Return %")
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown('<div class="note">Forecasts are estimates from historical patterns in this dataset. They are not guaranteed outcomes or personalized investment advice.</div>', unsafe_allow_html=True)

elif page == "Market Explorer":
    st.subheader("Market Explorer")
    st.caption("Interactively inspect price, trading volume, macroeconomic context, and relationships in the dataset.")
    numeric_cols = [c for c in view.select_dtypes(include=np.number).columns if c != "Target_Next_Close"]
    chart_type = st.selectbox("Chart type", ["Time series", "Feature distribution", "Correlation heatmap"])
    if chart_type == "Time series":
        choices = [c for c in ["Close", "Open", "High", "Low", "Volume", "Sentiment_Score",
                                "Inflation_Rate", "Interest_Rate", "Unemployment_Rate", "MA_5", "MA_20"] if c in view.columns]
        selected = st.selectbox("Metric to plot", choices, index=0)
        fig = px.line(view, x="Date", y=selected, markers=False, title=f"{selected} over time")
        fig.update_traces(line_width=2)
        fig.update_layout(height=430, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Feature distribution":
        selected = st.selectbox("Numeric feature", numeric_cols, index=numeric_cols.index("Close") if "Close" in numeric_cols else 0)
        fig = px.histogram(view, x=selected, nbins=35, marginal="box", title=f"Distribution of {selected}")
        fig.update_layout(height=430, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        corr_cols = [c for c in numeric_cols if view[c].nunique(dropna=True) > 1]
        corr = view[corr_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, title="Feature correlation matrix")
        fig.update_layout(height=620, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Filtered records")
    st.dataframe(view.drop(columns=["Target_Next_Close"], errors="ignore"), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download filtered data", view.to_csv(index=False).encode("utf-8"),
                       file_name="market_data_filtered.csv", mime="text/csv")

elif page == "Prediction Lab":
    st.subheader("Prediction Lab")
    st.caption("Train a regression model to estimate the next row's closing price using the available market and economic features.")
    left, right = st.columns([1, 1])
    with left:
        model_name = st.selectbox("Model", ["Random Forest", "Gradient Boosting", "Extra Trees", "Linear Regression"])
        train_pct = st.slider("Training data (%)", min_value=60, max_value=90, value=80, step=5)
    with right:
        st.markdown("**Validation setup**")
        st.write("Chronological split (older observations train; newer observations test).")
        st.write(f"Predictors: {len(features)}")
        st.write(f"Usable training examples: {len(model_df):,}")

    if len(model_df) < 10:
        st.error("At least 10 usable rows are recommended for a basic train/test evaluation.")
    else:
        split = int(len(model_df) * train_pct / 100)
        split = min(max(split, 2), len(model_df) - 1)
        train, test = model_df.iloc[:split], model_df.iloc[split:]
        model = build_model(model_name)
        model.fit(train[features], train["Target_Next_Close"])
        test_pred = model.predict(test[features])
        mae = mean_absolute_error(test["Target_Next_Close"], test_pred)
        rmse = np.sqrt(mean_squared_error(test["Target_Next_Close"], test_pred))
        r2 = r2_score(test["Target_Next_Close"], test_pred) if len(test) > 1 else np.nan
        mk1, mk2, mk3 = st.columns(3)
        mk1.metric("Mean absolute error", f"{mae:.4f}")
        mk2.metric("Root mean squared error", f"{rmse:.4f}")
        mk3.metric("Test R²", f"{r2:.3f}" if pd.notna(r2) else "N/A")
        st.markdown('<div class="small-muted">Lower MAE/RMSE generally indicates smaller prediction errors. R² can be negative when predictions perform worse than a simple mean baseline.</div>', unsafe_allow_html=True)

        actual = test["Target_Next_Close"].to_numpy()
        result_dates = df.loc[test.index, "Date"].to_numpy()
        results = pd.DataFrame({"Date": result_dates, "Actual next close": actual, "Predicted next close": test_pred})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results["Date"], y=results["Actual next close"], name="Actual", mode="lines",
                                 line=dict(color="#167d91", width=2)))
        fig.add_trace(go.Scatter(x=results["Date"], y=results["Predicted next close"], name="Predicted", mode="lines",
                                 line=dict(color="#f59e0b", width=2, dash="dash")))
        fig.update_layout(title="Out-of-sample predictions vs actual values", height=390,
                          margin=dict(l=10,r=10,t=50,b=10), hovermode="x unified", xaxis_title="", yaxis_title="Next close")
        st.plotly_chart(fig, use_container_width=True)

        # Retrain on all known target rows and predict the next observation after the latest row.
        final_model = build_model(model_name)
        final_model.fit(model_df[features], model_df["Target_Next_Close"])
        latest_x = df.iloc[[-1]][features].replace([np.inf, -np.inf], np.nan).fillna(0)
        forecast = float(final_model.predict(latest_x)[0])
        delta = forecast - last_close
        delta_pct = delta / last_close * 100 if last_close else 0
        st.markdown("### Next-step estimate")
        p1, p2, p3 = st.columns(3)
        p1.metric("Estimated next close", money(forecast), f"{delta:+.2f} vs latest close")
        p2.metric("Implied change", f"{delta_pct:+.2f}%")
        p3.metric("Model selected", model_name)
        st.markdown('<div class="note">This is a one-step estimate based on the final available row. It is not a live market quote. If the dataset is synthetic, sparse, or not representative, the forecast may not generalize.</div>', unsafe_allow_html=True)

        if hasattr(final_model, "feature_importances_"):
            importance = pd.DataFrame({"Feature": features, "Importance": final_model.feature_importances_})
            importance = importance.sort_values("Importance", ascending=False).head(12)
            st.subheader("Most influential features")
            fig_imp = px.bar(importance.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                             title="Top model feature importances")
            fig_imp.update_layout(height=400, margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig_imp, use_container_width=True)
        st.download_button("⬇️ Download test predictions", results.to_csv(index=False).encode("utf-8"),
                           file_name="market_predictions.csv", mime="text/csv")

elif page == "Data Quality":
    st.subheader("Data Quality & Preparation")
    missing = raw.isna().sum()
    quality = pd.DataFrame({
        "Column": raw.columns,
        "Data type": [str(raw[c].dtype) for c in raw.columns],
        "Missing values": [int(missing[c]) for c in raw.columns],
        "Missing (%)": [round(float(missing[c] / max(len(raw), 1) * 100), 2) for c in raw.columns],
        "Unique values": [int(raw[c].nunique(dropna=True)) for c in raw.columns],
    })
    q1, q2, q3 = st.columns(3)
    q1.metric("Original rows", f"{len(raw):,}")
    q2.metric("Rows after date cleanup", f"{len(df):,}")
    q3.metric("Columns", f"{len(raw.columns)}")
    st.dataframe(quality, hide_index=True, use_container_width=True)
    st.subheader("Numeric summary")
    st.dataframe(df.select_dtypes(include=np.number).describe().T.round(4), use_container_width=True)
    st.subheader("Preparation notes")
    st.markdown("""
    - Dates are parsed, invalid dates and missing closing prices are removed, then records are sorted chronologically.
    - Numeric predictor gaps are interpolated in time order and filled at the edges.
    - Additional features include daily return, 5-row and 20-row moving averages, and 10-row return volatility.
    - The prediction target is the **next row's closing price**. The final row is excluded from supervised training because its next close is unknown.
    - The model validation uses a chronological split rather than a random split to better respect time order.
    """)
    st.download_button("⬇️ Download cleaned dataset", df.to_csv(index=False).encode("utf-8"),
                       file_name="market_data_cleaned.csv", mime="text/csv")

st.markdown("---")
st.markdown('<div class="small-muted">MarketPulse · Educational analytics prototype · Verify data quality and independently assess model performance before making financial decisions.</div>', unsafe_allow_html=True)

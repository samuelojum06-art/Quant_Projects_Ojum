import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- PAGE SETUP ---
st.set_page_config(page_title="Portfolio Volatility Calculator", layout="wide")

st.title("Portfolio Volatility Calculator")
st.write("Estimate the volatility of a custom stock portfolio based on historical price data.")

# --- USER INPUTS ---
symbols_input = st.text_input(
    "Enter stock tickers separated by commas (e.g. AAPL, MSFT, TSLA):",
    "AAPL,MSFT,TSLA"
)
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip() != ""]

weights_input = st.text_input(
    "Enter portfolio weights (must sum to 1, e.g. 0.4, 0.3, 0.3):",
    "0.4,0.3,0.3"
)

try:
    weights = np.array([float(w.strip()) for w in weights_input.split(",") if w.strip() != ""])
except ValueError:
    st.error("Weights must be valid numbers separated by commas.")
    st.stop()

if len(symbols) == 0:
    st.error("Please enter at least one stock ticker.")
    st.stop()

if len(symbols) != len(weights):
    st.error("The number of weights must match the number of tickers.")
    st.stop()

if not np.isclose(weights.sum(), 1.0):
    st.error("Portfolio weights must sum to 1. Please adjust your inputs.")
    st.stop()

time_horizon = st.selectbox("Select time horizon:", ["6 months", "1 year", "2 years", "5 years"])
period_map = {
    "6 months": "6mo",
    "1 year": "1y",
    "2 years": "2y",
    "5 years": "5y"
}
period = period_map[time_horizon]

run_button = st.button("Calculate Portfolio Volatility")

# --- MAIN FUNCTIONALITY ---
if run_button:
    st.write("### Fetching historical data...")

    try:
        data = yf.download(symbols, period=period, group_by='column')
    except Exception as exc:
        st.error(f"Failed to download data: {exc}")
        st.stop()

    if data.empty:
        st.error(
            "No historical price data was returned for the provided tickers. "
            "Please verify the symbols or try again later."
        )
        st.stop()

    # Handle both single and multiple tickers safely
    if isinstance(data.columns, pd.MultiIndex):
        price_levels = data.columns.get_level_values(0)
        if "Adj Close" in price_levels:
            adj_close = data.xs("Adj Close", axis=1, level=0)
        elif "Close" in price_levels:
            st.warning("Using 'Close' prices instead of 'Adj Close' (adjusted) data.")
            adj_close = data.xs("Close", axis=1, level=0)
        else:
            st.error("Adjusted close or close prices were not found in the downloaded data.")
            st.stop()
    else:
        # Single ticker
        if "Adj Close" in data.columns:
            adj_close = data[["Adj Close"]].rename(columns={"Adj Close": symbols[0]})
        elif "Close" in data.columns:
            st.warning("Using 'Close' prices instead of 'Adj Close' (adjusted) data.")
            adj_close = data[["Close"]].rename(columns={"Close": symbols[0]})
        else:
            st.error("Adjusted close or close prices were not found in the downloaded data.")
            st.stop()

    adj_close = adj_close.loc[:, ~adj_close.columns.duplicated()]
    missing_tickers = [symbol for symbol in symbols if symbol not in adj_close.columns]
    if missing_tickers:
        st.error(
            "Historical data was not returned for the following tickers: "
            + ", ".join(missing_tickers)
        )
        st.stop()

    adj_close = adj_close.dropna()

    if adj_close.empty or len(adj_close) < 2:
        st.error(
            "Insufficient historical data to calculate volatility. "
            "Try selecting a longer time horizon or different tickers."
        )
        st.stop()

    # --- CALCULATIONS ---
    returns = adj_close.pct_change().dropna()
    returns = returns[symbols]
    volatilities = returns.std() * np.sqrt(252)
    volatilities = volatilities.reindex(symbols)
    corr_matrix = returns.corr().reindex(index=symbols, columns=symbols)
    cov_matrix = returns.cov().reindex(index=symbols, columns=symbols) * 252
    cov_values = cov_matrix.to_numpy()

    try:
        portfolio_volatility = np.sqrt(weights.T @ cov_values @ weights)
    except ValueError:
        st.error("Could not compute portfolio volatility with the available data.")
        st.stop()

    # --- RESULTS ---
    st.subheader("Individual Stock Volatilities")
    vol_df = pd.DataFrame({
        "Stock": symbols,
        "Weight": weights,
        "Annualized Volatility (%)": volatilities.values * 100
    })
    st.dataframe(vol_df.style.format({"Annualized Volatility (%)": "{:.2f}"}))

    st.subheader("Correlation Matrix")
    st.dataframe(corr_matrix.style.format("{:.2f}"))

    st.subheader("Portfolio Volatility")
    st.success(f"**Estimated Annualized Portfolio Volatility:** {portfolio_volatility * 100:.2f}%")

    # --- PLOT ---
    st.write("### Portfolio Risk Visualization")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(symbols, volatilities * 100, color="skyblue", label="Individual Volatility")
    ax.axhline(portfolio_volatility * 100, color="red", linestyle="--", label="Portfolio Volatility")
    ax.set_ylabel("Volatility (%)")
    ax.set_title("Individual vs. Portfolio Volatility")
    ax.legend()
    st.pyplot(fig)

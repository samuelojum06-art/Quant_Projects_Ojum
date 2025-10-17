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
symbols_input = st.text_input("Enter stock tickers separated by commas (e.g. AAPL, MSFT, TSLA):", "AAPL,MSFT,TSLA")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip() != ""]

weights_input = st.text_input("Enter portfolio weights (must sum to 1, e.g. 0.4, 0.3, 0.3):", "0.4,0.3,0.3")
weights = np.array([float(w.strip()) for w in weights_input.split(",") if w.strip() != ""])

if len(symbols) != len(weights):
    st.error("⚠️ The number of weights must match the number of tickers.")
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
    data = yf.download(symbols, period=period, group_by='ticker')

    # Handle single or multiple tickers cleanly
    if len(symbols) > 1:
        data = pd.concat([data[ticker]["Adj Close"] for ticker in symbols], axis=1)
        data.columns = symbols
    else:
        data = data["Adj Close"].to_frame(symbols[0])

    data = data.dropna()

    # --- CALCULATIONS ---
    returns = data.pct_change().dropna()
    volatilities = returns.std() * np.sqrt(252)
    corr_matrix = returns.corr()
    cov_matrix = returns.cov() * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # --- RESULTS ---
    st.subheader("📊 Individual Stock Volatilities")
    vol_df = pd.DataFrame({
        "Stock": symbols,

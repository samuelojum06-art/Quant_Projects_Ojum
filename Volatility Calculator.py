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
    st.error("The number of weights must match the number of tickers.")
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
    data = yf.download(symbols, period=period)["Adj Close"].dropna()

    # Calculate daily returns
    returns = data.pct_change().dropna()

    # Annualized volatilities
    volatilities = returns.std() * np.sqrt(252)

    # Correlation matrix
    corr_matrix = returns.corr()

    # Covariance matrix (annualized)
    cov_matrix = returns.cov() * 252

    # Portfolio volatility formula
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

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
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(symbols, volatilities * 100, color="skyblue", label="Individual Volatility")
    ax.axhline(portfolio_volatility * 100, color="red", linestyle="--", label="Portfolio Volatility")
    ax.set_ylabel("Volatility (%)")
    ax.set_title("Individual vs. Portfolio Volatility")
    ax.legend()
    st.pyplot(fig)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import date, timedelta

# --- PAGE CONFIG ---
st.set_page_config(page_title="Monte Carlo Stock Simulator", layout="wide")

st.title("📈 Monte Carlo Stock Price Simulation (Polygon.io Version)")
st.write("Simulate future stock prices using Polygon.io market data and the Monte Carlo method.")

# --- USER INPUTS ---
POLYGON_API_KEY = st.text_input("Enter your Polygon.io API Key:", type="password")

symbols_input = st.text_input("Enter stock tickers separated by commas (e.g. AAPL, MSFT, TSLA):", "AAPL,MSFT,TSLA")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip() != ""]

T = st.number_input("Time horizon (in years):", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
n_sims = st.number_input("Number of simulations:", min_value=100, max_value=5000, value=500, step=100)
steps = 252  # trading days per year

run_button = st.button("Run Simulation 🚀")

# --- FUNCTIONS ---
def get_stock_params_polygon(symbol, api_key):
    """Fetch historical prices from Polygon.io and calculate drift and volatility."""
    end_date = date.today()
    start_date = end_date - timedelta(days=500)
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&apiKey={api_key}"

    r = requests.get(url)
    if r.status_code != 200:
        st.warning(f"⚠️ Error fetching {symbol}: {r.text}")
        return None

    data = r.json()
    if "results" not in data:
        st.warning(f"⚠️ No data available for {symbol}. Check the ticker or API limits.")
        return None

    prices = pd.DataFrame(data["results"])
    prices["date"] = pd.to_datetime(prices["t"], unit="ms")
    prices.rename(columns={"c": "close"}, inplace=True)
    returns = prices["close"].pct_change().dropna()

    S0 = prices["close"].iloc[-1]
    mu = returns.mean() * 252
    sigma = returns.std() * np.sqrt(252)
    return S0, mu, sigma

def monte_carlo_stock(S0, mu, sigma, T, steps, n_sims):
    """Run Monte Carlo simulation."""
    dt = T / steps
    prices = np.zeros((steps + 1, n_sims))
    prices[0] = S0
    for t in range(1, steps + 1):
        z = np.random.standard_normal(n_sims)
        prices[t] = prices[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return prices

# --- RUN SIMULATIONS ---
if run_button:
    if not POLYGON_API_KEY:
        st.error("❌ Please enter your Polygon.io API key first.")
    else:
        results = []
        st.write("### Running Simulations...")
        progress = st.progress(0)

        for i, symbol in enumerate(symbols):
            progress.progress((i + 1) / len(symbols))
            params = get_stock_params_polygon(symbol, POLYGON_API_KEY)
            if params is None:
                continue

            S0, mu, sigma = params
            sims = monte_carlo_stock(S0, mu, sigma, T, steps, n_sims)
            final_prices = sims[-1]

            expected = np.mean(final_prices)
            median = np.median(final_prices)
            ci_lower = np.percentile(final_prices, 2.5)
            ci_upper = np.percentile(final_prices, 97.5)

            results.append({
                "Symbol": symbol,
                "Current Price": S0,
                "Expected Price (1yr)": expected,
                "Median Price (1yr)": median,
                "95% CI Lower": ci_lower,
                "95% CI Upper": ci_upper,
                "Drift (mu)": mu,
                "Volatility (sigma)": sigma
            })

            st.subheader(f"Simulation for {symbol}")
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(sims, linewidth=0.8)
            ax.set_title(f"Monte Carlo Simulation: {symbol}")
            ax.set_xlabel("Days")
            ax.set_ylabel("Stock Price")
            st.pyplot(fig)

        # --- COMPARISON HISTOGRAM ---
        if results:
            st.subheader(f"{T}-Year Final Price Distribution Comparison")
            fig, ax = plt.subplots(figsize=(10,5))
            for result in results:
                symbol = result["Symbol"]
                S0, mu, sigma = result["Current Price"], result["Drift (mu)"], result["Volatility (sigma)"]
                sims = monte_carlo_stock(S0, mu, sigma, T, steps, n_sims)
                ax.hist(sims[-1], bins=30, alpha=0.5, label=symbol)
            ax.set_xlabel("Final Price")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)

            # --- RESULTS TABLE + DOWNLOAD ---
            df_results = pd.DataFrame(results)
            st.subheader("📊 Summary Statistics")
            st.dataframe(df_results.style.format("{:.2f}"))

            csv = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="monte_carlo_results.csv",
                mime="text/csv"
            )
        else:
            st.error("No valid simulations were completed. Please check your tickers or API key.")

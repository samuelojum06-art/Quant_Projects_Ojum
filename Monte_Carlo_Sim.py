import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from requests.exceptions import HTTPError, RequestException

# --- PAGE CONFIG ---
st.set_page_config(page_title="Monte Carlo Stock Simulator", layout="wide")

# --- TITLE ---
st.title("Monte Carlo Stock Price Simulation")
st.markdown("**By Samuel Ojum**")
st.write("Simulate future stock prices using real market data and the Monte Carlo method.")

# --- USER INPUTS ---
API_KEY = "axUgQt57qF6D60tZBtu7TPo1dIU5yygb"

symbols_input = st.text_input(
    "Enter stock tickers separated by commas (e.g. AAPL, MSFT, TSLA):",
    "AAPL,MSFT,TSLA"
)
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip() != ""]

T = st.number_input("Time horizon (in years):", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
n_sims = st.number_input("Number of simulations:", min_value=100, max_value=5000, value=500, step=100)
steps = 252  # Trading days per year

run_button = st.button("Run Simulation")

# --- FUNCTIONS ---
@st.cache_data(show_spinner=False)
def get_stock_params(symbol):
    """Fetch historical prices and calculate drift (mu) and volatility (sigma)."""
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?timeseries=500&apikey={API_KEY}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except HTTPError as exc:
        st.warning(f"⚠️ Unable to load data for {symbol}: {exc}")
        return None
    except (RequestException, ValueError) as exc:
        st.warning(f"⚠️ Network error while requesting {symbol}: {exc}")
        return None

    if "historical" not in data or len(data["historical"]) == 0:
        st.warning(f"⚠️ No data found for {symbol}. Skipping...")
        return None

    prices = pd.DataFrame(data["historical"])[["date", "close"]].sort_values("date")
    returns = prices["close"].pct_change().dropna()

    if returns.empty:
        st.warning(f"⚠️ Not enough historical data to calculate returns for {symbol}. Skipping...")
        return None

    S0 = prices["close"].iloc[-1]
    mu = returns.mean() * 252
    sigma = returns.std() * np.sqrt(252)
    return S0, mu, sigma


def monte_carlo_stock(S0, mu, sigma, T, steps, n_sims):
    """Run the Monte Carlo simulation for one stock."""
    dt = T / steps
    prices = np.zeros((steps + 1, n_sims))
    prices[0] = S0

    for t in range(1, steps + 1):
        z = np.random.standard_normal(n_sims)
        prices[t] = prices[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return prices


# --- RUN SIMULATIONS ---
if run_button:
    results = []
    simulation_outputs = {}
    st.write("### Running Simulations...")
    progress = st.progress(0)

    for i, symbol in enumerate(symbols):
        progress.progress((i + 1) / len(symbols))
        params = get_stock_params(symbol)
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

        simulation_outputs[symbol] = {"final_prices": final_prices}

        # --- Plot simulated paths ---
        st.subheader(f"Simulation for {symbol}")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sims, linewidth=0.8)
        ax.set_title(f"Monte Carlo Simulation: {symbol}")
        ax.set_xlabel("Days")
        ax.set_ylabel("Stock Price")
        st.pyplot(fig)
        plt.close(fig)

    # --- COMPARISON HISTOGRAM ---
    if results:
        st.subheader(f"{T}-Year Final Price Distribution Comparison")
        fig, ax = plt.subplots(figsize=(10, 5))
        for result in results:
            symbol = result["Symbol"]
            final_prices = simulation_outputs[symbol]["final_prices"]
            ax.hist(final_prices, bins=30, alpha=0.5, label=symbol)
        ax.set_xlabel("Final Price")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        # --- SHOW RESULTS TABLE ---
        df_results = pd.DataFrame(results)
        st.subheader("📊 Summary Statistics")
        st.dataframe(df_results.style.format("{:.2f}"))

        # --- CSV DOWNLOAD BUTTON ---
        csv = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="monte_carlo_results.csv",
            mime="text/csv"
        )
    else:
        st.error("No valid simulations could be completed. Please check your tickers.")

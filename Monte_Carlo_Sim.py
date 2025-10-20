from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# --- PAGE CONFIG ---
st.set_page_config(page_title="Monte Carlo Stock Simulator", layout="wide")

# --- TITLE ---
st.title("Monte Carlo Stock Price Simulation")
st.markdown("**By Samuel Ojum**")
st.write("Simulate future stock prices using real market data and the Monte Carlo method.")

# --- USER INPUTS ---
symbols_input = st.text_input(
    "Enter stock tickers separated by commas (e.g. AAPL, MSFT, TSLA):",
    "AAPL,MSFT,TSLA"
)
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip() != ""]

T = st.number_input("Time horizon (in years):", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
n_sims = st.number_input("Number of simulations:", min_value=100, max_value=5000, value=500, step=100)
steps = 252  # Trading days per year

run_button = st.button("Run Simulation")

SAMPLE_DATA_PATH = Path(__file__).resolve().parent / "data" / "sample_prices.csv"


@st.cache_data(show_spinner=False)
def load_sample_prices() -> pd.DataFrame | None:
    """Load offline price data if available."""
    if not SAMPLE_DATA_PATH.exists():
        return None
    try:
        return pd.read_csv(SAMPLE_DATA_PATH, parse_dates=["Date"])
    except Exception as exc:  # pragma: no cover - defensive
        st.warning(f"⚠️ Failed to read fallback price data: {exc}")
        return None


@st.cache_data(show_spinner=False)
def build_synthetic_history(symbol: str, years: float = 5.0, steps_per_year: int = 252) -> pd.Series:
    """Generate a deterministic synthetic price history when no data is available."""
    total_steps = max(int(years * steps_per_year), 2)
    # Use a stable seed per symbol so reruns show the same paths.
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    dt = 1 / steps_per_year
    mu = 0.07  # Assumed annual drift for synthetic prices
    sigma = 0.25  # Assumed annual volatility
    prices = np.zeros(total_steps)
    prices[0] = 100.0
    for t in range(1, total_steps):
        shock = rng.standard_normal()
        prices[t] = prices[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shock)
    start_date = pd.Timestamp.today() - pd.Timedelta(days=total_steps)
    index = pd.date_range(start=start_date, periods=total_steps, freq="B")
    return pd.Series(prices, index=index, name=symbol)


# --- FUNCTIONS ---
@st.cache_data(show_spinner=False)
def get_stock_params(symbol):
    """Fetch historical prices and calculate drift (mu) and volatility (sigma)."""
    data = pd.DataFrame()
    try:
        data = yf.download(symbol, period="5y", progress=False, auto_adjust=True)
    except Exception as exc:
        st.warning(f"⚠️ Yahoo Finance request failed for {symbol}: {exc}")

    prices: pd.Series | None = None
    if not data.empty:
        if isinstance(data.columns, pd.MultiIndex):
            close_key = ("Close", symbol) if ("Close", symbol) in data.columns else None
            if close_key:
                prices = data[close_key].dropna()
        elif "Close" in data.columns:
            prices = data["Close"].dropna()

    if prices is None or prices.empty:
        fallback = load_sample_prices()
        if fallback is not None:
            symbol_data = fallback[fallback["Symbol"].str.upper() == symbol.upper()].sort_values("Date")
            if not symbol_data.empty:
                st.info(
                    f"ℹ️ Using bundled sample data for {symbol} because live market data is unavailable."
                )
                prices = symbol_data["Close"].dropna()

    if prices is None or prices.empty:
        st.info(
            "ℹ️ Generating synthetic price history because no market data was available."
        )
        prices = build_synthetic_history(symbol)

    returns = prices.pct_change().dropna()

    if returns.empty:
        st.warning(f"⚠️ Not enough historical data to calculate returns for {symbol}. Skipping...")
        return None

    S0 = prices.iloc[-1]
    mu = returns.mean() * 252
    sigma = returns.std() * np.sqrt(252)
    return S0, mu, sigma


def monte_carlo_stock(S0, mu, sigma, T, steps, n_sims):
    """Run the Monte Carlo simulation."""
    dt = T / steps
    prices = np.zeros((steps + 1, n_sims))
    prices[0] = S0

    for t in range(1, steps + 1):
        z = np.random.standard_normal(n_sims)
        prices[t] = prices[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return prices


# --- MAIN APP ---
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

        numeric_columns = df_results.select_dtypes(include="number").columns
        if len(numeric_columns) > 0:
            st.dataframe(
                df_results.style.format({col: "{:.2f}" for col in numeric_columns})
            )
        else:
            st.dataframe(df_results)

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

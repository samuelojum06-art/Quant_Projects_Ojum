"""Streamlit Monte Carlo stock price simulator with offline fallbacks.

Run with:

    streamlit run Monte_Carlo_Sim.py

The app downloads historical prices when possible and otherwise falls back to
bundled sample data or deterministic synthetic series so simulations always
complete. This file intentionally embeds the sample data so that copy/paste
usage works without needing any auxiliary assets.
"""

import hashlib
from typing import Optional

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


OFFLINE_SAMPLES = {
    "AAPL": {
        "dates": [
            "2020-12-21",
            "2020-12-22",
            "2020-12-23",
            "2020-12-24",
            "2020-12-25",
            "2020-12-28",
            "2020-12-29",
            "2020-12-30",
            "2020-12-31",
            "2021-01-01",
            "2021-01-04",
            "2021-01-05",
            "2021-01-06",
            "2021-01-07",
            "2021-01-08",
            "2021-01-11",
            "2021-01-12",
            "2021-01-13",
            "2021-01-14",
            "2021-01-15",
            "2021-01-18",
            "2021-01-19",
            "2021-01-20",
            "2021-01-21",
            "2021-01-22",
            "2021-01-25",
            "2021-01-26",
            "2021-01-27",
            "2021-01-28",
            "2021-01-29",
            "2021-02-01",
            "2021-02-02",
            "2021-02-03",
            "2021-02-04",
            "2021-02-05",
            "2021-02-08",
            "2021-02-09",
            "2021-02-10",
            "2021-02-11",
            "2021-02-12",
            "2021-02-15",
            "2021-02-16",
            "2021-02-17",
            "2021-02-18",
            "2021-02-19",
            "2021-02-22",
            "2021-02-23",
            "2021-02-24",
            "2021-02-25",
            "2021-02-26",
            "2021-03-01",
            "2021-03-02",
            "2021-03-03",
            "2021-03-04",
            "2021-03-05",
            "2021-03-08",
            "2021-03-09",
            "2021-03-10",
            "2021-03-11",
            "2021-03-12",
            "2021-03-15",
            "2021-03-16",
            "2021-03-17",
            "2021-03-18",
        ],
        "prices": [
            190.0,
            191.130704,
            187.761519,
            190.369722,
            193.662474,
            187.215905,
            183.067139,
            183.582946,
            182.663803,
            182.711738,
            180.083775,
            183.001072,
            185.632646,
            185.952842,
            189.793094,
            191.471566,
            188.696812,
            190.034273,
            186.951658,
            189.977037,
            189.915897,
            189.403449,
            187.246822,
            191.435318,
            191.020943,
            189.689174,
            188.620015,
            190.505729,
            191.844822,
            193.354535,
            194.938315,
            202.556127,
            201.221695,
            199.523271,
            196.789771,
            199.050705,
            203.167654,
            202.873049,
            200.000561,
            197.223266,
            199.611011,
            202.358126,
            204.420286,
            202.147644,
            203.090565,
            203.622723,
            204.523917,
            207.807924,
            208.745706,
            211.378515,
            211.748902,
            212.950441,
            215.455216,
            210.105543,
            209.040968,
            207.429679,
            205.219958,
            204.340567,
            209.917598,
            206.85167,
            210.532392,
            204.489212,
            203.398505,
            204.097242,
        ],
    },
    "MSFT": {
        "dates": [
            "2020-12-21",
            "2020-12-22",
            "2020-12-23",
            "2020-12-24",
            "2020-12-25",
            "2020-12-28",
            "2020-12-29",
            "2020-12-30",
            "2020-12-31",
            "2021-01-01",
            "2021-01-04",
            "2021-01-05",
            "2021-01-06",
            "2021-01-07",
            "2021-01-08",
            "2021-01-11",
            "2021-01-12",
            "2021-01-13",
            "2021-01-14",
            "2021-01-15",
            "2021-01-18",
            "2021-01-19",
            "2021-01-20",
            "2021-01-21",
            "2021-01-22",
            "2021-01-25",
            "2021-01-26",
            "2021-01-27",
            "2021-01-28",
            "2021-01-29",
            "2021-02-01",
            "2021-02-02",
            "2021-02-03",
            "2021-02-04",
            "2021-02-05",
            "2021-02-08",
            "2021-02-09",
            "2021-02-10",
            "2021-02-11",
            "2021-02-12",
            "2021-02-15",
            "2021-02-16",
            "2021-02-17",
            "2021-02-18",
            "2021-02-19",
            "2021-02-22",
            "2021-02-23",
            "2021-02-24",
            "2021-02-25",
            "2021-02-26",
            "2021-03-01",
            "2021-03-02",
            "2021-03-03",
            "2021-03-04",
            "2021-03-05",
            "2021-03-08",
            "2021-03-09",
            "2021-03-10",
            "2021-03-11",
            "2021-03-12",
            "2021-03-15",
            "2021-03-16",
            "2021-03-17",
            "2021-03-18",
        ],
        "prices": [
            320.0,
            318.027255,
            322.623431,
            323.220588,
            326.968121,
            325.110414,
            325.049424,
            324.691593,
            321.111473,
            324.233238,
            327.331669,
            330.200167,
            337.477414,
            344.496463,
            342.882144,
            344.831541,
            342.034949,
            342.316519,
            339.652577,
            347.538182,
            349.468743,
            348.588356,
            354.579626,
            344.243839,
            334.935042,
            338.818681,
            339.27206,
            340.621669,
            334.584502,
            334.643802,
            344.237534,
            348.365935,
            349.360585,
            347.414668,
            345.692625,
            335.808585,
            337.178033,
            341.322432,
            334.116841,
            331.233506,
            326.125357,
            321.952022,
            322.351617,
            313.3294,
            316.165948,
            319.64592,
            318.694555,
            308.077987,
            304.00468,
            309.398075,
            297.861479,
            296.578378,
            291.913481,
            295.334013,
            292.729791,
            294.520159,
            296.985466,
            304.792162,
            304.087899,
            305.829419,
            299.749092,
            304.887408,
            301.841516,
            299.654451,
        ],
    },
    "TSLA": {
        "dates": [
            "2020-12-21",
            "2020-12-22",
            "2020-12-23",
            "2020-12-24",
            "2020-12-25",
            "2020-12-28",
            "2020-12-29",
            "2020-12-30",
            "2020-12-31",
            "2021-01-01",
            "2021-01-04",
            "2021-01-05",
            "2021-01-06",
            "2021-01-07",
            "2021-01-08",
            "2021-01-11",
            "2021-01-12",
            "2021-01-13",
            "2021-01-14",
            "2021-01-15",
            "2021-01-18",
            "2021-01-19",
            "2021-01-20",
            "2021-01-21",
            "2021-01-22",
            "2021-01-25",
            "2021-01-26",
            "2021-01-27",
            "2021-01-28",
            "2021-01-29",
            "2021-02-01",
            "2021-02-02",
            "2021-02-03",
            "2021-02-04",
            "2021-02-05",
            "2021-02-08",
            "2021-02-09",
            "2021-02-10",
            "2021-02-11",
            "2021-02-12",
            "2021-02-15",
            "2021-02-16",
            "2021-02-17",
            "2021-02-18",
            "2021-02-19",
            "2021-02-22",
            "2021-02-23",
            "2021-02-24",
            "2021-02-25",
            "2021-02-26",
            "2021-03-01",
            "2021-03-02",
            "2021-03-03",
            "2021-03-04",
            "2021-03-05",
            "2021-03-08",
            "2021-03-09",
            "2021-03-10",
            "2021-03-11",
            "2021-03-12",
            "2021-03-15",
            "2021-03-16",
            "2021-03-17",
            "2021-03-18",
        ],
        "prices": [
            210.0,
            213.8109,
            202.690802,
            201.421181,
            213.157463,
            211.831938,
            200.252586,
            212.661344,
            217.389428,
            217.01819,
            214.600605,
            222.152829,
            218.501029,
            226.660529,
            230.678971,
            232.578964,
            233.763312,
            232.593937,
            240.501126,
            234.150733,
            222.599737,
            227.062278,
            231.73686,
            235.385527,
            238.546194,
            236.267098,
            242.32077,
            242.093358,
            238.525644,
            236.311111,
            251.964802,
            238.6836,
            242.590953,
            230.208571,
            229.53895,
            226.537565,
            212.243583,
            214.96692,
            212.514477,
            199.016426,
            204.450788,
            214.009794,
            224.404969,
            227.069109,
            227.511581,
            237.348654,
            250.38915,
            259.619318,
            252.158229,
            253.888344,
            257.555201,
            243.406412,
            247.1541,
            265.628993,
            271.171836,
            284.193937,
            285.9778,
            285.460612,
            289.052695,
            281.859555,
            273.567634,
            284.84061,
            297.570304,
            288.567674,
        ],
    },
}


@st.cache_data(show_spinner=False)
def load_offline_history(symbol: str) -> Optional[pd.Series]:
    """Return a bundled offline price history for well-known tickers."""
    sample = OFFLINE_SAMPLES.get(symbol.upper())
    if sample is None:
        return None

    dates = pd.to_datetime(sample["dates"])
    prices = pd.Series(sample["prices"], index=dates, name=symbol.upper())
    return prices.sort_index()


def _rng_for_symbol(symbol: str) -> np.random.Generator:
    """Return a deterministic RNG seeded from the ticker symbol."""
    digest = hashlib.sha256(symbol.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed)


@st.cache_data(show_spinner=False)
def build_synthetic_history(symbol: str, years: float = 5.0, steps_per_year: int = 252) -> pd.Series:
    """Generate a deterministic synthetic price history when no data is available."""
    total_steps = max(int(years * steps_per_year), 2)
    rng = _rng_for_symbol(symbol.upper())
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

    prices: Optional[pd.Series] = None
    if not data.empty:
        if isinstance(data.columns, pd.MultiIndex):
            close_key = ("Close", symbol) if ("Close", symbol) in data.columns else None
            if close_key:
                prices = data[close_key].dropna()
        elif "Close" in data.columns:
            prices = data["Close"].dropna()

    if prices is None or prices.empty:
        offline_prices = load_offline_history(symbol)
        if offline_prices is not None:
            st.info(
                f"ℹ️ Using bundled sample data for {symbol} because live market data is unavailable."
            )
            prices = offline_prices.dropna()

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

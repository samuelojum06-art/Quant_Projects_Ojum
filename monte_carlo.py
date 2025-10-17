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

# --- API KEY (built-in) ---
POLYGON_API_KEY = "gC6x9Qp7IjQPc3eehVVDxYwBAWHG92bc"

# --- USER INPUTS ---
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
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc

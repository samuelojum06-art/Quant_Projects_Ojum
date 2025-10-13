import math
from scipy.stats import norm


def black_scholes(S, K, T, r, sigma):
    """
    S: Stock price
    K: Strike price
    T: Time to expiry (in years)
    r: Risk-free interest rate (annual)
    sigma: Volatility (annual)

    Returns: call_price, put_price
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return call_price, put_price

import streamlit as st

st.title("Black-Scholes Options Pricer & P&L Heatmap")
st.subheader("Created by Samuel Ojum")

# User inputs
S = st.number_input("Stock Price (S)", min_value=0.01, value=100.0)
K = st.number_input("Strike Price (K)", min_value=0.01, value=100.0)
T = st.number_input("Time to Expiry (Years)", min_value=0.01, value=1.0)
r = st.number_input("Risk-free Interest Rate (r)", min_value=0.0, value=0.05)
sigma = st.number_input("Volatility (σ)", min_value=0.01, value=0.2)

call_purchase = st.number_input("Call Purchase Price", min_value=0.0, value=10.0)
put_purchase = st.number_input("Put Purchase Price", min_value=0.0, value=10.0)

# Calculate
call_price, put_price = black_scholes(S, K, T, r, sigma)

st.write(f"Calculated Call Price: ${call_price:.2f}")
st.write(f"Calculated Put Price: ${put_price:.2f}")

# Calculate P&L
call_pl = call_price - call_purchase
put_pl = put_price - put_purchase

st.markdown(f"Call P&L: <span style='color:{'green' if call_pl>0 else 'red'}'>{call_pl:.2f}</span>", unsafe_allow_html=True)
st.markdown(f"Put P&L: <span style='color:{'green' if put_pl>0 else 'red'}'>{put_pl:.2f}</span>", unsafe_allow_html=True)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create ranges for stock price and volatility
S_range = np.linspace(S*0.8, S*1.2, 20)
sigma_range = np.linspace(sigma*0.5, sigma*1.5, 20)

call_pl_matrix = np.zeros((len(sigma_range), len(S_range)))
put_pl_matrix = np.zeros((len(sigma_range), len(S_range)))

for i, sig in enumerate(sigma_range):
    for j, s_val in enumerate(S_range):
        c, p = black_scholes(s_val, K, T, r, sig)
        call_pl_matrix[i, j] = c - call_purchase
        put_pl_matrix[i, j] = p - put_purchase

call_df = pd.DataFrame(call_pl_matrix, index=np.round(sigma_range, 2), columns=np.round(S_range, 2))
put_df = pd.DataFrame(put_pl_matrix, index=np.round(sigma_range, 2), columns=np.round(S_range, 2))




fig, ax = plt.subplots()
sns.heatmap(call_df, cmap="RdYlGn", ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots()
sns.heatmap(put_df, cmap="RdYlGn", ax=ax)
st.pyplot(fig)




# Black-Scholes Options Pricer & P&L Heatmap
# Overview

This project is an interactive Streamlit web application that implements the Black-Scholes model to calculate the theoretical prices of European call and put options. It visualizes potential profit and loss (P&L) outcomes across different volatility and stock price scenarios using heatmaps. The project combines quantitative finance, Python programming, and data visualization to demonstrate how option values and portfolio sensitivities change with market conditions. Developed by Samuel Ojum, this application showcases both technical proficiency in financial computation and analytical insight into derivatives pricing and risk modeling.

# 1. Theoretical Background

The Black-Scholes model provides a closed-form solution for pricing European options under certain assumptions such as log-normally distributed asset returns and constant volatility. The call and put prices are calculated using the following equations:

C = S * N(d1) - K * exp(-r * T) * N(d2)
P = K * exp(-r * T) * N(-d2) - S * N(-d1)

where

d1 = [ln(S / K) + (r + 0.5 * sigma^2) * T] / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)

Here, 
𝑆
S represents the current stock price, 
𝐾
K the strike price, 
𝑇
T the time to maturity in years, 
𝑟
r the risk-free interest rate, and 
𝜎
σ the volatility of the underlying asset. 
𝑁
(
𝑥
)
N(x) is the cumulative distribution function of the standard normal distribution. The model allows for the computation of both call and put prices, which are then used to evaluate profit and loss (P&L) given a purchase cost.

# 2. Application Features

The Black-Scholes Options Pricer & P&L Heatmap provides a highly interactive experience where users can input stock price, strike price, volatility, time to expiry, and risk-free rate directly in the interface. It instantly computes theoretical call and put prices using the Black-Scholes formula and evaluates profit and loss based on the user’s purchase price for each option. Two dynamic heatmaps visualize how P&L changes as stock price and volatility vary, offering a clear picture of sensitivity and risk exposure. The app’s intuitive Streamlit interface makes it accessible for both finance students learning about derivatives and professionals seeking a quick analytical tool.

# 3. Technical Implementation

This project was developed in Python and leverages several key libraries to achieve its functionality. Streamlit provides the front-end interface for real-time interactivity, while NumPy and math are used for numerical computation and mathematical modeling. The scipy.stats module enables cumulative normal distribution calculations essential to the Black-Scholes framework. Pandas is employed to organize the simulation data into matrices, and Matplotlib along with Seaborn handle data visualization, producing the colorful heatmaps that display option profit and loss outcomes. Together, these tools create a seamless integration of quantitative modeling and visual analytics.

# 4. Key Learning Outcomes

This project demonstrates a practical understanding of derivatives pricing and risk visualization in financial modeling. By integrating theoretical concepts with computational methods and visualization, it bridges the gap between quantitative finance and interactive analytics. It serves as a portfolio-ready example of technical and analytical skills in Python, particularly relevant for careers in quantitative research, equity derivatives, and risk analytics.

# 5. Author

Samuel Ojum
Finance & Mathematics | University of Arizona
GitHub: Samuelojum06-art

LinkedIn: linkedin.com/in/samuelojum06

# 6. Running the Project

To run the Black-Scholes Options Pricer & P&L Heatmap locally, open a terminal or command prompt and execute the following commands:
bash``
# Clone the repository
git clone https://github.com/Samuelojum06-art/BlackScholes_Heatmap.git
cd BlackScholes_Heatmap

# Install dependencies
pip install -r requirements.txt
# (If the requirements file is unavailable, install manually:)
# pip install streamlit numpy scipy pandas matplotlib seaborn

# Launch the app
streamlit run "BlackScholes_Heatmap.py"
Once executed, Streamlit will display a local link (for example, http://localhost:8501). Open this link in your browser to access the interactive Black-Scholes pricing and heatmap tool.

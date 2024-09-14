import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# Define constants
risk_free_rate = 0.04
expected_return = 0.08

# Function to get data from Yahoo Finance
def get_data(ticker):
    try:
        # Fetch historical market data
        stock_data = yf.download(ticker, start='2023-01-01', end='2024-01-01')
        stock_data.to_csv(f'{ticker}_data.csv')
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Function to calculate daily returns
def calculate_daily_return(ticker):
    data = get_data(ticker)
    if data is not None:
        data['Daily Return'] = data['Close'].pct_change()
        return data['Daily Return']
    else:
        return None

# List of stocks to process
stocks = ['IAG.TO', 'HTB.TO', 'HXDM.TO', 'HXS.TO', 'SHOP.TO']

# Fetch and calculate daily returns for each stock
daily_returns = {ticker: calculate_daily_return(ticker) for ticker in stocks}
daily_returns = {k: v.dropna() for k, v in daily_returns.items() if v is not None}

# Function to plot daily returns
def plot_daily_returns(daily_returns):
    plt.figure(figsize=(12, 8))
    for ticker, returns in daily_returns.items():
        plt.plot(returns.index, returns, label=ticker)
    plt.title('Daily Returns of Stocks')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_daily_returns(daily_returns)

# Function to calculate correlation matrix
def calculate_correlation_matrix(daily_returns):
    returns_df = pd.DataFrame(daily_returns)
    correlation_matrix = returns_df.corr()
    return correlation_matrix

# Function to plot correlation heatmap
def plot_correlation_heatmap(correlation_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title('Correlation Matrix Heatmap')
    plt.show()

# Calculate and plot correlation matrix
correlation_matrix = calculate_correlation_matrix(daily_returns)
plot_correlation_heatmap(correlation_matrix)

# Define the expected returns
expected_returns = {
    "IAG.TO": 0.10,
    "HTB.TO": 0.05,
    "HXDM.TO": 0.08,
    "HXS.TO": 0.085,
    "SHOP.TO": 0.11
}

# Define the portfolio weights
portfolios = {
    "Portfolio 1": [1, 0, 0, 0, 0],
    "Portfolio 2": [0.2, 0.2, 0.2, 0.2, 0.2],
    "Portfolio 3": [0.5, 0.25, 0.25, 0, 0],
    "Portfolio 4": [0.1, 0.2, 0.2, 0.3, 0.2],
    "Portfolio 5": [0.05, 0.15, 0.2, 0.4, 0.1]
}

# Convert daily returns to DataFrame
returns_df = pd.DataFrame(daily_returns)

# Calculate covariance matrix
cov_matrix = returns_df.cov()

# Function to calculate portfolio expected return
def calculate_portfolio_return(weights, expected_returns):
    return np.dot(weights, list(expected_returns.values()))

# Function to calculate portfolio standard deviation
def calculate_portfolio_std(weights, cov_matrix):
    return np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

# Calculate expected returns and standard deviations for each portfolio
portfolio_returns = {}
portfolio_stds = {}

for name, weights in portfolios.items():
    portfolio_returns[name] = calculate_portfolio_return(weights, expected_returns)
    portfolio_stds[name] = calculate_portfolio_std(weights, cov_matrix)

# Plot the results
plt.figure(figsize=(10, 6))
for name in portfolios.keys():
    plt.scatter(portfolio_stds[name], portfolio_returns[name], label=name)
    plt.text(portfolio_stds[name], portfolio_returns[name], name, fontsize=9)

plt.xlabel('Expected Risk (Standard Deviation)')
plt.ylabel('Expected Return')
plt.title('Portfolio Expected Return vs. Expected Risk')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

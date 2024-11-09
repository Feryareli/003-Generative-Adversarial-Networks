# 003-Generative-Adversarial-Networks

## Introduction

This project leverages Generative Adversarial Networks (GANs)
to create synthetic market scenarios for backtesting a trading
strategy. By generating multiple scenarios, the goal is to simulate 
diverse market conditions that allow for a thorough evaluation of the 
trading strategy’s performance. This approach helps in understanding 
how the strategy might perform under different market dynamics compared 
to a passive buy-and-hold benchmark.

## Objective

The main objective is to develop a simple algorithmic trading strategy
using GAN-generated scenarios for backtesting. The performance of the
strategy is assessed based on various risk and return metrics, and it 
is compared to a passive investment strategy. Key performance indicators 
include Sharpe Ratio, Calmar Ratio, Max Drawdown, Profit & Loss, and 
Win-Loss ratio. The results, along with visualizations, are presented 
in a Jupyter notebook to summarize the effectiveness of the active 
trading strategy.

---


---

## Data Overview
 
The project uses 10 years of historical price data for a single stock
as the basis for training the GAN model. This historical data allows 
the model to learn patterns and generate synthetic data that mimics 
realistic market scenarios. The generated scenarios provide different 
conditions for backtesting, allowing a robust analysis of the trading 
strategy’s behavior in simulated environments.

### Datasets:
 
- **AAPL 5-minute Test Set (aapl_5m_test.csv)**
- **AAPL 5-minute Train Set (aapl_5m_train.csv)**
 
- **Bitcoin Project Test Set (btc_project_test.csv)**
- **Bitcoin Project Train Set (btc_project_train.csv)**
 
Each dataset contains key market data, including timestamp, price information (Open, High, Low, Close), and Volume. The training sets are significantly larger, allowing for better optimization of parameters, while the test sets are used to evaluate the performance of the strategies.

---

## Project Structure
 
- **gans_strategy/**: Contains the module-specific code for training the GAN and running the backtesting module.
  - **backtest.py**: Functions for backtesting the strategy, including buy/sell signal generation based on stop-loss and take-profit levels, and functions to calculate Max Drawdown, Sharpe Ratio, and Win-Loss Ratio.
  - **strategy.py**: Contains the simple algorithmic trading strategy using RSI-based signals for entry and exit.
  - **train_gan.py**: Defines the GAN model structure and training function, which generates synthetic market data based on historical data.


- **utils/**: Helper functions for data processing and visualization.
  - **utils.py**: Functions to download historical data, calculate returns, normalize data, and plot candlestick charts with trading signals.
  - **main.py**: The main script that integrates data preprocessing, GAN training, scenario generation, and backtesting for both active and passive strategies.


- **README.md**: This file, describing the project and setup instructions.
- **requirements.txt**: Python dependencies required to run the project.
- **report.ipynb**: Jupyter Notebook with visualizations and summary of the best strategies.
- **venv/**: Virtual environment for the project.

---

## Backtesting Summary

The backtesting process compares the active trading strategy with a passive benchmark strategy, assessing the following metrics across different synthetic market scenarios:

- **Initial Capital**: $1,000,000
- **Max Drawdown**: Calculated to measure the worst observed loss from a peak portfolio value.
- **Win-Loss Ratio**: Ratio of profitable trades to losing trades.
- **Sharpe Ratio**: Risk-adjusted return metric, calculated annually.
- **Calmar Ratio**: Return-to-risk ratio, calculated annually.
- **Profit & Loss (P&L)**: Overall profit or loss achieved by the strategy.

The backtest was performed on 100 generated scenarios, testing the active trading strategy at various stop-loss and take-profit levels. Results are presented in comparison to the passive buy-and-hold strategy.

### Key Parameters:
- **Number of Shares**: Predefined based on the trading strategy.
- **Stop-Loss Levels**: 10 predefined levels for analysis.
- **Take-Profit Levels**: 10 predefined levels for analysis.
- **RSI Window**: Used for buy and sell signal generation.
- **RSI Thresholds**: Values indicating overbought and oversold conditions to trigger signals.

---

## Instructions to Run the Project

### 1. Clone the Repository

git clone git@github.com:Feryareli/003-Generative-Adversarial-Networks.git

### 2. Navigate to the Project Directory

cd 003-Generative-Adversarial-Networks

### 3. Set Up a Virtual Environment

- **Windows**: python -m venv venv
- **Mac/Linux**: python3 -m venv venv 

Activate the virtual environment:

- **Windows**: .\venv\Scripts\activate
- **Mac/Linux**: source venv/bin/activate

### 4. Install Required Dependencies

pip install -r requirements.txt

### 5. Run the Main Analysis

python gans_strategy/main.py

### 9. Run the Jupyter Notebook for Visualization

jupyter notebook report.ipynb

### 10. Deactivate the Virtual Environment

deactivate
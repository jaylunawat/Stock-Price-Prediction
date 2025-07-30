import yfinance as yf
import pandas as pd

# Change ticker and dates as desired
TICKER = 'MSFT'
START = '2017-01-01'
END = '2025-07-01'

# Fetch data
stock = yf.Ticker(TICKER)
df = stock.history(start=START, end=END)

# Keep needed columns only
df = df[['Close']]
df.dropna(inplace=True)

# Save for modeling
df.to_csv('stock_data.csv')
print('Data saved to stock_data.csv')

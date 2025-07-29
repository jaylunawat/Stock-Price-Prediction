import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataCollector:
    """
    A class to collect stock data using yfinance library
    """

    def __init__(self, data_path='data/raw/'):
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)

    def download_stock_data(self, ticker, period='5y', interval='1d'):
        """
        Download historical stock data for a given ticker

        Parameters:
        ticker (str): Stock ticker symbol
        period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
        pandas.DataFrame: Historical stock data
        """
        try:
            logger.info(f"Downloading data for {ticker}")
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)

            if data.empty:
                logger.warning(f"No data found for ticker {ticker}")
                return None

            # Add ticker column
            data['Ticker'] = ticker

            # Save to CSV
            filename = f"{ticker}_{period}_{interval}.csv"
            filepath = os.path.join(self.data_path, filename)
            data.to_csv(filepath)
            logger.info(f"Data saved to {filepath}")

            return data

        except Exception as e:
            logger.error(f"Error downloading data for {ticker}: {str(e)}")
            return None

    def download_multiple_stocks(self, tickers, period='5y', interval='1d'):
        """
        Download data for multiple stock tickers

        Parameters:
        tickers (list): List of stock ticker symbols
        period (str): Time period
        interval (str): Data interval

        Returns:
        dict: Dictionary with ticker as key and DataFrame as value
        """
        stock_data = {}

        for ticker in tickers:
            data = self.download_stock_data(ticker, period, interval)
            if data is not None:
                stock_data[ticker] = data

        return stock_data

    def get_stock_info(self, ticker):
        """
        Get stock information and metadata

        Parameters:
        ticker (str): Stock ticker symbol

        Returns:
        dict: Stock information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info
        except Exception as e:
            logger.error(f"Error getting info for {ticker}: {str(e)}")
            return None

    def download_sp500_list(self):
        """
        Download list of S&P 500 companies

        Returns:
        list: List of S&P 500 ticker symbols
        """
        try:
            # Read S&P 500 list from Wikipedia
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()

            # Clean ticker symbols
            tickers = [ticker.replace('.', '-') for ticker in tickers]

            return tickers
        except Exception as e:
            logger.error(f"Error downloading S&P 500 list: {str(e)}")
            return []

if __name__ == "__main__":
    # Example usage
    collector = StockDataCollector()

    # Download data for specific stocks
    popular_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META']
    stock_data = collector.download_multiple_stocks(popular_stocks)

    print(f"Downloaded data for {len(stock_data)} stocks")
    for ticker, data in stock_data.items():
        print(f"{ticker}: {len(data)} rows, from {data.index[0]} to {data.index[-1]}")

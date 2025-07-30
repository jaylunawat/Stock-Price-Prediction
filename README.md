# Stock-Price-Prediction

# Project Overview
This project demonstrates how to forecast a stock’s closing price by leveraging sequences of past closing prices.

# Project Structure
stock-forecast-lstm-alt/
-> get_data.py           # Script to fetch and prepare the market data
-> model_lstm.py         # Training, performance checks, and charts with LSTM
-> requirements.txt      # Libraries to be installed
-> stock_data.csv        # Data file (created by get_data.py)
-> plot_alt_lstm.png     # Plot (created by model_lstm.py)
-> README.md             # Project details and guide

# Working
1. Gathering Market Data:
The first script automatically pulls historical price information for any specified stock ticker using `yfinance`. By default, it covers Microsoft (MSFT) from 2017 until July 2025.
Run this with:
  		python get_data.py
2. Data Preprocessing:
Only the “Close” values are retained, ensuring the model operates on the most stable, widely used financial indicator. This filtered data is stored in `stock_data.csv`.
3. Windowing and Scaling:
Past closing prices from a fixed window (the last 45 days) serve as input features for forecasting the following day’s closing price. All features are scaled, which improves model stability and performance.
4. Building and Training the Model:
The LSTM model is constructed and trained on historical sequences. Early-stopping techniques are used to prevent the model from overfitting to the training data.
5. Model Assessment and Visualization:
After training, the script compares the model’s price predictions with actual unseen data and reports the Mean Absolute Error (MAE). A comparison plot of predicted vs. actual price movement is saved as `plot_alt_lstm.png`.

# Running
1. Install the Required Python Libraries
   pip install -r requirements.txt
2. Run data download and preparation:
   python get_data.py
This creates `stock_data.csv`.
3. Run Model Training and Evaluation
   python model_lstm.py
This command trains the model, evaluates its performance, saves the trained weights as `lstm_model.keras`, prints the MAE score, and writes `plot_alt_lstm.png` charting how well predictions track with true values.

# File Reference
	•	`get_data.py`: Downloads and saves stock data as CSV.
	•	`model_lstm.py`: Builds model, trains, evaluates, and plots results.
	•	`stock_data.csv`: Processed historical price data for modeling.
	•	`plot_alt_lstm.png`: Saved figure of predicted vs actual prices.
	•	`requirements.txt`: List of all Python libraries needed.

# Switching Stocks: Change the `TICKER` variable at the top of `get_data.py` to another symbol (e.g., ‘AAPL’, ‘TSLA’).
# Changing the Date Range: Modify `START` and `END` in `get_data.py` to select different training or forecasting periods.

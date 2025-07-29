import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation and backtesting for stock price prediction
    """

    def __init__(self):
        self.results = {}

    def calculate_metrics(self, y_true, y_pred, prefix=''):
        """
        Calculate comprehensive evaluation metrics

        Parameters:
        y_true (numpy.array): True values
        y_pred (numpy.array): Predicted values
        prefix (str): Prefix for metric names

        Returns:
        dict: Dictionary of metrics
        """
        metrics = {}

        # Basic regression metrics
        metrics[f'{prefix}mse'] = mean_squared_error(y_true, y_pred)
        metrics[f'{prefix}rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics[f'{prefix}mae'] = mean_absolute_error(y_true, y_pred)
        metrics[f'{prefix}r2'] = r2_score(y_true, y_pred)

        # Mean Absolute Percentage Error
        metrics[f'{prefix}mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # Directional Accuracy
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        metrics[f'{prefix}directional_accuracy'] = np.mean(true_direction == pred_direction) * 100

        # Max Error
        metrics[f'{prefix}max_error'] = np.max(np.abs(y_true - y_pred))

        return metrics

    def plot_predictions(self, y_true, y_pred, dates=None, title='Predictions vs Actual'):
        """
        Plot predictions against actual values

        Parameters:
        y_true (numpy.array): True values
        y_pred (numpy.array): Predicted values
        dates (pandas.DatetimeIndex): Dates for x-axis
        title (str): Plot title

        Returns:
        matplotlib.figure.Figure: Plot figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Time series plot
        if dates is not None:
            ax1.plot(dates, y_true, label='Actual', alpha=0.7)
            ax1.plot(dates, y_pred, label='Predicted', alpha=0.7)
        else:
            ax1.plot(y_true, label='Actual', alpha=0.7)
            ax1.plot(y_pred, label='Predicted', alpha=0.7)

        ax1.set_title(title)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Scatter plot
        ax2.scatter(y_true, y_pred, alpha=0.6)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.set_title('Predicted vs Actual Scatter Plot')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_residuals(self, y_true, y_pred, dates=None):
        """
        Plot residual analysis

        Parameters:
        y_true (numpy.array): True values
        y_pred (numpy.array): Predicted values
        dates (pandas.DatetimeIndex): Dates for x-axis

        Returns:
        matplotlib.figure.Figure: Residual plots
        """
        residuals = y_true - y_pred

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Residuals over time
        if dates is not None:
            ax1.plot(dates, residuals)
        else:
            ax1.plot(residuals)
        ax1.set_title('Residuals Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Residuals')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax1.grid(True, alpha=0.3)

        # Residuals vs Predicted
        ax2.scatter(y_pred, residuals, alpha=0.6)
        ax2.set_title('Residuals vs Predicted')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax2.grid(True, alpha=0.3)

        # Histogram of residuals
        ax3.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        ax3.set_title('Distribution of Residuals')
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Frequency')
        ax3.axvline(x=0, color='r', linestyle='--', alpha=0.8)
        ax3.grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot of Residuals')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def backtest_strategy(self, predictions, prices, dates, initial_capital=10000, 
                         transaction_cost=0.001, strategy_type='threshold'):
        """
        Backtest a trading strategy based on predictions

        Parameters:
        predictions (numpy.array): Model predictions
        prices (numpy.array): Actual prices
        dates (pandas.DatetimeIndex): Trading dates
        initial_capital (float): Initial capital
        transaction_cost (float): Transaction cost as a fraction
        strategy_type (str): Type of strategy ('threshold', 'directional')

        Returns:
        dict: Backtesting results
        """
        logger.info(f"Backtesting {strategy_type} strategy...")

        portfolio_value = initial_capital
        cash = initial_capital
        position = 0
        trades = []
        portfolio_history = []

        for i in range(1, len(predictions)):
            current_price = prices[i]
            current_date = dates[i] if dates is not None else i

            if strategy_type == 'threshold':
                # Buy if prediction is higher than current price by threshold
                if predictions[i] > prices[i-1] * 1.02 and position == 0:  # Buy signal
                    shares_to_buy = cash // (current_price * (1 + transaction_cost))
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price * (1 + transaction_cost)
                        cash -= cost
                        position = shares_to_buy
                        trades.append({
                            'date': current_date,
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares_to_buy,
                            'cost': cost
                        })

                # Sell if prediction is lower than current price by threshold
                elif predictions[i] < prices[i-1] * 0.98 and position > 0:  # Sell signal
                    revenue = position * current_price * (1 - transaction_cost)
                    cash += revenue
                    trades.append({
                        'date': current_date,
                        'action': 'SELL',
                        'price': current_price,
                        'shares': position,
                        'revenue': revenue
                    })
                    position = 0

            elif strategy_type == 'directional':
                # Simple directional strategy
                if i > 0:
                    predicted_direction = predictions[i] > predictions[i-1]

                    if predicted_direction and position == 0:  # Buy signal
                        shares_to_buy = cash // (current_price * (1 + transaction_cost))
                        if shares_to_buy > 0:
                            cost = shares_to_buy * current_price * (1 + transaction_cost)
                            cash -= cost
                            position = shares_to_buy
                            trades.append({
                                'date': current_date,
                                'action': 'BUY',
                                'price': current_price,
                                'shares': shares_to_buy,
                                'cost': cost
                            })

                    elif not predicted_direction and position > 0:  # Sell signal
                        revenue = position * current_price * (1 - transaction_cost)
                        cash += revenue
                        trades.append({
                            'date': current_date,
                            'action': 'SELL',
                            'price': current_price,
                            'shares': position,
                            'revenue': revenue
                        })
                        position = 0

            # Calculate current portfolio value
            current_portfolio_value = cash + (position * current_price)
            portfolio_history.append({
                'date': current_date,
                'portfolio_value': current_portfolio_value,
                'cash': cash,
                'position_value': position * current_price,
                'position_shares': position
            })

        # Final portfolio value
        if position > 0:
            final_price = prices[-1]
            cash += position * final_price * (1 - transaction_cost)
            portfolio_value = cash
        else:
            portfolio_value = cash

        # Calculate metrics
        total_return = (portfolio_value - initial_capital) / initial_capital * 100

        # Buy and hold return for comparison
        buy_hold_return = (prices[-1] - prices[0]) / prices[0] * 100

        # Calculate Sharpe ratio (simplified)
        portfolio_df = pd.DataFrame(portfolio_history)
        if len(portfolio_df) > 1:
            daily_returns = portfolio_df['portfolio_value'].pct_change().dropna()
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        results = {
            'initial_capital': initial_capital,
            'final_portfolio_value': portfolio_value,
            'total_return_pct': total_return,
            'buy_hold_return_pct': buy_hold_return,
            'excess_return_pct': total_return - buy_hold_return,
            'number_of_trades': len(trades),
            'sharpe_ratio': sharpe_ratio,
            'trades': trades,
            'portfolio_history': portfolio_history
        }

        logger.info(f"Backtest completed - Total Return: {total_return:.2f}%, Buy & Hold: {buy_hold_return:.2f}%")

        return results

    def plot_backtest_results(self, backtest_results, prices, dates):
        """
        Plot backtesting results

        Parameters:
        backtest_results (dict): Results from backtest_strategy
        prices (numpy.array): Stock prices
        dates (pandas.DatetimeIndex): Trading dates

        Returns:
        matplotlib.figure.Figure: Backtest plots
        """
        portfolio_df = pd.DataFrame(backtest_results['portfolio_history'])

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

        # Portfolio value over time
        ax1.plot(portfolio_df['date'], portfolio_df['portfolio_value'], 
                label='Strategy Portfolio', linewidth=2)

        # Buy and hold comparison
        initial_shares = backtest_results['initial_capital'] / prices[0]
        buy_hold_values = initial_shares * prices
        ax1.plot(dates, buy_hold_values, label='Buy & Hold', linewidth=2)

        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Stock price with buy/sell signals
        ax2.plot(dates, prices, label='Stock Price', alpha=0.7)

        trades_df = pd.DataFrame(backtest_results['trades'])
        if len(trades_df) > 0:
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']

            if len(buy_trades) > 0:
                ax2.scatter(buy_trades['date'], buy_trades['price'], 
                           color='green', marker='^', s=100, label='Buy Signal')
            if len(sell_trades) > 0:
                ax2.scatter(sell_trades['date'], sell_trades['price'], 
                           color='red', marker='v', s=100, label='Sell Signal')

        ax2.set_title('Stock Price with Trading Signals')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Returns comparison
        portfolio_returns = portfolio_df['portfolio_value'].pct_change().dropna()
        stock_returns = pd.Series(prices).pct_change().dropna()

        ax3.plot(portfolio_df['date'][1:], portfolio_returns.cumsum() * 100, 
                label='Strategy Cumulative Return', linewidth=2)
        ax3.plot(dates[1:], stock_returns.cumsum() * 100, 
                label='Buy & Hold Cumulative Return', linewidth=2)

        ax3.set_title('Cumulative Returns Comparison')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Cumulative Return (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def compare_models(self, results_dict, metric='rmse'):
        """
        Compare multiple models based on a specific metric

        Parameters:
        results_dict (dict): Dictionary with model names as keys and results as values
        metric (str): Metric to compare

        Returns:
        pandas.DataFrame: Comparison results
        """
        comparison_data = []

        for model_name, results in results_dict.items():
            if metric in results:
                comparison_data.append({
                    'model': model_name,
                    'metric': metric,
                    'value': results[metric]
                })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('value')

        return comparison_df

    def create_model_comparison_plot(self, results_dict, metrics=['rmse', 'mae', 'r2']):
        """
        Create a comparison plot for multiple models

        Parameters:
        results_dict (dict): Dictionary with model names as keys and results as values
        metrics (list): List of metrics to compare

        Returns:
        matplotlib.figure.Figure: Comparison plot
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))

        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            model_names = []
            values = []

            for model_name, results in results_dict.items():
                if metric in results:
                    model_names.append(model_name)
                    values.append(results[metric])

            if values:
                axes[i].bar(model_names, values)
                axes[i].set_title(f'{metric.upper()} Comparison')
                axes[i].set_ylabel(metric.upper())
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def generate_report(self, model_results, backtest_results=None):
        """
        Generate a comprehensive evaluation report

        Parameters:
        model_results (dict): Model evaluation results
        backtest_results (dict): Backtesting results (optional)

        Returns:
        str: Formatted report
        """
        report = "\n" + "="*60 + "\n"
        report += "STOCK PRICE PREDICTION MODEL EVALUATION REPORT\n"
        report += "="*60 + "\n\n"

        # Model Performance
        report += "MODEL PERFORMANCE METRICS:\n"
        report += "-"*30 + "\n"
        for metric, value in model_results.items():
            if isinstance(value, float):
                report += f"{metric.upper()}: {value:.4f}\n"
            else:
                report += f"{metric.upper()}: {value}\n"

        # Backtesting Results
        if backtest_results:
            report += "\nBACKTESTING RESULTS:\n"
            report += "-"*20 + "\n"
            report += f"Initial Capital: ${backtest_results['initial_capital']:,.2f}\n"
            report += f"Final Portfolio Value: ${backtest_results['final_portfolio_value']:,.2f}\n"
            report += f"Total Return: {backtest_results['total_return_pct']:.2f}%\n"
            report += f"Buy & Hold Return: {backtest_results['buy_hold_return_pct']:.2f}%\n"
            report += f"Excess Return: {backtest_results['excess_return_pct']:.2f}%\n"
            report += f"Number of Trades: {backtest_results['number_of_trades']}\n"
            report += f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.4f}\n"

        report += "\n" + "="*60 + "\n"

        return report

if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    y_true = np.cumsum(np.random.randn(100)) + 100
    y_pred = y_true + np.random.randn(100) * 0.5

    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

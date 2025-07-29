import pandas as pd
import numpy as np
import talib
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    A class to calculate various technical indicators for stock price prediction
    """

    def __init__(self):
        pass

    def add_moving_averages(self, df, periods=[5, 10, 20, 50, 200]):
        """
        Add Simple Moving Averages and Exponential Moving Averages

        Parameters:
        df (pandas.DataFrame): Stock data with OHLCV columns
        periods (list): List of periods for moving averages

        Returns:
        pandas.DataFrame: Data with moving average features
        """
        logger.info("Adding moving averages...")

        df = df.copy()

        for period in periods:
            # Simple Moving Average
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()

            # Exponential Moving Average
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()

            # Price relative to moving averages
            df[f'Close_SMA_{period}_Ratio'] = df['Close'] / df[f'SMA_{period}']
            df[f'Close_EMA_{period}_Ratio'] = df['Close'] / df[f'EMA_{period}']

        return df

    def add_momentum_indicators(self, df):
        """
        Add momentum indicators like RSI, MACD, Stochastic

        Parameters:
        df (pandas.DataFrame): Stock data with OHLCV columns

        Returns:
        pandas.DataFrame: Data with momentum indicators
        """
        logger.info("Adding momentum indicators...")

        df = df.copy()

        # RSI (Relative Strength Index)
        df['RSI_14'] = talib.RSI(df['Close'].values, timeperiod=14)
        df['RSI_30'] = talib.RSI(df['Close'].values, timeperiod=30)

        # MACD (Moving Average Convergence Divergence)
        macd, macdsignal, macdhist = talib.MACD(df['Close'].values, 
                                                fastperiod=12, 
                                                slowperiod=26, 
                                                signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = macdsignal
        df['MACD_Hist'] = macdhist

        # Stochastic Oscillator
        df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['High'].values,
                                                   df['Low'].values,
                                                   df['Close'].values,
                                                   fastk_period=14,
                                                   slowk_period=3,
                                                   slowd_period=3)

        # Williams %R
        df['Williams_R'] = talib.WILLR(df['High'].values,
                                       df['Low'].values,
                                       df['Close'].values,
                                       timeperiod=14)

        # Commodity Channel Index
        df['CCI'] = talib.CCI(df['High'].values,
                              df['Low'].values,
                              df['Close'].values,
                              timeperiod=14)

        # Rate of Change
        df['ROC_10'] = talib.ROC(df['Close'].values, timeperiod=10)
        df['ROC_30'] = talib.ROC(df['Close'].values, timeperiod=30)

        return df

    def add_volatility_indicators(self, df):
        """
        Add volatility indicators like Bollinger Bands, ATR

        Parameters:
        df (pandas.DataFrame): Stock data with OHLCV columns

        Returns:
        pandas.DataFrame: Data with volatility indicators
        """
        logger.info("Adding volatility indicators...")

        df = df.copy()

        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['Close'].values,
                                                                       timeperiod=20,
                                                                       nbdevup=2,
                                                                       nbdevdn=2)

        # Bollinger Band Width and Position
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # Average True Range
        df['ATR_14'] = talib.ATR(df['High'].values,
                                 df['Low'].values,
                                 df['Close'].values,
                                 timeperiod=14)

        # Historical Volatility
        df['HV_10'] = df['Close'].pct_change().rolling(window=10).std() * np.sqrt(252)
        df['HV_30'] = df['Close'].pct_change().rolling(window=30).std() * np.sqrt(252)

        return df

    def add_volume_indicators(self, df):
        """
        Add volume-based indicators

        Parameters:
        df (pandas.DataFrame): Stock data with OHLCV columns

        Returns:
        pandas.DataFrame: Data with volume indicators
        """
        logger.info("Adding volume indicators...")

        df = df.copy()

        # Volume Moving Averages
        df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_SMA_30'] = df['Volume'].rolling(window=30).mean()

        # Volume Ratios
        df['Volume_Ratio_10'] = df['Volume'] / df['Volume_SMA_10']
        df['Volume_Ratio_30'] = df['Volume'] / df['Volume_SMA_30']

        # On Balance Volume
        df['OBV'] = talib.OBV(df['Close'].values, df['Volume'].values)

        # Volume Price Trend
        df['VPT'] = talib.TRIX(df['Close'].values, timeperiod=14) * df['Volume']

        # Accumulation/Distribution Line
        df['AD'] = talib.AD(df['High'].values,
                            df['Low'].values,
                            df['Close'].values,
                            df['Volume'].values)

        # Chaikin A/D Oscillator
        df['ADOSC'] = talib.ADOSC(df['High'].values,
                                  df['Low'].values,
                                  df['Close'].values,
                                  df['Volume'].values,
                                  fastperiod=3,
                                  slowperiod=10)

        return df

    def add_pattern_recognition(self, df):
        """
        Add candlestick pattern recognition

        Parameters:
        df (pandas.DataFrame): Stock data with OHLCV columns

        Returns:
        pandas.DataFrame: Data with pattern recognition features
        """
        logger.info("Adding pattern recognition...")

        df = df.copy()

        # Major candlestick patterns
        patterns = {
            'DOJI': talib.CDLDOJI,
            'HAMMER': talib.CDLHAMMER,
            'HANGINGMAN': talib.CDLHANGINGMAN,
            'SHOOTINGSTAR': talib.CDLSHOOTINGSTAR,
            'ENGULFING': talib.CDLENGULFING,
            'HARAMI': talib.CDLHARAMI,
            'MORNINGSTAR': talib.CDLMORNINGSTAR,
            'EVENINGSTAR': talib.CDLEVENINGSTAR,
            'THREEWHITESOLDIERS': talib.CDL3WHITESOLDIERS,
            'THREEBLACKCROWS': talib.CDL3BLACKCROWS
        }

        for pattern_name, pattern_func in patterns.items():
            df[f'Pattern_{pattern_name}'] = pattern_func(df['Open'].values,
                                                         df['High'].values,
                                                         df['Low'].values,
                                                         df['Close'].values)

        return df

    def add_trend_indicators(self, df):
        """
        Add trend-following indicators

        Parameters:
        df (pandas.DataFrame): Stock data with OHLCV columns

        Returns:
        pandas.DataFrame: Data with trend indicators
        """
        logger.info("Adding trend indicators...")

        df = df.copy()

        # Average Directional Index
        df['ADX'] = talib.ADX(df['High'].values,
                              df['Low'].values,
                              df['Close'].values,
                              timeperiod=14)

        # Parabolic SAR
        df['SAR'] = talib.SAR(df['High'].values,
                              df['Low'].values,
                              acceleration=0.02,
                              maximum=0.2)

        # Aroon Oscillator
        df['AROON_Up'], df['AROON_Down'] = talib.AROON(df['High'].values,
                                                       df['Low'].values,
                                                       timeperiod=14)
        df['AROON_OSC'] = df['AROON_Up'] - df['AROON_Down']

        # Plus/Minus Directional Indicators
        df['PLUS_DI'] = talib.PLUS_DI(df['High'].values,
                                      df['Low'].values,
                                      df['Close'].values,
                                      timeperiod=14)

        df['MINUS_DI'] = talib.MINUS_DI(df['High'].values,
                                        df['Low'].values,
                                        df['Close'].values,
                                        timeperiod=14)

        return df

    def add_all_indicators(self, df):
        """
        Add all technical indicators

        Parameters:
        df (pandas.DataFrame): Stock data with OHLCV columns

        Returns:
        pandas.DataFrame: Data with all technical indicators
        """
        logger.info("Adding all technical indicators...")

        df = self.add_moving_averages(df)
        df = self.add_momentum_indicators(df)
        df = self.add_volatility_indicators(df)
        df = self.add_volume_indicators(df)
        df = self.add_pattern_recognition(df)
        df = self.add_trend_indicators(df)

        # Remove infinite values and replace with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Forward fill any remaining NaN values
        df = df.fillna(method='ffill')

        # Drop any remaining NaN rows
        df = df.dropna()

        logger.info(f"Final feature count: {len(df.columns)}")

        return df

    def get_feature_importance_data(self, df, target_col='Target'):
        """
        Prepare data for feature importance analysis

        Parameters:
        df (pandas.DataFrame): Data with features and target
        target_col (str): Name of target column

        Returns:
        tuple: X (features), y (target), feature_names
        """
        # Get feature columns (exclude target and ticker)
        feature_cols = [col for col in df.columns 
                       if col not in [target_col, 'Ticker'] 
                       and not col.startswith('Pattern_')]  # Exclude pattern columns for importance

        X = df[feature_cols].values
        y = df[target_col].values

        return X, y, feature_cols

if __name__ == "__main__":
    # Example usage
    import yfinance as yf

    # Download sample data
    data = yf.Ticker('AAPL').history(period='2y')

    # Initialize technical indicators
    tech_indicators = TechnicalIndicators()

    # Add all indicators
    enhanced_data = tech_indicators.add_all_indicators(data)

    print(f"Original columns: {len(data.columns)}")
    print(f"Enhanced columns: {len(enhanced_data.columns)}")
    print(f"New features: {[col for col in enhanced_data.columns if col not in data.columns]}")

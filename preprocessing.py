import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class StockDataPreprocessor:
    """
    A class to preprocess stock data for machine learning models
    """

    def __init__(self):
        self.scaler = None

    def clean_data(self, df):
        """
        Clean the stock data by handling missing values and outliers

        Parameters:
        df (pandas.DataFrame): Raw stock data

        Returns:
        pandas.DataFrame: Cleaned stock data
        """
        logger.info("Cleaning stock data...")

        # Make a copy to avoid modifying original data
        cleaned_df = df.copy()

        # Remove rows with all NaN values
        cleaned_df = cleaned_df.dropna(how='all')

        # Forward fill missing values (common practice for stock data)
        cleaned_df = cleaned_df.fillna(method='ffill')

        # Remove any remaining NaN values
        cleaned_df = cleaned_df.dropna()

        # Remove outliers using IQR method
        for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if column in cleaned_df.columns:
                Q1 = cleaned_df[column].quantile(0.25)
                Q3 = cleaned_df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Cap outliers instead of removing them
                cleaned_df[column] = np.clip(cleaned_df[column], lower_bound, upper_bound)

        logger.info(f"Data cleaned: {len(cleaned_df)} rows remaining")
        return cleaned_df

    def add_basic_features(self, df):
        """
        Add basic derived features to the dataset

        Parameters:
        df (pandas.DataFrame): Stock data

        Returns:
        pandas.DataFrame: Data with additional features
        """
        logger.info("Adding basic features...")

        df = df.copy()

        # Price-based features
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Open_Close_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100

        # Returns
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

        # Price position within the day's range
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])

        # Volume features
        if 'Volume' in df.columns:
            df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_10']

        # Volatility (rolling standard deviation of returns)
        df['Volatility_10'] = df['Daily_Return'].rolling(window=10).std()
        df['Volatility_30'] = df['Daily_Return'].rolling(window=30).std()

        return df

    def create_target_variable(self, df, target_days=1, target_type='price'):
        """
        Create target variable for prediction

        Parameters:
        df (pandas.DataFrame): Stock data
        target_days (int): Number of days ahead to predict
        target_type (str): Type of target ('price', 'return', 'direction')

        Returns:
        pandas.DataFrame: Data with target variable
        """
        logger.info(f"Creating target variable: {target_type} for {target_days} days ahead")

        df = df.copy()

        if target_type == 'price':
            df['Target'] = df['Close'].shift(-target_days)
        elif target_type == 'return':
            df['Target'] = (df['Close'].shift(-target_days) - df['Close']) / df['Close']
        elif target_type == 'direction':
            future_price = df['Close'].shift(-target_days)
            df['Target'] = (future_price > df['Close']).astype(int)

        # Remove rows where target is NaN
        df = df.dropna()

        return df

    def scale_features(self, X_train, X_test=None, scaler_type='standard'):
        """
        Scale features using StandardScaler or MinMaxScaler

        Parameters:
        X_train (pandas.DataFrame): Training features
        X_test (pandas.DataFrame): Testing features (optional)
        scaler_type (str): Type of scaler ('standard' or 'minmax')

        Returns:
        tuple: Scaled training data, scaled testing data (if provided), scaler object
        """
        logger.info(f"Scaling features using {scaler_type} scaler...")

        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")

        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            return X_train_scaled, X_test_scaled, self.scaler

        return X_train_scaled, self.scaler

    def prepare_sequences(self, data, sequence_length, target_col='Target'):
        """
        Prepare sequences for LSTM model

        Parameters:
        data (pandas.DataFrame): Stock data
        sequence_length (int): Length of input sequences
        target_col (str): Name of target column

        Returns:
        tuple: X sequences, y targets
        """
        logger.info(f"Preparing sequences with length {sequence_length}")

        # Get feature columns (exclude target)
        feature_cols = [col for col in data.columns if col != target_col and col != 'Ticker']

        X, y = [], []

        for i in range(sequence_length, len(data)):
            # Get sequence of features
            X.append(data[feature_cols].iloc[i-sequence_length:i].values)
            # Get target value
            y.append(data[target_col].iloc[i])

        return np.array(X), np.array(y)

    def train_test_split_temporal(self, df, test_size=0.2, val_size=0.1):
        """
        Split data temporally (time-based split)

        Parameters:
        df (pandas.DataFrame): Stock data
        test_size (float): Proportion of data for testing
        val_size (float): Proportion of data for validation

        Returns:
        tuple: train_df, val_df, test_df
        """
        logger.info("Performing temporal train-test split...")

        total_size = len(df)
        test_start = int(total_size * (1 - test_size))
        val_start = int(total_size * (1 - test_size - val_size))

        train_df = df.iloc[:val_start].copy()
        val_df = df.iloc[val_start:test_start].copy()
        test_df = df.iloc[test_start:].copy()

        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

if __name__ == "__main__":
    # Example usage
    preprocessor = StockDataPreprocessor()

    # Load sample data (you would load your actual data here)
    import yfinance as yf
    data = yf.Ticker('AAPL').history(period='2y')

    # Clean and preprocess data
    cleaned_data = preprocessor.clean_data(data)
    featured_data = preprocessor.add_basic_features(cleaned_data)
    target_data = preprocessor.create_target_variable(featured_data, target_days=1, target_type='return')

    print(f"Original data shape: {data.shape}")
    print(f"Processed data shape: {target_data.shape}")
    print(f"Features: {target_data.columns.tolist()}")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import os

logger = logging.getLogger(__name__)

class LSTMModels:
    """
    LSTM-based models for stock price prediction
    """

    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.models = {}
        self.model_history = {}

        # Set random seed for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

    def create_simple_lstm(self, input_shape, units=50, dropout=0.2):
        """
        Create a simple LSTM model

        Parameters:
        input_shape (tuple): Shape of input data (sequence_length, features)
        units (int): Number of LSTM units
        dropout (float): Dropout rate

        Returns:
        tensorflow.keras.Model: LSTM model
        """
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout),
            LSTM(units, return_sequences=False),
            Dropout(dropout),
            Dense(25, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

        return model

    def create_stacked_lstm(self, input_shape, units=[50, 50, 25], dropout=0.2):
        """
        Create a stacked LSTM model

        Parameters:
        input_shape (tuple): Shape of input data
        units (list): List of units for each LSTM layer
        dropout (float): Dropout rate

        Returns:
        tensorflow.keras.Model: Stacked LSTM model
        """
        model = Sequential()

        # First LSTM layer
        model.add(LSTM(units[0], return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))

        # Additional LSTM layers
        for i in range(1, len(units)):
            return_sequences = i < len(units) - 1
            model.add(LSTM(units[i], return_sequences=return_sequences))
            model.add(Dropout(dropout))

        # Dense layers
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

        return model

    def create_bidirectional_lstm(self, input_shape, units=50, dropout=0.2):
        """
        Create a bidirectional LSTM model

        Parameters:
        input_shape (tuple): Shape of input data
        units (int): Number of LSTM units
        dropout (float): Dropout rate

        Returns:
        tensorflow.keras.Model: Bidirectional LSTM model
        """
        model = Sequential([
            Bidirectional(LSTM(units, return_sequences=True), input_shape=input_shape),
            Dropout(dropout),
            Bidirectional(LSTM(units, return_sequences=False)),
            Dropout(dropout),
            Dense(25, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

        return model

    def create_attention_lstm(self, input_shape, units=50, dropout=0.2):
        """
        Create an LSTM model with attention mechanism

        Parameters:
        input_shape (tuple): Shape of input data
        units (int): Number of LSTM units
        dropout (float): Dropout rate

        Returns:
        tensorflow.keras.Model: LSTM model with attention
        """
        inputs = Input(shape=input_shape)

        # LSTM layers
        lstm_out = LSTM(units, return_sequences=True)(inputs)
        lstm_out = Dropout(dropout)(lstm_out)

        # Attention mechanism (simplified)
        attention_weights = Dense(1, activation='tanh')(lstm_out)
        attention_weights = tf.nn.softmax(attention_weights, axis=1)

        # Apply attention
        context_vector = tf.reduce_sum(attention_weights * lstm_out, axis=1)

        # Dense layers
        dense_out = Dense(25, activation='relu')(context_vector)
        outputs = Dense(1)(dense_out)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

        return model

    def prepare_sequences(self, data, target_col='Target'):
        """
        Prepare sequences for LSTM training

        Parameters:
        data (pandas.DataFrame): Stock data
        target_col (str): Name of target column

        Returns:
        tuple: X sequences, y targets
        """
        logger.info(f"Preparing sequences with length {self.sequence_length}")

        # Get feature columns (exclude target and ticker)
        feature_cols = [col for col in data.columns 
                       if col not in [target_col, 'Ticker'] 
                       and not data[col].dtype == 'object']

        X, y = [], []

        for i in range(self.sequence_length, len(data)):
            # Get sequence of features
            X.append(data[feature_cols].iloc[i-self.sequence_length:i].values)
            # Get target value
            y.append(data[target_col].iloc[i])

        return np.array(X), np.array(y)

    def train_model(self, X_train, y_train, X_val=None, y_val=None, 
                   model_type='simple', model_name='lstm_model', 
                   epochs=100, batch_size=32, verbose=1):
        """
        Train an LSTM model

        Parameters:
        X_train (numpy.array): Training sequences
        y_train (numpy.array): Training targets
        X_val (numpy.array): Validation sequences
        y_val (numpy.array): Validation targets
        model_type (str): Type of LSTM model
        model_name (str): Name for the model
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        verbose (int): Verbosity level

        Returns:
        tensorflow.keras.Model: Trained model
        """
        logger.info(f"Training {model_type} LSTM model...")

        input_shape = (X_train.shape[1], X_train.shape[2])

        # Create model based on type
        if model_type == 'simple':
            model = self.create_simple_lstm(input_shape)
        elif model_type == 'stacked':
            model = self.create_stacked_lstm(input_shape)
        elif model_type == 'bidirectional':
            model = self.create_bidirectional_lstm(input_shape)
        elif model_type == 'attention':
            model = self.create_attention_lstm(input_shape)
        else:
            raise ValueError(f"Model type {model_type} not supported")

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]

        if not os.path.exists('models/checkpoints'):
            os.makedirs('models/checkpoints')

        callbacks.append(
            ModelCheckpoint(
                f'models/checkpoints/{model_name}_best.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True
            )
        )

        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            validation_split = 0.2

        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=0.2 if validation_data is None else 0,
            callbacks=callbacks,
            verbose=verbose
        )

        # Store model and history
        self.models[model_name] = model
        self.model_history[model_name] = history.history

        logger.info(f"{model_type} LSTM model training completed")

        return model

    def predict(self, X_test, model_name='lstm_model'):
        """
        Make predictions using trained model

        Parameters:
        X_test (numpy.array): Test sequences
        model_name (str): Name of the model to use

        Returns:
        numpy.array: Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} has not been trained")

        model = self.models[model_name]
        predictions = model.predict(X_test, verbose=0)

        return predictions.flatten()

    def evaluate_model(self, X_test, y_test, model_name='lstm_model'):
        """
        Evaluate model performance

        Parameters:
        X_test (numpy.array): Test sequences
        y_test (numpy.array): True values
        model_name (str): Name of the model to evaluate

        Returns:
        dict: Evaluation metrics
        """
        predictions = self.predict(X_test, model_name)

        metrics = {
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions),
            'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
        }

        logger.info(f"{model_name} evaluation - RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}")

        return metrics

    def plot_training_history(self, model_name='lstm_model'):
        """
        Plot training history

        Parameters:
        model_name (str): Name of the model

        Returns:
        matplotlib.figure.Figure: Training history plot
        """
        if model_name not in self.model_history:
            raise ValueError(f"Training history not available for {model_name}")

        import matplotlib.pyplot as plt

        history = self.model_history[model_name]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        ax1.plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot MAE
        ax2.plot(history['mae'], label='Training MAE')
        if 'val_mae' in history:
            ax2.plot(history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()

        plt.tight_layout()
        return fig

    def save_model(self, model_name, filepath):
        """
        Save trained model to disk

        Parameters:
        model_name (str): Name of model to save
        filepath (str): Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} has not been trained")

        self.models[model_name].save(filepath)
        logger.info(f"Model {model_name} saved to {filepath}")

    def load_model(self, model_name, filepath):
        """
        Load trained model from disk

        Parameters:
        model_name (str): Name for the loaded model
        filepath (str): Path to load the model from
        """
        model = tf.keras.models.load_model(filepath)
        self.models[model_name] = model
        logger.info(f"Model {model_name} loaded from {filepath}")

    def predict_future(self, last_sequence, n_steps=5, model_name='lstm_model'):
        """
        Predict future values using the last sequence

        Parameters:
        last_sequence (numpy.array): Last sequence of features
        n_steps (int): Number of future steps to predict
        model_name (str): Name of the model to use

        Returns:
        numpy.array: Future predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} has not been trained")

        model = self.models[model_name]
        predictions = []

        current_sequence = last_sequence.copy()

        for _ in range(n_steps):
            # Predict next value
            pred = model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)[0][0]
            predictions.append(pred)

            # Update sequence (simplified - in practice, you'd need to update all features)
            # This is a basic implementation and may need refinement based on your specific use case
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = pred  # Assuming first feature is the price

        return np.array(predictions)

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression
    import numpy as np

    # Generate sample sequential data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

    # Reshape for sequence format
    sequence_length = 60
    X_seq, y_seq = [], []

    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    # Split data
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    # Initialize and train LSTM
    lstm_models = LSTMModels(sequence_length=sequence_length)
    model = lstm_models.train_model(X_train, y_train, model_type='simple', epochs=10)

    # Evaluate model
    metrics = lstm_models.evaluate_model(X_test, y_test)
    print("LSTM Model Evaluation:")
    print(metrics)

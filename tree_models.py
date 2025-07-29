import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging

logger = logging.getLogger(__name__)

class TreeModels:
    """
    Tree-based models for stock price prediction
    """

    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        }
        self.fitted_models = {}
        self.feature_importance = {}

    def train_model(self, X_train, y_train, model_type='random_forest', **kwargs):
        """
        Train a tree-based model

        Parameters:
        X_train (numpy.array): Training features
        y_train (numpy.array): Training targets
        model_type (str): Type of model to train
        **kwargs: Additional parameters for the model

        Returns:
        object: Trained model
        """
        logger.info(f"Training {model_type} model...")

        if model_type not in self.models:
            raise ValueError(f"Model type {model_type} not supported")

        # Update model parameters if provided
        model = self.models[model_type]
        if kwargs:
            model.set_params(**kwargs)

        # Handle XGBoost and LightGBM specific training
        if model_type == 'xgboost':
            model.fit(X_train, y_train, eval_metric='rmse', verbose=False)
        elif model_type == 'lightgbm':
            model.fit(X_train, y_train, eval_metric='rmse')
        else:
            model.fit(X_train, y_train)

        self.fitted_models[model_type] = model

        # Store feature importance
        if hasattr(model, 'feature_importances_'):
            self.feature_importance[model_type] = model.feature_importances_

        logger.info(f"{model_type} model training completed")
        return model

    def predict(self, X_test, model_type='random_forest'):
        """
        Make predictions using trained model

        Parameters:
        X_test (numpy.array): Test features
        model_type (str): Type of model to use

        Returns:
        numpy.array: Predictions
        """
        if model_type not in self.fitted_models:
            raise ValueError(f"Model {model_type} has not been trained")

        model = self.fitted_models[model_type]
        predictions = model.predict(X_test)

        return predictions

    def evaluate_model(self, X_test, y_test, model_type='random_forest'):
        """
        Evaluate model performance

        Parameters:
        X_test (numpy.array): Test features
        y_test (numpy.array): True values
        model_type (str): Type of model to evaluate

        Returns:
        dict: Evaluation metrics
        """
        predictions = self.predict(X_test, model_type)

        metrics = {
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions),
            'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
        }

        logger.info(f"{model_type} model evaluation - RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}")

        return metrics

    def train_all_models(self, X_train, y_train):
        """
        Train all tree-based models

        Parameters:
        X_train (numpy.array): Training features
        y_train (numpy.array): Training targets

        Returns:
        dict: Dictionary of trained models
        """
        logger.info("Training all tree-based models...")

        for model_type in self.models.keys():
            try:
                self.train_model(X_train, y_train, model_type)
            except Exception as e:
                logger.error(f"Error training {model_type}: {str(e)}")
                continue

        return self.fitted_models

    def compare_models(self, X_test, y_test):
        """
        Compare performance of all trained models

        Parameters:
        X_test (numpy.array): Test features
        y_test (numpy.array): True values

        Returns:
        pandas.DataFrame: Comparison results
        """
        results = []

        for model_type in self.fitted_models.keys():
            try:
                metrics = self.evaluate_model(X_test, y_test, model_type)
                metrics['model'] = model_type
                results.append(metrics)
            except Exception as e:
                logger.error(f"Error evaluating {model_type}: {str(e)}")
                continue

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values('rmse')

        return results_df

    def get_feature_importance(self, feature_names, model_type='random_forest', top_n=20):
        """
        Get feature importance for a model

        Parameters:
        feature_names (list): List of feature names
        model_type (str): Type of model
        top_n (int): Number of top features to return

        Returns:
        pandas.DataFrame: Feature importance
        """
        if model_type not in self.feature_importance:
            raise ValueError(f"Feature importance not available for {model_type}")

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance[model_type]
        })

        importance_df = importance_df.sort_values('importance', ascending=False)

        return importance_df.head(top_n)

    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val, model_type='random_forest'):
        """
        Perform basic hyperparameter tuning

        Parameters:
        X_train (numpy.array): Training features
        y_train (numpy.array): Training targets
        X_val (numpy.array): Validation features
        y_val (numpy.array): Validation targets
        model_type (str): Type of model to tune

        Returns:
        dict: Best parameters and score
        """
        logger.info(f"Performing hyperparameter tuning for {model_type}...")

        best_score = float('inf')
        best_params = {}

        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
        elif model_type == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            }
        elif model_type == 'lightgbm':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            }
        else:
            logger.warning(f"Hyperparameter tuning not implemented for {model_type}")
            return {}

        # Simple grid search
        for params in self._generate_param_combinations(param_grid):
            try:
                self.train_model(X_train, y_train, model_type, **params)
                predictions = self.predict(X_val, model_type)
                score = mean_squared_error(y_val, predictions)

                if score < best_score:
                    best_score = score
                    best_params = params

            except Exception as e:
                logger.error(f"Error with parameters {params}: {str(e)}")
                continue

        logger.info(f"Best parameters for {model_type}: {best_params}")
        logger.info(f"Best validation RMSE: {np.sqrt(best_score):.4f}")

        return {'best_params': best_params, 'best_score': best_score}

    def _generate_param_combinations(self, param_grid):
        """
        Generate parameter combinations for grid search
        """
        import itertools

        keys = param_grid.keys()
        values = param_grid.values()

        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

    def save_model(self, model_type, filepath):
        """
        Save trained model to disk

        Parameters:
        model_type (str): Type of model to save
        filepath (str): Path to save the model
        """
        if model_type not in self.fitted_models:
            raise ValueError(f"Model {model_type} has not been trained")

        joblib.dump(self.fitted_models[model_type], filepath)
        logger.info(f"Model {model_type} saved to {filepath}")

    def load_model(self, model_type, filepath):
        """
        Load trained model from disk

        Parameters:
        model_type (str): Type of model
        filepath (str): Path to load the model from
        """
        model = joblib.load(filepath)
        self.fitted_models[model_type] = model
        logger.info(f"Model {model_type} loaded from {filepath}")

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    # Generate sample data
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train models
    tree_models = TreeModels()
    tree_models.train_all_models(X_train, y_train)

    # Compare models
    comparison = tree_models.compare_models(X_test, y_test)
    print("Model Comparison:")
    print(comparison)

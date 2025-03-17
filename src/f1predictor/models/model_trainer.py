import logging
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

logger = logging.getLogger("F1Predictor.ModelTrainer")


class F1ModelTrainer:
    def __init__(self, models_dir="models"):
        """
        Initialize the model trainer

        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = models_dir
        self.feature_importance = {}
        self.model = None
        self.feature_columns = [
            'Position',
            'QualifyingPerformance',
            'DNFProbability',
            'RacePaceScore'
        ]
        self.target_column = 'ProjectedPosition'
        self.scaler = StandardScaler()

        # Create models directory if it doesn't exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            logger.info(f"Created models directory: {models_dir}")

    def _prepare_data(self, features_df, target_column="ProjectedPosition"):
        """
        Prepare data for model training

        Args:
            features_df: DataFrame with features and target
            target_column: Target column name

        Returns:
            X, y, feature_names
        """
        try:
            if features_df.empty or target_column not in features_df.columns:
                logger.error(f"Invalid data or missing target column: {target_column}")
                return None, None, None

            # Debug: Print column data types
            logger.debug("DataFrame column types:")
            for col in features_df.columns:
                logger.debug(f"{col}: {features_df[col].dtype}")

            # Select features (exclude any non-feature columns)
            non_feature_cols = set(
                [
                    target_column,
                    "DriverId",
                    "FullName",
                    "TeamName",
                    "Status",
                    "CompletedLaps",
                    "TotalRaceTime",
                    "FormattedGap",
                    "FormattedRaceTime",
                    "Points",
                    "FastestLap",
                    "GapToWinner",
                    "FormattedPositionChange",
                    "Position",
                    "BestTime",
                    "GapToPole",  # Additional qualifying columns to exclude
                ]
            )

            # Convert numeric columns to float first
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            logger.debug(f"Converting numeric columns to float: {list(numeric_cols)}")

            # Create a copy of the DataFrame to avoid modifying the original
            features_df = features_df.copy()

            for col in numeric_cols:
                try:
                    features_df[col] = pd.to_numeric(features_df[col], errors="coerce").astype(
                        float
                    )
                except Exception as e:
                    logger.error(f"Error converting column {col} to float: {e}")
                    logger.debug(f"Sample values from {col}: {features_df[col].head()}")

            # Select feature columns (only numeric columns that are not in non_feature_cols)
            feature_cols = [col for col in numeric_cols if col not in non_feature_cols]

            # Log the features being used
            logger.info(f"Using {len(feature_cols)} features for model training")
            logger.debug(f"Selected features: {', '.join(str(col) for col in feature_cols)}")

            # Handle missing values
            X = features_df[feature_cols].copy()
            X = X.fillna(X.mean())

            # Debug: Print feature matrix info
            logger.debug(f"Feature matrix shape: {X.shape}")
            logger.debug("Feature matrix data types:")
            for col in X.columns:
                logger.debug(f"{col}: {X[col].dtype}")

            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Prepare target
            y = pd.to_numeric(features_df[target_column], errors="coerce").astype(float)

            # Debug: Print target info
            logger.debug(f"Target values range: [{y.min()}, {y.max()}]")
            logger.debug(f"Target dtype: {y.dtype}")

            return X_scaled, y, feature_cols
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            logger.debug("Exception details:", exc_info=True)
            return None, None, None

    def train_model(self, race_features: pd.DataFrame, model_type: str = "xgboost", tune_hyperparams: bool = False, target_column: str = "ProjectedPosition"):
        """
        Train a model using the provided race features.

        Args:
            race_features: DataFrame or dictionary containing race features
            model_type: Type of model to train ("xgboost", "gradient_boosting", "random_forest")
            tune_hyperparams: Whether to perform hyperparameter tuning
            target_column: Target column for prediction (default: "ProjectedPosition")

        Returns:
            Trained model
        """
        try:
            # Check if race_features is None, empty DataFrame, or empty dictionary
            if race_features is None or (isinstance(race_features, pd.DataFrame) and race_features.empty) or (isinstance(race_features, dict) and not race_features):
                logger.error("No features provided for training")
                return None

            # Convert dictionary to DataFrame if needed
            if isinstance(race_features, dict):
                race_features = pd.DataFrame(race_features)

            # Prepare features and target
            X = race_features[self.feature_columns]
            y = race_features[target_column] if target_column in race_features.columns else race_features['GridPosition']

            # Handle missing values
            X = X.fillna(X.mean())

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Select and train model
            if model_type == "xgboost":
                if tune_hyperparams:
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 4, 5],
                        'learning_rate': [0.01, 0.1, 0.3]
                    }
                    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
                else:
                    model = xgb.XGBRegressor(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.1,
                        objective='reg:squarederror',
                        random_state=42
                    )
            elif model_type == "gradient_boosting":
                if tune_hyperparams:
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 4, 5],
                        'learning_rate': [0.01, 0.1, 0.3]
                    }
                    model = GradientBoostingRegressor(random_state=42)
                else:
                    model = GradientBoostingRegressor(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.1,
                        random_state=42
                    )
            else:  # random_forest
                if tune_hyperparams:
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 4, 5],
                        'min_samples_split': [2, 5, 10]
                    }
                    model = RandomForestRegressor(random_state=42)
                else:
                    model = RandomForestRegressor(
                        n_estimators=200,
                        max_depth=4,
                        random_state=42
                    )

            # Perform hyperparameter tuning if requested
            if tune_hyperparams:
                grid_search = GridSearchCV(
                    model,
                    param_grid,
                    cv=5,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")
            else:
                model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            logger.info(f"Model performance - MSE: {mse:.4f}, R2: {r2:.4f}")

            self.model = model
            return model

        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None

    def compare_models(self, race_features: pd.DataFrame, tune_hyperparams: bool = False):
        """
        Compare different model types and return the best performing one.

        Args:
            race_features: DataFrame containing race features
            tune_hyperparams: Whether to perform hyperparameter tuning

        Returns:
            Best performing model
        """
        try:
            model_types = ["xgboost", "gradient_boosting", "random_forest"]
            best_model = None
            best_score = float('inf')

            for model_type in model_types:
                logger.info(f"Training {model_type} model...")
                model = self.train_model(race_features, model_type, tune_hyperparams)
                
                if model is not None:
                    # Evaluate model
                    X = race_features[self.feature_columns]
                    y = race_features[self.target_column] if self.target_column in race_features.columns else race_features['GridPosition']
                    X_scaled = self.scaler.transform(X)
                    y_pred = model.predict(X_scaled)
                    score = mean_squared_error(y, y_pred)
                    
                    logger.info(f"{model_type} model MSE: {score:.4f}")
                    
                    if score < best_score:
                        best_score = score
                        best_model = model

            if best_model is not None:
                logger.info(f"Best model selected with MSE: {best_score:.4f}")
                self.model = best_model
                return best_model
            else:
                logger.error("No models were successfully trained")
                return None

        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return None

    def train_position_model(
        self,
        features_df,
        model_type="xgboost",
        target_column="ProjectedPosition",
        tune_hyperparams=True,
    ):
        """
        Train a model to predict finishing positions

        Args:
            features_df: DataFrame with features and target
            model_type: Type of model to train
            target_column: Target column name
            tune_hyperparams: Whether to tune hyperparameters

        Returns:
            Trained model
        """
        try:
            # Prepare data
            X, y, feature_names = self._prepare_data(features_df, target_column)

            if X is None or y is None:
                logger.error("Failed to prepare data for training")
                return None

            logger.info(f"Training {model_type} model for {target_column} prediction")
            logger.info(f"Data shape: X={X.shape}, y={len(y)}")

            # Initialize model based on type
            if model_type == "xgboost":
                model = self._train_xgboost(X, y, tune_hyperparams)
            elif model_type == "gradient_boosting":
                model = self._train_gradient_boosting(X, y, tune_hyperparams)
            elif model_type == "random_forest":
                model = self._train_random_forest(X, y, tune_hyperparams)
            elif model_type == "ridge":
                model = self._train_ridge(X, y, tune_hyperparams)
            elif model_type == "lasso":
                model = self._train_lasso(X, y, tune_hyperparams)
            elif model_type == "svr":
                model = self._train_svr(X, y, tune_hyperparams)
            else:
                logger.error(f"Unknown model type: {model_type}")
                return None

            # Calculate feature importance if available
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
                self.feature_importance = {
                    feature_names[i]: importance[i] for i in range(len(feature_names))
                }

                # Log top 10 features
                top_features = sorted(
                    self.feature_importance.items(), key=lambda x: x[1], reverse=True
                )[:10]
                logger.info("Top 10 features by importance:")
                for feature, importance in top_features:
                    logger.info(f"  {feature}: {importance:.4f}")

            # Save model
            model_path = os.path.join(
                self.models_dir,
                f"{model_type}_{target_column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
            )
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            logger.info(f"Model saved to {model_path}")

            return model
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None

    def _train_xgboost(self, X, y, tune_hyperparams=True):
        """
        Train XGBoost model

        Args:
            X: Features
            y: Target
            tune_hyperparams: Whether to tune hyperparameters

        Returns:
            Trained model
        """
        try:
            if tune_hyperparams:
                # Define parameter grid
                param_grid = {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "min_child_weight": [1, 3, 5],
                    "subsample": [0.8, 0.9, 1.0],
                    "colsample_bytree": [0.8, 0.9, 1.0],
                }

                # Use TimeSeriesSplit for time series data
                tscv = TimeSeriesSplit(n_splits=3)

                # Init model
                model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

                # Perform grid search
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring="neg_mean_squared_error",
                    cv=tscv,
                    verbose=1,
                    n_jobs=-1,
                )

                grid_search.fit(X, y)

                # Get best model
                best_model = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")

                return best_model
            else:
                # Use default parameters
                model = xgb.XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42,
                )

                model.fit(X, y)

                return model
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            return None

    def _train_gradient_boosting(self, X, y, tune_hyperparams=True):
        """
        Train Gradient Boosting model

        Args:
            X: Features
            y: Target
            tune_hyperparams: Whether to tune hyperparameters

        Returns:
            Trained model
        """
        try:
            if tune_hyperparams:
                # Define parameter grid
                param_grid = {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                }

                # Use TimeSeriesSplit for time series data
                tscv = TimeSeriesSplit(n_splits=3)

                # Init model
                model = GradientBoostingRegressor(random_state=42)

                # Perform grid search
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring="neg_mean_squared_error",
                    cv=tscv,
                    verbose=1,
                    n_jobs=-1,
                )

                grid_search.fit(X, y)

                # Get best model
                best_model = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")

                return best_model
            else:
                # Use default parameters
                model = GradientBoostingRegressor(
                    n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
                )

                model.fit(X, y)

                return model
        except Exception as e:
            logger.error(f"Error training Gradient Boosting model: {e}")
            return None

    def _train_random_forest(self, X, y, tune_hyperparams=True):
        """
        Train Random Forest model

        Args:
            X: Features
            y: Target
            tune_hyperparams: Whether to tune hyperparameters

        Returns:
            Trained model
        """
        try:
            if tune_hyperparams:
                # Define parameter grid
                param_grid = {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 5, 10, 15],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                }

                # Use TimeSeriesSplit for time series data
                tscv = TimeSeriesSplit(n_splits=3)

                # Init model
                model = RandomForestRegressor(random_state=42)

                # Perform grid search
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring="neg_mean_squared_error",
                    cv=tscv,
                    verbose=1,
                    n_jobs=-1,
                )

                grid_search.fit(X, y)

                # Get best model
                best_model = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")

                return best_model
            else:
                # Use default parameters
                model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42)

                model.fit(X, y)

                return model
        except Exception as e:
            logger.error(f"Error training Random Forest model: {e}")
            return None

    def _train_ridge(self, X, y, tune_hyperparams=True):
        """
        Train Ridge model

        Args:
            X: Features
            y: Target
            tune_hyperparams: Whether to tune hyperparameters

        Returns:
            Trained model
        """
        try:
            if tune_hyperparams:
                # Define parameter grid
                param_grid = {
                    "alpha": [0.1, 1.0, 10.0, 100.0],
                    "fit_intercept": [True, False],
                    "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                }

                # Use TimeSeriesSplit for time series data
                tscv = TimeSeriesSplit(n_splits=3)

                # Init model
                model = Ridge(random_state=42)

                # Perform grid search
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring="neg_mean_squared_error",
                    cv=tscv,
                    verbose=1,
                    n_jobs=-1,
                )

                grid_search.fit(X, y)

                # Get best model
                best_model = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")

                return best_model
            else:
                # Use default parameters
                model = Ridge(alpha=1.0, random_state=42)

                model.fit(X, y)

                return model
        except Exception as e:
            logger.error(f"Error training Ridge model: {e}")
            return None

    def _train_lasso(self, X, y, tune_hyperparams=True):
        """
        Train Lasso model

        Args:
            X: Features
            y: Target
            tune_hyperparams: Whether to tune hyperparameters

        Returns:
            Trained model
        """
        try:
            if tune_hyperparams:
                # Define parameter grid
                param_grid = {
                    "alpha": [0.1, 1.0, 10.0, 100.0],
                    "fit_intercept": [True, False],
                    "selection": ["cyclic", "random"],
                }

                # Use TimeSeriesSplit for time series data
                tscv = TimeSeriesSplit(n_splits=3)

                # Init model
                model = Lasso(random_state=42)

                # Perform grid search
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring="neg_mean_squared_error",
                    cv=tscv,
                    verbose=1,
                    n_jobs=-1,
                )

                grid_search.fit(X, y)

                # Get best model
                best_model = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")

                return best_model
            else:
                # Use default parameters
                model = Lasso(alpha=1.0, random_state=42)

                model.fit(X, y)

                return model
        except Exception as e:
            logger.error(f"Error training Lasso model: {e}")
            return None

    def _train_svr(self, X, y, tune_hyperparams=True):
        """
        Train SVR model

        Args:
            X: Features
            y: Target
            tune_hyperparams: Whether to tune hyperparameters

        Returns:
            Trained model
        """
        try:
            if tune_hyperparams:
                # Define parameter grid
                param_grid = {
                    "C": [0.1, 1.0, 10.0, 100.0],
                    "kernel": ["linear", "rbf", "poly"],
                    "epsilon": [0.1, 0.2, 0.5],
                    "gamma": ["scale", "auto"],
                }

                # Use TimeSeriesSplit for time series data
                tscv = TimeSeriesSplit(n_splits=3)

                # Init model
                model = SVR()

                # Perform grid search
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring="neg_mean_squared_error",
                    cv=tscv,
                    verbose=1,
                    n_jobs=-1,
                )

                grid_search.fit(X, y)

                # Get best model
                best_model = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")

                return best_model
            else:
                # Use default parameters
                model = SVR(C=1.0, kernel="rbf", epsilon=0.1)

                model.fit(X, y)

                return model
        except Exception as e:
            logger.error(f"Error training SVR model: {e}")
            return None

    def evaluate_model(self, model, features_df, target_column="ProjectedPosition"):
        """
        Evaluate model performance

        Args:
            model: Trained model
            features_df: DataFrame with features and target
            target_column: Target column name

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Prepare data
            X, y, _ = self._prepare_data(features_df, target_column)

            if X is None or y is None:
                logger.error("Failed to prepare data for evaluation")
                return None

            # Make predictions
            y_pred = model.predict(X)

            # Calculate metrics
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, y_pred)

            # CV score
            cv_scores = cross_val_score(
                model, X, y, cv=TimeSeriesSplit(n_splits=3), scoring="neg_mean_squared_error"
            )
            cv_rmse = np.sqrt(-cv_scores.mean())

            # Log results
            logger.info(f"Model evaluation results for {target_column}:")
            logger.info(f"  MSE: {mse:.4f}")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  R²: {r2:.4f}")
            logger.info(f"  CV RMSE: {cv_rmse:.4f}")

            metrics = {
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "cv_rmse": cv_rmse,
                "predictions": y_pred,
            }

            return metrics
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return None

    def plot_feature_importance(self, output_file=None):
        """
        Plot feature importance

        Args:
            output_file: File path to save the plot

        Returns:
            matplotlib figure
        """
        try:
            if not self.feature_importance:
                logger.warning("No feature importance available")
                return None

            # Convert to DataFrame for plotting
            importance_df = pd.DataFrame(
                {
                    "Feature": list(self.feature_importance.keys()),
                    "Importance": list(self.feature_importance.values()),
                }
            )

            # Sort by importance
            importance_df = importance_df.sort_values("Importance", ascending=False)

            # Plot
            plt.figure(figsize=(12, 8))
            sns.barplot(x="Importance", y="Feature", data=importance_df.head(20))
            plt.title("Top 20 Feature Importance", fontsize=16)
            plt.tight_layout()

            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
                logger.info(f"Feature importance plot saved to {output_file}")

            return plt.gcf()
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
            return None

    def predict_with_model(self, model, features_df):
        """
        Make predictions with trained model

        Args:
            model: Trained model
            features_df: DataFrame with features

        Returns:
            Predictions array
        """
        try:
            # Filter only feature columns (exclude target and metadata columns)
            non_feature_cols = set(
                [
                    "ProjectedPosition",
                    "DriverId",
                    "FullName",
                    "TeamName",
                    "Status",
                    "CompletedLaps",
                    "TotalRaceTime",
                    "FormattedGap",
                    "FormattedRaceTime",
                    "Points",
                    "FastestLap",
                    "GapToWinner",
                    "FormattedPositionChange",
                    "Position",
                    "BestTime",
                    "GapToPole",  # Match training exclusions
                ]
            )

            # Convert numeric columns to float first
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            logger.debug(f"Converting numeric columns to float: {list(numeric_cols)}")

            # Create a copy of the DataFrame to avoid modifying the original
            features_df = features_df.copy()

            for col in numeric_cols:
                try:
                    features_df[col] = pd.to_numeric(features_df[col], errors="coerce").astype(
                        float
                    )
                except Exception as e:
                    logger.error(f"Error converting column {col} to float: {e}")
                    logger.debug(f"Sample values from {col}: {features_df[col].head()}")

            # Select feature columns (only numeric columns that are not in non_feature_cols)
            feature_cols = [col for col in numeric_cols if col not in non_feature_cols]

            # Log the features being used
            logger.info(f"Using {len(feature_cols)} features for prediction")
            logger.debug(f"Selected features: {', '.join(str(col) for col in feature_cols)}")

            # Handle missing values
            X = features_df[feature_cols].copy()
            X = X.fillna(X.mean())

            # Debug: Print feature matrix info
            logger.debug(f"Feature matrix shape: {X.shape}")
            logger.debug("Feature matrix data types:")
            for col in X.columns:
                logger.debug(f"{col}: {X[col].dtype}")

            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Make predictions
            predictions = model.predict(X_scaled)
            predictions = predictions.astype(float)

            logger.debug(f"Predictions shape: {predictions.shape}")
            logger.debug(f"Predictions dtype: {predictions.dtype}")

            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None

    def load_model(self, model_path):
        """
        Load a trained model from file

        Args:
            model_path: Path to model file

        Returns:
            Loaded model
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            logger.info(f"Model loaded from {model_path}")

            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    def list_available_models(self):
        """
        List all available trained models

        Returns:
            List of model file paths
        """
        try:
            if not os.path.exists(self.models_dir):
                logger.warning(f"Models directory not found: {self.models_dir}")
                return []

            model_files = [
                os.path.join(self.models_dir, f)
                for f in os.listdir(self.models_dir)
                if f.endswith(".pkl")
            ]

            if not model_files:
                logger.info("No trained models found")
            else:
                logger.info(f"Found {len(model_files)} trained models")
                for model_file in model_files:
                    logger.info(f"  {os.path.basename(model_file)}")

            return model_files
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

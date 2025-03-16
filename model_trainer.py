import pandas as pd
import numpy as np
import logging
import pickle
import os
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

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
        
        # Create models directory if it doesn't exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            logger.info(f"Created models directory: {models_dir}")
            
    def _prepare_data(self, features_df, target_column="FinishPosition"):
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
                
            # Select features (exclude any non-feature columns)
            non_feature_cols = [
                target_column, "DriverId", "FullName", "TeamName", 
                "Status", "CompletedLaps", "TotalRaceTime", 
                "FormattedGap", "FormattedRaceTime", "Points",
                "FastestLap", "GapToWinner", "FormattedPositionChange"
            ]
            
            feature_cols = [col for col in features_df.columns 
                           if col not in non_feature_cols]
            
            # Log the features being used
            logger.info(f"Using {len(feature_cols)} features for model training")
            
            # Handle missing values
            X = features_df[feature_cols].copy()
            X = X.fillna(X.mean())
            
            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Prepare target
            y = features_df[target_column].copy()
            
            return X_scaled, y, feature_cols
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None, None, None
    
    def train_position_model(self, features_df, model_type="xgboost", 
                          target_column="FinishPosition",
                          tune_hyperparams=True):
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
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                self.feature_importance = {feature_names[i]: importance[i] 
                                           for i in range(len(feature_names))}
                
                # Log top 10 features
                top_features = sorted(self.feature_importance.items(), 
                                     key=lambda x: x[1], reverse=True)[:10]
                logger.info("Top 10 features by importance:")
                for feature, importance in top_features:
                    logger.info(f"  {feature}: {importance:.4f}")
                    
            # Save model
            model_path = os.path.join(
                self.models_dir, 
                f"{model_type}_{target_column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            )
            with open(model_path, 'wb') as f:
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
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_child_weight': [1, 3, 5],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
                
                # Use TimeSeriesSplit for time series data
                tscv = TimeSeriesSplit(n_splits=3)
                
                # Init model
                model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
                
                # Perform grid search
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring='neg_mean_squared_error',
                    cv=tscv,
                    verbose=1,
                    n_jobs=-1
                )
                
                grid_search.fit(X, y)
                
                # Get best model
                best_model = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")
                
                return best_model
            else:
                # Use default parameters
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42
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
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                
                # Use TimeSeriesSplit for time series data
                tscv = TimeSeriesSplit(n_splits=3)
                
                # Init model
                model = GradientBoostingRegressor(random_state=42)
                
                # Perform grid search
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring='neg_mean_squared_error',
                    cv=tscv,
                    verbose=1,
                    n_jobs=-1
                )
                
                grid_search.fit(X, y)
                
                # Get best model
                best_model = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")
                
                return best_model
            else:
                # Use default parameters
                model = GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42
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
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                
                # Use TimeSeriesSplit for time series data
                tscv = TimeSeriesSplit(n_splits=3)
                
                # Init model
                model = RandomForestRegressor(random_state=42)
                
                # Perform grid search
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring='neg_mean_squared_error',
                    cv=tscv,
                    verbose=1,
                    n_jobs=-1
                )
                
                grid_search.fit(X, y)
                
                # Get best model
                best_model = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")
                
                return best_model
            else:
                # Use default parameters
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=None,
                    random_state=42
                )
                
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
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'fit_intercept': [True, False],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
                }
                
                # Use TimeSeriesSplit for time series data
                tscv = TimeSeriesSplit(n_splits=3)
                
                # Init model
                model = Ridge(random_state=42)
                
                # Perform grid search
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring='neg_mean_squared_error',
                    cv=tscv,
                    verbose=1,
                    n_jobs=-1
                )
                
                grid_search.fit(X, y)
                
                # Get best model
                best_model = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")
                
                return best_model
            else:
                # Use default parameters
                model = Ridge(
                    alpha=1.0,
                    random_state=42
                )
                
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
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'fit_intercept': [True, False],
                    'selection': ['cyclic', 'random']
                }
                
                # Use TimeSeriesSplit for time series data
                tscv = TimeSeriesSplit(n_splits=3)
                
                # Init model
                model = Lasso(random_state=42)
                
                # Perform grid search
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring='neg_mean_squared_error',
                    cv=tscv,
                    verbose=1,
                    n_jobs=-1
                )
                
                grid_search.fit(X, y)
                
                # Get best model
                best_model = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")
                
                return best_model
            else:
                # Use default parameters
                model = Lasso(
                    alpha=1.0,
                    random_state=42
                )
                
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
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'epsilon': [0.1, 0.2, 0.5],
                    'gamma': ['scale', 'auto']
                }
                
                # Use TimeSeriesSplit for time series data
                tscv = TimeSeriesSplit(n_splits=3)
                
                # Init model
                model = SVR()
                
                # Perform grid search
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring='neg_mean_squared_error',
                    cv=tscv,
                    verbose=1,
                    n_jobs=-1
                )
                
                grid_search.fit(X, y)
                
                # Get best model
                best_model = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")
                
                return best_model
            else:
                # Use default parameters
                model = SVR(
                    C=1.0,
                    kernel='rbf',
                    epsilon=0.1
                )
                
                model.fit(X, y)
                
                return model
        except Exception as e:
            logger.error(f"Error training SVR model: {e}")
            return None
    
    def evaluate_model(self, model, features_df, target_column="FinishPosition"):
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
                model, X, y, 
                cv=TimeSeriesSplit(n_splits=3),
                scoring='neg_mean_squared_error'
            )
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            # Log results
            logger.info(f"Model evaluation results for {target_column}:")
            logger.info(f"  MSE: {mse:.4f}")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  R²: {r2:.4f}")
            logger.info(f"  CV RMSE: {cv_rmse:.4f}")
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'cv_rmse': cv_rmse,
                'predictions': y_pred
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
            importance_df = pd.DataFrame({
                'Feature': list(self.feature_importance.keys()),
                'Importance': list(self.feature_importance.values())
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Plot
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
            plt.title('Top 20 Feature Importance', fontsize=16)
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
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
            non_feature_cols = [
                "DriverId", "FullName", "TeamName", "GridPosition",
                "Status", "CompletedLaps", "TotalRaceTime", "FinishPosition",
                "FormattedGap", "FormattedRaceTime", "Points",
                "FastestLap", "GapToWinner", "FormattedPositionChange"
            ]
            
            feature_cols = [col for col in features_df.columns 
                           if col not in non_feature_cols]
            
            # Handle missing values
            X = features_df[feature_cols].copy()
            X = X.fillna(X.mean())
            
            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Make predictions
            predictions = model.predict(X_scaled)
            
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
                
            with open(model_path, 'rb') as f:
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
                
            model_files = [os.path.join(self.models_dir, f) 
                          for f in os.listdir(self.models_dir) 
                          if f.endswith('.pkl')]
            
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
            
    def compare_models(self, features_df, target_column="FinishPosition", 
                     model_types=None, tune_hyperparams=False):
        """
        Compare multiple model types
        
        Args:
            features_df: DataFrame with features and target
            target_column: Target column name
            model_types: List of model types to compare
            tune_hyperparams: Whether to tune hyperparameters
            
        Returns:
            Dictionary with model comparison results
        """
        try:
            if model_types is None:
                model_types = ["xgboost", "gradient_boosting", "random_forest", 
                             "ridge", "lasso", "svr"]
                
            results = {}
            
            for model_type in model_types:
                logger.info(f"Training and evaluating {model_type} model")
                
                # Train model
                model = self.train_position_model(
                    features_df, 
                    model_type=model_type,
                    target_column=target_column,
                    tune_hyperparams=tune_hyperparams
                )
                
                if model is None:
                    logger.error(f"Failed to train {model_type} model")
                    continue
                
                # Evaluate model
                metrics = self.evaluate_model(
                    model, 
                    features_df, 
                    target_column=target_column
                )
                
                if metrics is None:
                    logger.error(f"Failed to evaluate {model_type} model")
                    continue
                    
                results[model_type] = metrics
            
            # Compare results
            if results:
                comparison_df = pd.DataFrame({
                    model_type: {
                        'MSE': metrics['mse'],
                        'RMSE': metrics['rmse'],
                        'R²': metrics['r2'],
                        'CV RMSE': metrics['cv_rmse']
                    }
                    for model_type, metrics in results.items()
                }).T
                
                # Log comparison
                logger.info("Model comparison results:")
                logger.info("\n" + comparison_df.to_string())
                
                # Plot comparison
                plt.figure(figsize=(12, 8))
                comparison_df[['RMSE', 'CV RMSE']].plot(kind='bar')
                plt.title('Model Comparison - RMSE', fontsize=16)
                plt.ylabel('RMSE (lower is better)')
                plt.tight_layout()
                
                # Save comparison plot
                plot_path = os.path.join(
                    self.models_dir, 
                    f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Model comparison plot saved to {plot_path}")
                
                return {
                    'metrics': comparison_df,
                    'models': {model_type: model for model_type, model in zip(results.keys(), results.values())},
                    'best_model': min(results.items(), key=lambda x: x[1]['rmse'])[0]
                }
            else:
                logger.error("No models were successfully trained and evaluated")
                return None
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return None 
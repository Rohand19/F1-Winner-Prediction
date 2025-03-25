#!/usr/bin/env python3
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import random
import json

# Import custom modules
from src.data.data_processor import F1DataProcessor
from src.features.feature_engineering import F1FeatureEngineer
from src.f1predictor.models.race_predictor import RacePredictor, print_race_results
from src.f1predictor.models.model_trainer import F1ModelTrainer

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"f1_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("F1Predictor.Main")


def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="F1 Race Prediction System")

    # Race information
    parser.add_argument("--year", type=int, required=True, help="Year of the race to predict")
    parser.add_argument("--race", type=int, required=True, help="Race number to predict")
    parser.add_argument("--event", type=str, help="Event name (optional)")

    # Data options
    parser.add_argument(
        "--historical-races",
        type=int,
        default=5,
        help="Number of historical races to use for training data",
    )
    parser.add_argument(
        "--include-practice",
        action="store_true",
        help="Include practice session data for predictions",
    )
    parser.add_argument(
        "--reload-data", action="store_true", help="Force reload of all data (ignore cache)"
    )

    # Training options
    parser.add_argument(
        "--model-type",
        type=str,
        default="gradient_boosting",
        choices=["gradient_boosting", "random_forest", "neural_network"],
        help="Model type to use for predictions",
    )
    parser.add_argument(
        "--tune-hyperparams", action="store_true", help="Tune model hyperparameters (slower)"
    )
    parser.add_argument(
        "--compare-models", action="store_true", help="Compare multiple model types"
    )
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results and visualizations",
    )

    # Weather and race conditions
    parser.add_argument(
        "--rain-chance", type=float, default=0.0, help="Chance of rain during race (0.0-1.0)"
    )
    parser.add_argument(
        "--changing-conditions",
        action="store_true",
        help="Enable changing weather conditions during race",
    )
    parser.add_argument(
        "--total-laps",
        type=int,
        default=None,
        help="Set custom number of race laps (overrides default for track)",
    )

    return parser.parse_args()


def setup_directories(args):
    """
    Create required directories
    """
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logger.info(f"Created output directory: {args.output_dir}")

    # Create cache directory if it doesn't exist
    cache_dir = os.path.join("cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        logger.info(f"Created cache directory: {cache_dir}")

    # Create models directory if it doesn't exist
    models_dir = os.path.join("models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"Created models directory: {models_dir}")


def get_current_race_info(args, data_processor):
    """
    Get information about the race to predict

    Args:
        args: Command line arguments
        data_processor: F1DataProcessor object

    Returns:
        Tuple of (year, race_round, event_name)
    """
    try:
        year = args.year

        # If race round is not specified, get the next upcoming race
        if args.race is None and args.event is None:
            upcoming_race = data_processor.get_upcoming_race()
            if upcoming_race is None:
                logger.warning("Could not determine upcoming race. Using latest completed race.")
                # Use the latest completed race
                schedule = data_processor.get_season_schedule(year)
                latest_race = schedule[schedule["EventDate"] < datetime.now()].iloc[-1]
                race_round = latest_race["RoundNumber"]
                event_name = latest_race["EventName"]
                logger.info(f"Using latest completed race: {event_name} (Round {race_round})")
            else:
                race_round = upcoming_race["RoundNumber"]
                event_name = upcoming_race["EventName"]
                logger.info(f"Predicting upcoming race: {event_name} (Round {race_round})")
        elif args.event is not None:
            # Find race by event name
            schedule = data_processor.get_season_schedule(year)
            matching_events = schedule[schedule["EventName"].str.contains(args.event, case=False)]

            if matching_events.empty:
                logger.error(f"Could not find event matching '{args.event}' in {year} season")
                sys.exit(1)

            race_round = matching_events.iloc[0]["RoundNumber"]
            event_name = matching_events.iloc[0]["EventName"]
            logger.info(f"Predicting race: {event_name} (Round {race_round})")
        else:
            # Use specified race round
            race_round = args.race
            schedule = data_processor.get_season_schedule(year)
            matching_round = schedule[schedule["RoundNumber"] == race_round]

            if matching_round.empty:
                logger.error(f"Could not find Round {race_round} in {year} season")
                sys.exit(1)

            event_name = matching_round.iloc[0]["EventName"]
            logger.info(f"Predicting race: {event_name} (Round {race_round})")

        return year, race_round, event_name
    except Exception as e:
        logger.error(f"Error getting race information: {e}")
        sys.exit(1)


def train_model(X_train, y_train, model_type="gradient_boosting", tune_hyperparams=False):
    """Train a model with optional hyperparameter tuning."""
    if model_type == "gradient_boosting":
        if tune_hyperparams:
            # Define parameter grid
            param_grid = {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
            }

            # Create base model
            base_model = GradientBoostingRegressor(random_state=42)

            # Perform grid search
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=5,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )

            # Fit grid search
            grid_search.fit(X_train, y_train)

            # Get best model
            best_params = grid_search.best_params_
            model = GradientBoostingRegressor(**best_params, random_state=42)
            model.fit(X_train, y_train)

            logger.info(f"Best parameters: {best_params}")
        else:
            # Use default parameters
            model = GradientBoostingRegressor(random_state=42)
            model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Model performance - MSE: {mse:.4f}, R2: {r2:.4f}")

    return mse, r2


def prepare_data_for_prediction(qualifying_data, historical_data=None):
    """Prepare data for prediction by adding necessary features."""
    # Create a copy to avoid modifying the original
    prediction_data = qualifying_data.copy()

    # Ensure we have the required columns
    if "Driver" not in prediction_data.columns and "FullName" in prediction_data.columns:
        prediction_data["Driver"] = prediction_data["FullName"]

    if "Team" not in prediction_data.columns and "TeamName" in prediction_data.columns:
        prediction_data["Team"] = prediction_data["TeamName"]

    # Ensure DriverId is properly formatted
    if "DriverId" not in prediction_data.columns and "Driver" in prediction_data.columns:
        prediction_data["DriverId"] = prediction_data.index

    # Ensure Grid_Position is properly formatted
    if "Grid_Position" not in prediction_data.columns and "Position" in prediction_data.columns:
        prediction_data["Grid_Position"] = pd.to_numeric(
            prediction_data["Position"], errors="coerce"
        )

    # Add historical performance features with more realistic values
    prediction_data["Last_Race_Position"] = 10  # Default middle of the field
    prediction_data["Last_Race_Points"] = 0
    prediction_data["Season_Points"] = 0
    prediction_data["Season_Position"] = 10
    prediction_data["Recent_Form"] = 0.5  # Normalized 0-1 scale
    prediction_data["Points_Form"] = 0

    # 2024 driver form factors (higher is better)
    driver_form = {
        "Max Verstappen": 0.95,  # Dominant early season form
        "Sergio Perez": 0.85,  # Strong but not at Max's level
        "Charles Leclerc": 0.90,  # Very strong early season
        "Carlos Sainz": 0.88,  # Race winner, consistent
        "Lewis Hamilton": 0.82,  # Struggling with car
        "George Russell": 0.85,  # Better adapting to 2024 car
        "Lando Norris": 0.88,  # Very competitive
        "Oscar Piastri": 0.84,  # Strong sophomore season
        "Fernando Alonso": 0.86,  # Consistent performer
        "Lance Stroll": 0.77,  # Inconsistent performances
        "Pierre Gasly": 0.78,  # Struggling with Alpine
        "Esteban Ocon": 0.76,  # Struggling with Alpine
        "Alexander Albon": 0.82,  # Performing well in Williams
        "Logan Sargeant": 0.72,  # Struggling in comparison
        "Yuki Tsunoda": 0.81,  # Strong start to 2024
        "Daniel Ricciardo": 0.78,  # Variable form
        "Nico Hulkenberg": 0.81,  # Solid performances
        "Kevin Magnussen": 0.78,  # Inconsistent
        "Valtteri Bottas": 0.76,  # Struggling with Sauber
        "Guanyu Zhou": 0.74,  # Struggling with Sauber
    }

    # Update driver form based on known data
    for idx, row in prediction_data.iterrows():
        driver = row["Driver"] if "Driver" in row else "Unknown"
        if driver in driver_form:
            prediction_data.at[idx, "Recent_Form"] = driver_form[driver]

    # Add track-specific features
    prediction_data["Track_Experience"] = 2  # Default some experience
    prediction_data["Track_Best_Position"] = 10
    prediction_data["Track_Average_Position"] = 10
    prediction_data["Track_Type"] = "permanent"  # Default track type
    prediction_data["Track_Length"] = 5.0  # Default track length in km
    prediction_data["Track_Corners"] = 15  # Default number of corners

    # Add reliability features - 2024 data
    team_reliability = {
        "Red Bull Racing": 0.92,  # Very reliable
        "Ferrari": 0.89,  # Generally reliable
        "Mercedes": 0.91,  # Very reliable
        "McLaren": 0.88,  # Good reliability
        "Aston Martin": 0.87,  # Good reliability
        "Alpine": 0.84,  # Some issues early season
        "Williams": 0.85,  # Mixed reliability
        "RB": 0.86,  # Good reliability
        "Kick Sauber": 0.82,  # Some issues
        "Haas F1 Team": 0.86,  # Improved reliability
    }

    # Add reliability rates based on team
    for idx, row in prediction_data.iterrows():
        team = row["Team"] if "Team" in row else "Unknown"
        # Use default 0.85 if team not found in the dictionary
        team_dnf_rate = 1.0 - team_reliability.get(team, 0.85)
        prediction_data.at[idx, "Driver_DNF_Rate"] = team_dnf_rate + np.random.uniform(
            -0.02, 0.02
        )  # Small per-driver variation
        prediction_data.at[idx, "Team_DNF_Rate"] = team_dnf_rate

    # Add driver consistency features
    prediction_data["Qualifying_Consistency"] = 0.5  # Default middle value
    prediction_data["Race_Consistency"] = 0.5

    # 2024 car performance data (0-1 scale, higher is better)
    team_performance_2024 = {
        "Red Bull Racing": {"straight": 0.95, "corner": 0.93, "overall": 0.95},
        "Ferrari": {"straight": 0.93, "corner": 0.92, "overall": 0.92},
        "Mercedes": {"straight": 0.89, "corner": 0.90, "overall": 0.89},
        "McLaren": {"straight": 0.92, "corner": 0.94, "overall": 0.93},
        "Aston Martin": {"straight": 0.87, "corner": 0.88, "overall": 0.87},
        "Alpine": {"straight": 0.82, "corner": 0.81, "overall": 0.81},
        "Williams": {"straight": 0.86, "corner": 0.78, "overall": 0.82},
        "RB": {"straight": 0.85, "corner": 0.84, "overall": 0.84},
        "Kick Sauber": {"straight": 0.78, "corner": 0.76, "overall": 0.77},
        "Haas F1 Team": {"straight": 0.82, "corner": 0.80, "overall": 0.81},
        # Aliases
        "Red Bull": {"straight": 0.95, "corner": 0.93, "overall": 0.95},
        "Haas": {"straight": 0.82, "corner": 0.80, "overall": 0.81},
        "Sauber": {"straight": 0.78, "corner": 0.76, "overall": 0.77},
    }

    # Add car performance metrics based on team
    for idx, row in prediction_data.iterrows():
        team = row["Team"] if "Team" in row else "Unknown"
        if team in team_performance_2024:
            prediction_data.at[idx, "Team_Straight_Speed"] = team_performance_2024[team]["straight"]
            prediction_data.at[idx, "Team_Corner_Speed"] = team_performance_2024[team]["corner"]
            prediction_data.at[idx, "Car_Performance"] = team_performance_2024[team]["overall"]
        else:
            # Default values if team not found
            prediction_data.at[idx, "Team_Straight_Speed"] = 0.8
            prediction_data.at[idx, "Team_Corner_Speed"] = 0.8
            prediction_data.at[idx, "Car_Performance"] = 0.8

    # Add weather data with default values (for now)
    prediction_data["Track_Temperature"] = 35.0  # Bahrain is typically hot
    prediction_data["Air_Temperature"] = 25.0
    prediction_data["Humidity"] = 60.0

    # Grid Position Factor - Importance of starting position varies by track
    # Initialize a grid position importance factor - higher means starting position matters more
    prediction_data["Grid_Importance"] = 0.85  # High importance for Bahrain

    # Calculate Grid Advantage based on position with diminishing returns
    # Front row has biggest advantage, then diminishing returns
    for idx, row in prediction_data.iterrows():
        grid_pos = row["Grid_Position"] if "Grid_Position" in row else 10
        # Exponential decay of advantage
        grid_advantage = min(1.0, np.exp(-0.15 * (grid_pos - 1)))
        prediction_data.at[idx, "Grid_Advantage"] = grid_advantage

    # Add qualifying time features if available
    if "Q1" in prediction_data.columns:
        # Convert timedelta to seconds if needed
        if pd.api.types.is_timedelta64_dtype(prediction_data["Q1"]):
            prediction_data["Q1_Time"] = prediction_data["Q1"].dt.total_seconds()
        else:
            # Try to convert string to timedelta
            try:
                prediction_data["Q1_Time"] = pd.to_timedelta(
                    prediction_data["Q1"]
                ).dt.total_seconds()
            except:
                prediction_data["Q1_Time"] = 90.0  # Default Q1 time
    else:
        prediction_data["Q1_Time"] = 90.0  # Default Q1 time

    if "Q2" in prediction_data.columns:
        if pd.api.types.is_timedelta64_dtype(prediction_data["Q2"]):
            prediction_data["Q2_Time"] = prediction_data["Q2"].dt.total_seconds()
        else:
            try:
                prediction_data["Q2_Time"] = pd.to_timedelta(
                    prediction_data["Q2"]
                ).dt.total_seconds()
            except:
                prediction_data["Q2_Time"] = 89.0  # Default Q2 time
    else:
        prediction_data["Q2_Time"] = 89.0  # Default Q2 time

    if "Q3" in prediction_data.columns:
        if pd.api.types.is_timedelta64_dtype(prediction_data["Q3"]):
            prediction_data["Q3_Time"] = prediction_data["Q3"].dt.total_seconds()
        else:
            try:
                prediction_data["Q3_Time"] = pd.to_timedelta(
                    prediction_data["Q3"]
                ).dt.total_seconds()
            except:
                prediction_data["Q3_Time"] = 88.0  # Default Q3 time
    else:
        prediction_data["Q3_Time"] = 88.0  # Default Q3 time

    # Add special feature for tracks where overtaking is difficult
    prediction_data["Overtaking_Difficulty"] = 0.6  # Medium difficulty in Bahrain

    # Calculate qualifying improvements
    prediction_data["Q2_Improvement"] = prediction_data["Q1_Time"] - prediction_data["Q2_Time"]
    prediction_data["Q3_Improvement"] = prediction_data["Q2_Time"] - prediction_data["Q3_Time"]
    prediction_data["Q3_Gap"] = prediction_data["Q3_Time"] - prediction_data["Q3_Time"].min()

    # Convert categorical variables to numeric using one-hot encoding
    categorical_columns = ["Track_Type"]
    prediction_data = pd.get_dummies(
        prediction_data, columns=categorical_columns, prefix=categorical_columns
    )

    # Fill NaN values
    numeric_columns = prediction_data.select_dtypes(include=["number"]).columns
    for col in numeric_columns:
        prediction_data[col] = prediction_data[col].fillna(prediction_data[col].mean())

    prediction_data = prediction_data.fillna(0)  # Fill any remaining NaNs with 0

    # Ensure all columns are numeric
    for col in prediction_data.columns:
        if not pd.api.types.is_numeric_dtype(prediction_data[col]):
            try:
                prediction_data[col] = pd.to_numeric(prediction_data[col], errors="coerce")
            except:
                # If conversion fails, drop the column
                prediction_data = prediction_data.drop(columns=[col])

    # Add track-specific features based on calendar
    track_features = {
        "Bahrain": {
            "track_type": "desert",
            "overtaking_difficulty": 0.6,
            "tire_deg": 0.85,
            "default_strategy": "2-stop",
            "track_temp": 35.0,
            "total_laps": 57,
        },
        "Saudi Arabia": {
            "track_type": "street",
            "overtaking_difficulty": 0.8,
            "tire_deg": 0.6,
            "default_strategy": "1-stop",
            "track_temp": 30.0,
            "total_laps": 50,
        },
        "Australia": {
            "track_type": "street/permanent",
            "overtaking_difficulty": 0.7,
            "tire_deg": 0.7,
            "default_strategy": "2-stop",
            "track_temp": 25.0,
            "total_laps": 58,
        },
        "Japan": {
            "track_type": "permanent",
            "overtaking_difficulty": 0.65,
            "tire_deg": 0.75,
            "default_strategy": "2-stop",
            "track_temp": 20.0,
            "total_laps": 53,
        },
        "China": {
            "track_type": "permanent",
            "overtaking_difficulty": 0.55,
            "tire_deg": 0.8,
            "default_strategy": "2-stop",
            "track_temp": 18.0,
            "total_laps": 56,
        },
        "Miami": {
            "track_type": "street",
            "overtaking_difficulty": 0.6,
            "tire_deg": 0.7,
            "default_strategy": "1-stop",
            "track_temp": 40.0,
            "total_laps": 57,
        },
        "Imola": {
            "track_type": "permanent",
            "overtaking_difficulty": 0.85,
            "tire_deg": 0.65,
            "default_strategy": "1-stop",
            "track_temp": 22.0,
            "total_laps": 63,
        },
    }

    return prediction_data


def get_track_details(track_name):
    """Get track-specific details for race simulation.

    Args:
        track_name: Name of the track

    Returns:
        dict: Track details with total laps, track type, etc.
    """
    track_details = {
        "Bahrain": {
            "total_laps": 57,
            "track_type": "desert",
            "track_temp": 35.0,
            "air_temp": 25.0,
            "humidity": 60.0,
        },
        "Saudi Arabia": {
            "total_laps": 50,
            "track_type": "street",
            "track_temp": 30.0,
            "air_temp": 25.0,
            "humidity": 45.0,
        },
        "Australia": {
            "total_laps": 58,
            "track_type": "street/permanent",
            "track_temp": 25.0,
            "air_temp": 20.0,
            "humidity": 70.0,
        },
        "Japan": {
            "total_laps": 53,
            "track_type": "permanent",
            "track_temp": 20.0,
            "air_temp": 18.0,
            "humidity": 65.0,
        },
        "China": {
            "total_laps": 56,
            "track_type": "permanent",
            "track_temp": 18.0,
            "air_temp": 16.0,
            "humidity": 75.0,
        },
        "Miami": {
            "total_laps": 57,
            "track_type": "street",
            "track_temp": 40.0,
            "air_temp": 30.0,
            "humidity": 80.0,
        },
        "Imola": {
            "total_laps": 63,
            "track_type": "permanent",
            "track_temp": 22.0,
            "air_temp": 20.0,
            "humidity": 60.0,
        },
        "Monaco": {
            "total_laps": 78,
            "track_type": "street",
            "track_temp": 30.0,
            "air_temp": 25.0,
            "humidity": 50.0,
        },
        "Spain": {
            "total_laps": 66,
            "track_type": "permanent",
            "track_temp": 35.0,
            "air_temp": 30.0,
            "humidity": 40.0,
        },
        "Austria": {
            "total_laps": 71,
            "track_type": "permanent",
            "track_temp": 25.0,
            "air_temp": 22.0,
            "humidity": 50.0,
        },
        "United Kingdom": {
            "total_laps": 52,
            "track_type": "permanent",
            "track_temp": 20.0,
            "air_temp": 18.0,
            "humidity": 80.0,
        },
        "Hungary": {
            "total_laps": 70,
            "track_type": "permanent",
            "track_temp": 35.0,
            "air_temp": 30.0,
            "humidity": 45.0,
        },
        "Belgium": {
            "total_laps": 44,
            "track_type": "permanent",
            "track_temp": 22.0,
            "air_temp": 20.0,
            "humidity": 70.0,
        },
        "Netherlands": {
            "total_laps": 72,
            "track_type": "permanent",
            "track_temp": 20.0,
            "air_temp": 18.0,
            "humidity": 75.0,
        },
        "Italy": {
            "total_laps": 53,
            "track_type": "permanent",
            "track_temp": 30.0,
            "air_temp": 25.0,
            "humidity": 55.0,
        },
        "Azerbaijan": {
            "total_laps": 51,
            "track_type": "street",
            "track_temp": 30.0,
            "air_temp": 25.0,
            "humidity": 50.0,
        },
        "Singapore": {
            "total_laps": 62,
            "track_type": "street",
            "track_temp": 35.0,
            "air_temp": 30.0,
            "humidity": 85.0,
        },
        "United States": {
            "total_laps": 56,
            "track_type": "permanent",
            "track_temp": 35.0,
            "air_temp": 25.0,
            "humidity": 40.0,
        },
        "Mexico": {
            "total_laps": 71,
            "track_type": "permanent",
            "track_temp": 40.0,
            "air_temp": 25.0,
            "humidity": 35.0,
        },
        "Brazil": {
            "total_laps": 71,
            "track_type": "permanent",
            "track_temp": 35.0,
            "air_temp": 28.0,
            "humidity": 70.0,
        },
        "Las Vegas": {
            "total_laps": 50,
            "track_type": "street",
            "track_temp": 15.0,
            "air_temp": 10.0,
            "humidity": 30.0,
        },
        "Qatar": {
            "total_laps": 57,
            "track_type": "permanent",
            "track_temp": 40.0,
            "air_temp": 35.0,
            "humidity": 45.0,
        },
        "Abu Dhabi": {
            "total_laps": 58,
            "track_type": "permanent",
            "track_temp": 35.0,
            "air_temp": 30.0,
            "humidity": 50.0,
        },
    }

    # Default values for unknown tracks
    default_details = {
        "total_laps": 55,
        "track_type": "permanent",
        "track_temp": 30.0,
        "air_temp": 25.0,
        "humidity": 60.0,
    }

    return track_details.get(track_name, default_details)


def get_track_name_from_event(event_name):
    """Extract track name from event name."""
    track_mapping = {
        "bahrain": "Bahrain",
        "saudi": "Saudi Arabia",
        "jeddah": "Saudi Arabia",
        "australia": "Australia",
        "melbourne": "Australia",
        "japan": "Japan",
        "suzuka": "Japan",
        "china": "China",
        "shanghai": "China",
        "miami": "Miami",
        "imola": "Imola",
        "emilia": "Imola",
        "monaco": "Monaco",
        "monte carlo": "Monaco",
        "spain": "Spain",
        "barcelona": "Spain",
        "catalunya": "Spain",
        "austria": "Austria",
        "styria": "Austria",
        "britain": "United Kingdom",
        "silverstone": "United Kingdom",
        "hungary": "Hungary",
        "budapest": "Hungary",
        "belgium": "Belgium",
        "spa": "Belgium",
        "netherlands": "Netherlands",
        "zandvoort": "Netherlands",
        "italy": "Italy",
        "monza": "Italy",
        "azerbaijan": "Azerbaijan",
        "baku": "Azerbaijan",
        "singapore": "Singapore",
        "marina bay": "Singapore",
        "united states": "United States",
        "austin": "United States",
        "mexico": "Mexico",
        "mexico city": "Mexico",
        "brazil": "Brazil",
        "sao paulo": "Brazil",
        "interlagos": "Brazil",
        "las vegas": "Las Vegas",
        "qatar": "Qatar",
        "losail": "Qatar",
        "abu dhabi": "Abu Dhabi",
        "yas marina": "Abu Dhabi",
    }

    # Convert to lowercase for matching
    event_lower = event_name.lower()

    # Try to find a match
    for key, value in track_mapping.items():
        if key in event_lower:
            return value

    # If no match found, return the original event name
    return event_name


def calculate_prediction_metrics(predicted_results, actual_results):
    """Calculate various metrics to evaluate prediction accuracy."""
    metrics = {}
    
    # Calculate position-based metrics
    position_diff = abs(predicted_results['Position'] - actual_results['Position'])
    metrics['MAE'] = position_diff.mean()
    metrics['MSE'] = (position_diff ** 2).mean()
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    
    # Calculate exact position matches
    exact_matches = (predicted_results['Position'] == actual_results['Position']).mean()
    metrics['ExactPositionAccuracy'] = exact_matches
    
    # Calculate top-N accuracy
    metrics['Top3Accuracy'] = ((position_diff <= 2).sum() / len(position_diff))
    metrics['Top5Accuracy'] = ((position_diff <= 4).sum() / len(position_diff))
    metrics['Top10Accuracy'] = ((position_diff <= 9).sum() / len(position_diff))
    
    return metrics


def main():
    """
    Main prediction pipeline
    """
    # Parse command line arguments
    args = parse_arguments()

    # Setup required directories
    setup_directories(args)

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("Starting F1 Race Prediction System")
    logger.info(f"Arguments: {args}")

    try:
        # Initialize data processor
        data_processor = F1DataProcessor(args.year)

        # Get race information
        year, race_round, event_name = get_current_race_info(args, data_processor)

        # Determine track name from event name
        track_name = get_track_name_from_event(event_name)
        logger.info(f"Determined track name: {track_name}")

        # Get track-specific details
        track_details = get_track_details(track_name)
        logger.info(f"Track details: {json.dumps(track_details, indent=2)}")

        # Set total laps from track details or from args if provided
        total_laps = args.total_laps if args.total_laps is not None else track_details["total_laps"]

        # Create unique output directory for this run
        run_output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
        os.makedirs(run_output_dir, exist_ok=True)

        # Load qualifying data for current race
        qualifying_data = data_processor.load_qualifying_data(year, race_round)

        if qualifying_data is None or qualifying_data.empty:
            logger.error(f"No qualifying data available for {event_name} {year}")
            logger.info("Checking if qualifying session has been completed...")

            # Check if qualifying has been completed
            event_schedule = data_processor.get_event_schedule(year, race_round)
            if event_schedule is not None:
                qualifying_session = event_schedule[
                    event_schedule["Session"].str.contains("Qualifying", case=False)
                ]

                if not qualifying_session.empty:
                    qual_time = qualifying_session.iloc[0]["SessionStart"]
                    if qual_time > datetime.now():
                        logger.info(
                            f"Qualifying session has not started yet. Scheduled for: {qual_time}"
                        )
                    else:
                        logger.info(f"Qualifying session should have completed at: {qual_time}")
                        logger.info(
                            "Data may not be available yet or there was an error loading it."
                        )

            # Exit if no qualifying data (can't make predictions without it)
            logger.error("Cannot make predictions without qualifying data")
            sys.exit(1)

        logger.info(f"Loaded qualifying data for {len(qualifying_data)} drivers")

        # Process qualifying data to extract driver and team information
        logger.info("Processing qualifying data...")

        # Create a simplified qualifying data for race prediction
        simplified_qualifying = pd.DataFrame()

        # Add Driver field based on available data
        if "BroadcastName" in qualifying_data.columns:
            simplified_qualifying["Driver"] = qualifying_data["BroadcastName"]
        elif "Abbreviation" in qualifying_data.columns:
            simplified_qualifying["Driver"] = qualifying_data["Abbreviation"]
        elif "FullName" in qualifying_data.columns:
            simplified_qualifying["Driver"] = qualifying_data["FullName"]
        else:
            simplified_qualifying["Driver"] = [
                "Driver " + str(i + 1) for i in range(len(qualifying_data))
            ]

        # Add Team field based on available data
        if "TeamName" in qualifying_data.columns:
            simplified_qualifying["Team"] = qualifying_data["TeamName"]
        else:
            simplified_qualifying["Team"] = [
                "Team " + str(i + 1) for i in range(len(qualifying_data))
            ]

        # Add DriverId if available
        if "DriverId" in qualifying_data.columns:
            simplified_qualifying["DriverId"] = qualifying_data["DriverId"]
        elif "Abbreviation" in qualifying_data.columns:
            simplified_qualifying["DriverId"] = qualifying_data["Abbreviation"]
        else:
            simplified_qualifying["DriverId"] = simplified_qualifying.index

        # Add Grid Position
        if "Position" in qualifying_data.columns:
            simplified_qualifying["Grid_Position"] = pd.to_numeric(
                qualifying_data["Position"], errors="coerce"
            )
        else:
            simplified_qualifying["Grid_Position"] = range(1, len(qualifying_data) + 1)

        # Add timing information
        if "Q1" in qualifying_data.columns:
            simplified_qualifying["Q1_Time"] = qualifying_data["Q1"]
        if "Q2" in qualifying_data.columns:
            simplified_qualifying["Q2_Time"] = qualifying_data["Q2"]
        if "Q3" in qualifying_data.columns:
            simplified_qualifying["Q3_Time"] = qualifying_data["Q3"]

        # Log drivers and teams for debugging
        logger.info("Drivers in simplified qualifying data:")
        for idx, row in simplified_qualifying.iterrows():
            driver = row.get("Driver", "Unknown")
            team = row.get("Team", "Unknown")
            pos = row.get("Grid_Position", "Unknown")
            logger.info(f"  {pos}: {driver} ({team})")

        # Collect historical race data for feature engineering
        historical_data = data_processor.collect_historical_race_data(
            current_year=year,
            current_round=race_round,
            num_races=args.historical_races,
            include_practice=args.include_practice,
        )

        if historical_data is None or len(historical_data) == 0:
            logger.warning(
                "No historical race data available, proceeding with qualifying data only"
            )
        else:
            logger.info(f"Collected historical data from {len(historical_data)} races")

        # Get track information
        track_info = data_processor.get_track_info(year, race_round)

        if track_info is None:
            logger.warning("Could not retrieve track information, using defaults")
        else:
            logger.info(
                f"Track: {track_info.get('Name', 'Unknown')} ({track_info.get('Length', 'Unknown')}km)"
            )

        # Prepare data for prediction
        prediction_data = prepare_data_for_prediction(simplified_qualifying, historical_data)

        # Initialize feature engineer
        feature_engineer = F1FeatureEngineer()

        # Generate features for prediction
        features = feature_engineer.engineer_features(
            qualifying_results=prediction_data,
            historical_data=historical_data,
            track_info=track_info,
        )

        if features is None or features.empty:
            logger.error("Failed to engineer features for prediction")
            sys.exit(1)

        logger.info(f"Generated features for {len(features)} drivers")

        # Make sure features contain only numeric data
        numeric_features = features.select_dtypes(include=["number"])
        if numeric_features.shape[1] < features.shape[1]:
            logger.warning(
                f"Dropping {features.shape[1] - numeric_features.shape[1]} non-numeric columns"
            )
            features = numeric_features

        # Check for NaN values
        if features.isna().any().any():
            logger.warning("Found NaN values in features, filling with column means")
            features = features.fillna(features.mean())

        # Double check all NaN values are removed
        if features.isna().any().any():
            logger.warning("Still found NaN values after filling with means, filling with zeros")
            features = features.fillna(0)

        # Split data for training
        X = features
        y = prediction_data["Grid_Position"]  # Use grid position as a proxy for race position

        # Check for NaN values in target
        if y.isna().any():
            logger.warning("Found NaN values in target variable, filling with median")
            y = y.fillna(y.median())

        # Debug info
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target variable length: {len(y)}")

        # Train model
        logger.info(f"Training {args.model_type} model...")
        model = train_model(
            X, y, model_type=args.model_type, tune_hyperparams=args.tune_hyperparams
        )

        # Evaluate model
        evaluate_model(model, X, y)

        # Initialize weather conditions
        weather_conditions = {
            "track_temp": track_details["track_temp"],
            "air_temp": track_details["air_temp"],
            "humidity": track_details["humidity"],
            "rain_chance": args.rain_chance,
            "wind_speed": 5.0,  # Default
            "changing_conditions": args.changing_conditions,
        }

        # Create race predictor with track-specific parameters
        race_predictor = RacePredictor(
            model=model, feature_columns=features.columns.tolist(), total_laps=total_laps
        )

        # Make predictions with the simplified qualifying data and track-specific settings
        logger.info(f"Predicting race results for {track_name} with {total_laps} laps...")
        race_results = race_predictor.predict_finishing_positions(
            simplified_qualifying, track_name=track_name, weather_conditions=weather_conditions
        )

        # Print results
        print_race_results(race_results)

        # Save results to CSV
        results_file = os.path.join(run_output_dir, f"race_results_{timestamp}.csv")
        race_results.to_csv(results_file, index=False)
        logger.info(f"Saved race results to {results_file}")

        if args.visualize:
            # Create grid vs finish position plot
            plt.figure(figsize=(12, 6))
            plt.scatter(race_results["GridPosition"], race_results["Position"])
            plt.plot([1, 20], [1, 20], "r--")  # Diagonal line for reference
            plt.xlabel("Grid Position")
            plt.ylabel("Finish Position")
            plt.title(f"Grid vs Finish Position - {track_name} {year}")
            plt.grid(True)

            # Add driver labels
            for _, row in race_results.iterrows():
                plt.annotate(
                    row["Driver"] if not pd.isna(row["Driver"]) else "Unknown",
                    (row["GridPosition"], row["Position"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                )

            grid_finish_file = os.path.join(run_output_dir, f"grid_vs_finish_{timestamp}.png")
            plt.savefig(grid_finish_file)
            plt.close()
            logger.info(f"Saved grid vs. finish plot to {grid_finish_file}")

            # Create team performance visualization
            team_results = (
                race_results.groupby("Team")
                .agg({"Points": "sum", "Position": "mean", "PitStops": "mean"})
                .sort_values("Points", ascending=False)
            )

            plt.figure(figsize=(12, 6))
            sns.barplot(x=team_results.index, y="Points", data=team_results)
            plt.xticks(rotation=45, ha="right")
            plt.title(f"Team Performance - {track_name} {year}")
            plt.tight_layout()

            team_perf_file = os.path.join(run_output_dir, f"team_performance_{timestamp}.png")
            plt.savefig(team_perf_file)
            plt.close()
            logger.info(f"Saved team performance plot to {team_perf_file}")

            # Create position changes visualization
            plt.figure(figsize=(12, 8))
            position_changes = race_results.copy()
            position_changes["Change"] = (
                position_changes["GridPosition"] - position_changes["Position"]
            )
            position_changes = position_changes.sort_values("Change", ascending=False)

            # Use team colors if available
            team_colors = {
                "Red Bull Racing": "#0600EF",
                "Ferrari": "#DC0000",
                "Mercedes": "#00D2BE",
                "McLaren": "#FF8700",
                "Aston Martin": "#006F62",
                "Alpine": "#0090FF",
                "Williams": "#005AFF",
                "RB": "#2B4562",
                "Kick Sauber": "#900000",
                "Haas F1 Team": "#FFFFFF",
            }

            # Use team colors for bars
            bar_colors = [team_colors.get(team, "#808080") for team in position_changes["Team"]]

            plt.barh(position_changes["Driver"], position_changes["Change"], color=bar_colors)
            plt.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
            plt.xlabel("Position Change (Positive = Gained, Negative = Lost)")
            plt.ylabel("Driver")
            plt.title(f"Position Changes - {track_name} {year}")
            plt.grid(axis="x", linestyle="--", alpha=0.7)

            pos_changes_file = os.path.join(run_output_dir, f"position_changes_{timestamp}.png")
            plt.savefig(pos_changes_file)
            plt.close()
            logger.info(f"Saved position changes plot to {pos_changes_file}")

            # Create a pit strategy visualization if we have pit stop data
            if "PitStops" in race_results.columns:
                plt.figure(figsize=(12, 6))
                plt.bar(race_results["Driver"], race_results["PitStops"])
                plt.xticks(rotation=45, ha="right")
                plt.xlabel("Driver")
                plt.ylabel("Number of Pit Stops")
                plt.title(f"Pit Strategy - {track_name} {year}")
                plt.tight_layout()

                pit_strategy_file = os.path.join(run_output_dir, f"pit_strategy_{timestamp}.png")
                plt.savefig(pit_strategy_file)
                plt.close()
                logger.info(f"Saved pit strategy plot to {pit_strategy_file}")

        # Load actual race results
        try:
            logger.info("Loading actual race results...")
            race_session = data_processor.load_session_data(args.year, args.race, "R")
            if race_session is not None:
                actual_results = data_processor.process_actual_race_data(args.year, args.race)
                if actual_results is not None and not actual_results.empty:
                    # Create a mapping of predicted to actual results
                    predicted_positions = pd.DataFrame({
                        'Driver': race_results['Driver'],
                        'PredictedPosition': race_results['Position']
                    })
                    
                    actual_positions = pd.DataFrame({
                        'Driver': actual_results['Driver'],
                        'ActualPosition': actual_results['Position']
                    })
                    
                    # Merge predicted and actual results
                    comparison = predicted_positions.merge(
                        actual_positions,
                        on='Driver',
                        how='inner'
                    )
                    
                    if not comparison.empty:
                        # Calculate metrics
                        metrics = calculate_prediction_metrics(
                            comparison[['PredictedPosition']].rename(columns={'PredictedPosition': 'Position'}),
                            comparison[['ActualPosition']].rename(columns={'ActualPosition': 'Position'})
                        )
                        
                        # Log metrics
                        logger.info("\nPrediction Metrics:")
                        logger.info(f"Mean Absolute Error: {metrics['MAE']:.2f} positions")
                        logger.info(f"Root Mean Square Error: {metrics['RMSE']:.2f} positions")
                        logger.info(f"Exact Position Accuracy: {metrics['ExactPositionAccuracy']*100:.1f}%")
                        logger.info(f"Top 3 Accuracy: {metrics['Top3Accuracy']*100:.1f}%")
                        logger.info(f"Top 5 Accuracy: {metrics['Top5Accuracy']*100:.1f}%")
                        logger.info(f"Top 10 Accuracy: {metrics['Top10Accuracy']*100:.1f}%")
                        
                        # Save metrics to file
                        metrics_file = os.path.join(run_output_dir, f"metrics_{timestamp}.json")
                        with open(metrics_file, 'w') as f:
                            json.dump(metrics, f, indent=2)
                        logger.info(f"Saved metrics to {metrics_file}")
                        
                        # Save comparison to file
                        comparison_file = os.path.join(run_output_dir, f"comparison_{timestamp}.csv")
                        comparison.to_csv(comparison_file, index=False)
                        logger.info(f"Saved position comparison to {comparison_file}")
                    else:
                        logger.warning("No matching drivers found between predicted and actual results")
                else:
                    logger.warning("Could not process actual race results")
            else:
                logger.warning("Could not load actual race session data")
        except Exception as e:
            logger.error(f"Error loading actual race results: {e}")
        
        logger.info("F1 Race Prediction completed successfully")
        logger.info(f"Results saved to {results_file}")

    except Exception as e:
        logger.error(f"Error in prediction pipeline: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

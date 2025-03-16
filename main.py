#!/usr/bin/env python3
"""
F1 Race Winner Prediction - Main Entry Point
Run this script with command line arguments to predict race results for different Grand Prix events.
"""

import argparse
import sys
import os
from datetime import datetime
import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Create cache directory if it doesn't exist
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    print(f"Creating cache directory: {cache_dir}")
    os.makedirs(cache_dir)

# Enable fastf1 cache
fastf1.Cache.enable_cache(cache_dir)

def load_race_data(year, gp_round, session_type):
    """Load race data from FastF1 API"""
    try:
        session = fastf1.get_session(year, gp_round, session_type)
        session.load()
        print(f"Successfully loaded {year} Round {gp_round} {session_type} data")
        
        laps = session.laps[["Driver", "LapTime", "LapNumber", "Stint", "Compound"]].copy()
        laps.dropna(subset=["LapTime"], inplace=True)
        laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()
        
        return laps
    except Exception as e:
        print(f"Error loading session data: {e}")
        return None

def process_historical_data(laps_data):
    """Process lap data to get median race pace for each driver"""
    driver_race_pace = laps_data.groupby("Driver")["LapTime (s)"].median().reset_index()
    driver_race_pace.rename(columns={"LapTime (s)": "MedianLapTime (s)"}, inplace=True)
    return driver_race_pace

def load_qualifying_data(quali_data_file=None):
    """
    Load qualifying data from file or use default fictional data if file not provided
    """
    if quali_data_file:
        try:
            qualifying_data = pd.read_csv(quali_data_file)
            print(f"Loaded qualifying data from {quali_data_file}")
            return qualifying_data
        except Exception as e:
            print(f"Error loading qualifying data file: {e}")
            print("Using default qualifying data instead.")
    
    # Default fictional qualifying data
    qualifying_data = pd.DataFrame(
    {
        "Driver": 
            ["Lando Norris", "Oscar Piastri", "Max Verstappen", "George Russell", 
             "Yuki Tsunoda", "Alex Albon", "Charles Leclerc", "Lewis Hamilton", 
             "Pierre Gasly", "Carlos Sainz", "Isack Hadjar", "Fernando Alonso", 
             "Lance Stroll", "Jack Doohan", "Gabriel Bortoleto", "Kimi Antonelli", 
             "Nico Hulkenberg", "Liam Lawson", "Esteban Ocon", "Ollie Bearman"],
         
        "QualifyingTime (s)": 
            [75.096, 75.180, 75.481, 75.546, 75.670, 75.750, 75.675, 75.473, 
             75.900, 76.000, 76.100, 76.200, 76.250, 76.300, 76.350, 76.400, 
             76.450, 76.500, 76.550, 76.600]
    })
    
    # Map full driver names to driver codes
    driver_mapping = {
        "Lando Norris": "NOR", "Oscar Piastri": "PIA", "Max Verstappen": "VER", 
        "George Russell": "RUS", "Yuki Tsunoda": "TSU", "Alex Albon": "ALB", 
        "Charles Leclerc": "LEC", "Lewis Hamilton": "HAM", "Pierre Gasly": "GAS", 
        "Carlos Sainz": "SAI", "Isack Hadjar": "HAD", "Fernando Alonso": "ALO", 
        "Lance Stroll": "STR", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR", 
        "Kimi Antonelli": "ANT", "Nico Hulkenberg": "HUL", "Liam Lawson": "LAW", 
        "Esteban Ocon": "OCO", "Ollie Bearman": "BEA"
    }
    
    qualifying_data["DriverCode"] = qualifying_data["Driver"].map(driver_mapping)
    return qualifying_data

def add_fictional_data(driver_data, race_name):
    """Add fictional data for drivers without historical data"""
    fictional_times = {
        "Australian Grand Prix": {
            "HAD": 81.2, "DOO": 81.3, "BOR": 81.5, "ANT": 81.4, "LAW": 81.6, "BEA": 81.7
        },
        "Monaco Grand Prix": {
            "HAD": 74.5, "DOO": 74.6, "BOR": 74.8, "ANT": 74.7, "LAW": 74.9, "BEA": 75.0
        },
        "default": {
            "HAD": 80.0, "DOO": 80.1, "BOR": 80.3, "ANT": 80.2, "LAW": 80.4, "BEA": 80.5
        }
    }
    
    times_dict = fictional_times.get(race_name, fictional_times["default"])
    fictional_data = pd.DataFrame({
        "Driver": list(times_dict.keys()),
        "MedianLapTime (s)": list(times_dict.values())
    })
    
    return pd.concat([driver_data, fictional_data])

def predict_race_results(qualifying_data, driver_data, num_laps, race_name="Race"):
    """
    Predict race results based on qualifying data and historical race pace
    
    Args:
        qualifying_data (DataFrame): Qualifying data with driver names and times
        driver_data (DataFrame): Historical driver race pace data
        num_laps (int): Number of laps in the race
        race_name (str): Name of the race
        
    Returns:
        DataFrame: Final standings with predicted race times
    """
    # Merge qualifying data with race pace data
    model_data = qualifying_data.merge(driver_data, left_on="DriverCode", right_on="Driver", 
                                      suffixes=('', '_RacePace'))
    
    # Verify data is properly merged
    print(f"Model data shape: {model_data.shape}")
    if model_data.shape[0] == 0:
        raise ValueError("Dataset empty after preprocessing. Check merge conditions.")
    
    # Prepare data for training
    X = model_data[["QualifyingTime (s)"]].values  # 2D array for sklearn
    y = model_data["MedianLapTime (s)"].values  # Target values
    
    # Train a gradient boosting regressor
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X, y)
    
    # Predict race pace for all drivers based on qualifying times
    qualifying_X = qualifying_data[["QualifyingTime (s)"]].values  # 2D array
    predicted_lap_times = model.predict(qualifying_X)
    
    # Calculate total race time
    qualifying_data["PredictedLapTime (s)"] = predicted_lap_times
    qualifying_data["PredictedRaceTime (s)"] = qualifying_data["PredictedLapTime (s)"] * num_laps
    
    # Sort by predicted race time to get final standings
    final_standings = qualifying_data.sort_values(by="PredictedRaceTime (s)")
    
    # Calculate time deltas from winner
    winner_time = final_standings.iloc[0]["PredictedRaceTime (s)"]
    final_standings["Gap to Winner (s)"] = final_standings["PredictedRaceTime (s)"] - winner_time
    
    return final_standings

def display_results(standings, race_name, year, visualize=True):
    """Display and visualize race results"""
    print(f"\n=== Predicted {year} {race_name} Race Results ===\n")
    print(standings[["Driver", "QualifyingTime (s)", "PredictedLapTime (s)", "PredictedRaceTime (s)"]].head(20))
    
    # Format results for better display
    print(f"\n=== Final {race_name} Standings with Time Gaps ===\n")
    for i, (_, driver) in enumerate(standings.iterrows(), 1):
        gap = driver["Gap to Winner (s)"]
        gap_str = f"+{gap:.3f}s" if i > 1 else "WINNER"
        print(f"{i}. {driver['Driver']} ({driver['DriverCode']}) - {gap_str}")
    
    if visualize:
        # Visualize the results
        plt.figure(figsize=(12, 8))
        sns.barplot(x="Gap to Winner (s)", y="Driver", data=standings.iloc[:10], palette="viridis")
        plt.title(f"Predicted Time Gaps to Winner - {year} {race_name}", fontsize=16)
        plt.xlabel("Gap to Winner (seconds)", fontsize=12)
        plt.ylabel("Driver", fontsize=12)
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predicted_{race_name.lower().replace(' ', '_')}_{year}_{timestamp}.png"
        plt.savefig(filename)
        print(f"\nResults visualization saved as '{filename}'")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="F1 Race Winner Prediction")
    parser.add_argument("--historical-year", type=int, default=2024, 
                        help="Year of historical race data (default: 2024)")
    parser.add_argument("--prediction-year", type=int, default=2025,
                        help="Year to predict race results for (default: 2025)")
    parser.add_argument("--gp-round", type=int, default=3,
                        help="Grand Prix round number (default: 3 for Australian GP)")
    parser.add_argument("--race-name", type=str, default="Australian Grand Prix",
                        help="Name of the race (default: 'Australian Grand Prix')")
    parser.add_argument("--laps", type=int, default=58,
                        help="Number of laps in the race (default: 58 for Australian GP)")
    parser.add_argument("--quali-data", type=str, default=None,
                        help="Path to CSV file with qualifying data (optional)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable visualization output")
    
    args = parser.parse_args()
    
    print(f"Predicting {args.prediction_year} {args.race_name} results...")
    print(f"Using historical data from {args.historical_year} Round {args.gp_round}")
    
    # Load historical race data
    laps_data = load_race_data(args.historical_year, args.gp_round, "R")
    if laps_data is None:
        print("Failed to load race data. Exiting.")
        return 1
    
    # Process lap data
    driver_race_pace = process_historical_data(laps_data)
    print(f"Processed race pace data for {len(driver_race_pace)} drivers")
    
    # Add fictional data for rookies/new drivers
    full_driver_data = add_fictional_data(driver_race_pace, args.race_name)
    
    # Load qualifying data
    qualifying_data = load_qualifying_data(args.quali_data)
    
    # Predict race results
    final_standings = predict_race_results(
        qualifying_data, full_driver_data, args.laps, args.race_name
    )
    
    # Display and visualize results
    display_results(final_standings, args.race_name, args.prediction_year, not args.no_viz)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

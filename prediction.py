import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create cache directory if it doesn't exist
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    print(f"Creating cache directory: {cache_dir}")
    os.makedirs(cache_dir)

# Enable caching for faster data loading
fastf1.Cache.enable_cache(cache_dir)

def load_race_data(year, gp_round, session_type):
    """
    Load race data from the FastF1 API
    
    Args:
        year (int): F1 season year
        gp_round (int): Grand Prix round number
        session_type (str): Session type (e.g., "R" for race, "Q" for qualifying)
        
    Returns:
        pandas.DataFrame: Processed lap data
    """
    try:
        session = fastf1.get_session(year, gp_round, session_type)
        session.load()
        print(f"Successfully loaded {year} Round {gp_round} {session_type} data")
        
        # Extract relevant lap data
        laps = session.laps[["Driver", "LapTime", "LapNumber", "Stint", "Compound"]].copy()
        laps.dropna(subset=["LapTime"], inplace=True)
        laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()
        
        return laps
    except Exception as e:
        print(f"Error loading session data: {e}")
        return None

# Load historical Australian GP race data (round 3)
race_year = 2024
race_round = 3
race_name = "Australian Grand Prix"
laps_data = load_race_data(race_year, race_round, "R")

if laps_data is None:
    print("Failed to load race data. Exiting.")
    exit(1)

# Process lap data - use median lap time for each driver to get representative race pace
driver_race_pace = laps_data.groupby("Driver")["LapTime (s)"].median().reset_index()
driver_race_pace.rename(columns={"LapTime (s)": "MedianLapTime (s)"}, inplace=True)
print(f"Processed race pace data for {len(driver_race_pace)} drivers")

# Load qualifying data from CSV file - this is required
qualifying_times_csv = "qualifying_times.csv"
print(f"Loading qualifying data from {qualifying_times_csv}")

try:
    if not os.path.exists(qualifying_times_csv):
        raise FileNotFoundError(f"Required file {qualifying_times_csv} not found. Please run qual_data.py first to generate qualifying data.")
    
    qualifying_data = pd.read_csv(qualifying_times_csv)
    
    # Rename columns to match expected format
    qualifying_data = qualifying_data.rename(columns={
        "FullName": "Driver",
        "DriverCode": "DriverCode"
    })
    
    # Check if data was loaded successfully
    if qualifying_data.empty:
        raise ValueError("Qualifying data is empty")
        
    print(f"Successfully loaded qualifying data for {len(qualifying_data)} drivers")
    print(qualifying_data.head())
    
except Exception as e:
    print(f"Error loading qualifying data: {e}")
    print("Please run qual_data.py first to generate the qualifying_times.csv file")
    exit(1)

# Drop rows with NaN qualifying times
qualifying_data_valid = qualifying_data.dropna(subset=["QualifyingTime (s)"])
print(f"Number of drivers with valid qualifying times: {len(qualifying_data_valid)}")

if len(qualifying_data_valid) == 0:
    print("No valid qualifying times found. Exiting.")
    exit(1)

# List of available driver codes from historical data
available_drivers = driver_race_pace["Driver"].tolist()
print(f"Drivers with historical race data: {available_drivers}")

# Filter qualifying data to only include drivers with historical data
qualifying_filtered = qualifying_data_valid[qualifying_data_valid["DriverCode"].isin(available_drivers)]
print(f"\nFiltered out {len(qualifying_data_valid) - len(qualifying_filtered)} drivers without historical data")
print(f"Remaining drivers for prediction: {len(qualifying_filtered)}")

if len(qualifying_filtered) == 0:
    print("No drivers with both qualifying times and historical data. Exiting.")
    exit(1)

# Merge qualifying data with race pace data
model_data = qualifying_filtered.merge(driver_race_pace, left_on="DriverCode", right_on="Driver", suffixes=('', '_RacePace'))

# Verify data is properly merged
print(f"Model data shape: {model_data.shape}")
if model_data.shape[0] == 0:
    raise ValueError("Dataset empty after preprocessing. Check merge conditions.")

# Prepare data for training
X = model_data[["QualifyingTime (s)"]].values  # 2D array for sklearn
y = model_data["MedianLapTime (s)"].values  # Target values

# Split data for training and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a gradient boosting regressor
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Validate model on test set
y_pred_test = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
print(f"\nModel Error (MAE) on test set: {test_mae:.2f} seconds\n")

# Predict race pace for all drivers based on qualifying times
qualifying_filtered_X = qualifying_filtered[["QualifyingTime (s)"]].values  # 2D array
predicted_lap_times = model.predict(qualifying_filtered_X)

# Calculate total race time (number of laps for the Australian GP)
NUM_LAPS = 58
qualifying_filtered["PredictedLapTime (s)"] = predicted_lap_times
qualifying_filtered["PredictedRaceTime (s)"] = qualifying_filtered["PredictedLapTime (s)"] * NUM_LAPS

# Sort by predicted race time to get final standings
final_standings = qualifying_filtered.sort_values(by="PredictedRaceTime (s)")

# Display predicted race winner and standings
print(f"\n=== Predicted {race_name} Race Results (Drivers with Historical Data) ===\n")
print(final_standings[["Driver", "QualifyingTime (s)", "PredictedLapTime (s)", "PredictedRaceTime (s)"]].head(20))

# Calculate time deltas from winner
winner_time = final_standings.iloc[0]["PredictedRaceTime (s)"]
final_standings["Gap to Winner (s)"] = final_standings["PredictedRaceTime (s)"] - winner_time

# Format results for better display
print("\n=== Final Race Standings with Time Gaps ===\n")
for i, (_, driver) in enumerate(final_standings.iterrows(), 1):
    gap = driver["Gap to Winner (s)"]
    gap_str = f"+{gap:.3f}s" if i > 1 else "WINNER"
    print(f"{i}. {driver['Driver']} ({driver['DriverCode']}) - {gap_str}")

# Print excluded drivers due to lack of historical data
excluded_due_to_history = qualifying_data_valid[~qualifying_data_valid["DriverCode"].isin(available_drivers)]
if not excluded_due_to_history.empty:
    print("\n=== Excluded Drivers (No Historical Data Available) ===")
    for _, driver in excluded_due_to_history.iterrows():
        print(f"- {driver['Driver']} ({driver['DriverCode']})")

# Print drivers excluded due to missing qualifying time
excluded_due_to_qualifying = qualifying_data[pd.isna(qualifying_data["QualifyingTime (s)"])]
if not excluded_due_to_qualifying.empty:
    print("\n=== Excluded Drivers (No Valid Qualifying Time) ===")
    for _, driver in excluded_due_to_qualifying.iterrows():
        print(f"- {driver['Driver']} ({driver['DriverCode']})")

# Visualize the results
plt.figure(figsize=(12, 8))
sns.barplot(x="Gap to Winner (s)", y="Driver", data=final_standings.iloc[:10], palette="viridis", hue="DriverCode")
plt.title(f"Predicted Time Gaps to Winner - {race_name}\n(Only Drivers with Historical Data)", fontsize=16)
plt.xlabel("Gap to Winner (seconds)", fontsize=12)
plt.ylabel("Driver", fontsize=12)
plt.tight_layout()
plt.savefig("predicted_race_results.png")
print("\nResults visualization saved as 'predicted_race_results.png'")


    






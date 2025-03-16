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

# Load 2024 Australian GP race data (round 3)
laps_2024 = load_race_data(2024, 3, "R")

if laps_2024 is None:
    print("Failed to load race data. Exiting.")
    exit(1)

# Process lap data - use median lap time for each driver to get representative race pace
driver_race_pace = laps_2024.groupby("Driver")["LapTime (s)"].median().reset_index()
driver_race_pace.rename(columns={"LapTime (s)": "MedianLapTime (s)"}, inplace=True)
print(f"Processed race pace data for {len(driver_race_pace)} drivers")

# Fictional qualifying data for 2025 Australian GP
qualifying_2025 = pd.DataFrame(
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
         76.450, 76.500, 76.550, 76.600]  # Fixed None value
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

# Add driver codes to the qualifying dataframe
qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Add some fictional historical data for drivers without 2024 data
fictional_data = pd.DataFrame({
    "Driver": ["HAD", "DOO", "BOR", "ANT", "LAW", "BEA"],
    "MedianLapTime (s)": [81.2, 81.3, 81.5, 81.4, 81.6, 81.7]
})

# Combine real and fictional data
full_driver_data = pd.concat([driver_race_pace, fictional_data])

# Merge qualifying data with race pace data
model_data = qualifying_2025.merge(full_driver_data, left_on="DriverCode", right_on="Driver", suffixes=('', '_RacePace'))

# Verify data is properly merged
print(f"Model data shape: {model_data.shape}")
if model_data.shape[0] == 0:
    raise ValueError("Dataset empty after preprocessing. Check merge conditions.")

# Prepare data for training
X = model_data[["QualifyingTime (s)"]].values  # 2D array for sklearn
y = model_data["MedianLapTime (s)"].values  # Target values

# Split data for training and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a gradient boosting REGRESSOR
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Validate model on test set
y_pred_test = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
print(f"\nModel Error (MAE) on test set: {test_mae:.2f} seconds\n")

# Predict race pace for all drivers based on qualifying times
qualifying_2025_X = qualifying_2025[["QualifyingTime (s)"]].values  # 2D array
predicted_lap_times = model.predict(qualifying_2025_X)

# Calculate total race time (assuming 58 laps for Australian GP)
NUM_LAPS = 58
qualifying_2025["PredictedLapTime (s)"] = predicted_lap_times
qualifying_2025["PredictedRaceTime (s)"] = qualifying_2025["PredictedLapTime (s)"] * NUM_LAPS

# Sort by predicted race time to get final standings
final_standings = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

# Display predicted race winner and standings
print("\n=== Predicted 2025 Australian GP Race Results ===\n")
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

# Visualize the results
plt.figure(figsize=(12, 8))
sns.barplot(x="Gap to Winner (s)", y="Driver", data=final_standings.iloc[:10], palette="viridis", hue="DriverCode")
plt.title("Predicted Time Gaps to Winner - 2025 Australian GP", fontsize=16)
plt.xlabel("Gap to Winner (seconds)", fontsize=12)
plt.ylabel("Driver", fontsize=12)
plt.tight_layout()
plt.savefig("predicted_race_results.png")
print("\nResults visualization saved as 'predicted_race_results.png'")


    






import pandas as pd

def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """Enhanced feature engineering with more relevant factors."""
    features = pd.DataFrame()

    # Basic timing features
    features["Q1_Time"] = data["Q1_Time"]
    features["Q2_Time"] = data["Q2_Time"]
    features["Q3_Time"] = data["Q3_Time"]

    # Calculate qualifying gaps and improvements
    features["Q2_Improvement"] = data["Q1_Time"] - data["Q2_Time"]
    features["Q3_Improvement"] = data["Q2_Time"] - data["Q3_Time"]
    features["Q3_Gap"] = data["Q3_Time"] - data["Q3_Time"].min()

    # Grid position features
    features["Grid_Position"] = data["Grid_Position"]
    features["Grid_Position_Normalized"] = data["Grid_Position"] / data["Grid_Position"].max()

    # Historical performance features
    features["Last_Race_Position"] = data["Last_Race_Position"]
    features["Last_Race_Points"] = data["Last_Race_Points"]
    features["Season_Points"] = data["Season_Points"]
    features["Season_Position"] = data["Season_Position"]

    # Team performance features
    features["Team_Last_Race_Points"] = data["Team_Last_Race_Points"]
    features["Team_Season_Points"] = data["Team_Season_Points"]
    features["Team_Season_Position"] = data["Team_Season_Position"]

    # Track-specific features
    features["Track_Experience"] = data["Track_Experience"]
    features["Track_Best_Position"] = data["Track_Best_Position"]
    features["Track_Average_Position"] = data["Track_Average_Position"]

    # Driver form features
    features["Recent_Form"] = data["Recent_Form"]  # Average position in last 3 races
    features["Points_Form"] = data["Points_Form"]  # Average points in last 3 races

    # Team reliability features
    features["Team_DNF_Rate"] = data["Team_DNF_Rate"]  # Team's DNF rate in last 5 races
    features["Driver_DNF_Rate"] = data["Driver_DNF_Rate"]  # Driver's DNF rate in last 5 races

    # Weather-related features (if available)
    if "Track_Temperature" in data.columns:
        features["Track_Temperature"] = data["Track_Temperature"]
    if "Air_Temperature" in data.columns:
        features["Air_Temperature"] = data["Air_Temperature"]
    if "Humidity" in data.columns:
        features["Humidity"] = data["Humidity"]

    # Track characteristics features
    features["Track_Type"] = data["Track_Type"]  # Categorical: street, permanent, etc.
    features["Track_Length"] = data["Track_Length"]
    features["Track_Corners"] = data["Track_Corners"]

    # Driver consistency features
    features["Qualifying_Consistency"] = data[
        "Qualifying_Consistency"
    ]  # Standard deviation of qualifying positions
    features["Race_Consistency"] = data["Race_Consistency"]  # Standard deviation of race positions

    # Team car performance features
    features["Team_Straight_Speed"] = data[
        "Team_Straight_Speed"
    ]  # Team's average straight-line speed
    features["Team_Corner_Speed"] = data["Team_Corner_Speed"]  # Team's average cornering speed

    # Fill missing values with appropriate defaults
    features = features.fillna(
        {
            "Q1_Time": features["Q1_Time"].mean(),
            "Q2_Time": features["Q2_Time"].mean(),
            "Q3_Time": features["Q3_Time"].mean(),
            "Q2_Improvement": 0,
            "Q3_Improvement": 0,
            "Q3_Gap": features["Q3_Gap"].mean(),
            "Grid_Position": features["Grid_Position"].max(),
            "Last_Race_Position": features["Last_Race_Position"].mean(),
            "Last_Race_Points": 0,
            "Season_Points": 0,
            "Season_Position": features["Season_Position"].max(),
            "Team_Last_Race_Points": 0,
            "Team_Season_Points": 0,
            "Team_Season_Position": features["Team_Season_Position"].max(),
            "Track_Experience": 0,
            "Track_Best_Position": features["Track_Best_Position"].max(),
            "Track_Average_Position": features["Track_Average_Position"].mean(),
            "Recent_Form": features["Recent_Form"].mean(),
            "Points_Form": 0,
            "Team_DNF_Rate": 0,
            "Driver_DNF_Rate": 0,
            "Qualifying_Consistency": features["Qualifying_Consistency"].mean(),
            "Race_Consistency": features["Race_Consistency"].mean(),
            "Team_Straight_Speed": features["Team_Straight_Speed"].mean(),
            "Team_Corner_Speed": features["Team_Corner_Speed"].mean(),
        }
    )

    # Handle categorical features
    if "Track_Type" in features.columns:
        features = pd.get_dummies(features, columns=["Track_Type"], prefix="Track_Type")

    return features


def process_qualifying_data(self, qualifying_data):
    """Process qualifying data to prepare for race predictions."""
    if qualifying_data is None or qualifying_data.empty:
        self.logger.warning("No qualifying data to process")
        return None

    # Create a copy to avoid modifying the original
    processed_data = qualifying_data.copy()

    # Ensure we have Driver and Team columns
    if "Driver" not in processed_data.columns and "FullName" in processed_data.columns:
        processed_data["Driver"] = processed_data["FullName"]

    if "Team" not in processed_data.columns and "TeamName" in processed_data.columns:
        processed_data["Team"] = processed_data["TeamName"]

    # Ensure we have DriverId column
    if "DriverId" not in processed_data.columns and "Driver" in processed_data.columns:
        processed_data["DriverId"] = processed_data["Driver"].str.replace(" ", "_").str.lower()

    # Ensure we have Grid_Position column
    if "Grid_Position" not in processed_data.columns and "Position" in processed_data.columns:
        processed_data["Grid_Position"] = processed_data["Position"]

    # Add historical performance data
    processed_data = self._add_historical_performance(processed_data)

    # Add track-specific data
    processed_data = self._add_track_data(processed_data)

    # Add reliability data
    processed_data = self._add_reliability_data(processed_data)

    # Add car performance data
    processed_data = self._add_car_performance_data(processed_data)

    # Add weather data if available
    processed_data = self._add_weather_data(processed_data)

    # Fill missing values with appropriate defaults
    processed_data = self._fill_missing_values(processed_data)

    return processed_data


def _add_historical_performance(self, data):
    """Add historical performance data for each driver."""
    # Create a copy to avoid modifying the original
    enhanced_data = data.copy()

    # Initialize columns with default values
    enhanced_data["Last_Race_Position"] = 10  # Middle of the field
    enhanced_data["Last_Race_Points"] = 0
    enhanced_data["Season_Points"] = 0
    enhanced_data["Season_Position"] = 10
    enhanced_data["Recent_Form"] = 10  # Middle of the field
    enhanced_data["Points_Form"] = 0

    # If we have historical data, use it to populate these columns
    if hasattr(self, "historical_races") and self.historical_races:
        for idx, row in enhanced_data.iterrows():
            driver_id = row["DriverId"]

            # Get driver's last race results
            last_races = [race for race in self.historical_races if driver_id in race["results"]]
            if last_races:
                last_race = last_races[-1]
                if driver_id in last_race["results"]:
                    enhanced_data.at[idx, "Last_Race_Position"] = last_race["results"][driver_id][
                        "position"
                    ]
                    enhanced_data.at[idx, "Last_Race_Points"] = last_race["results"][driver_id][
                        "points"
                    ]

            # Calculate season points and position
            season_points = sum(
                race["results"].get(driver_id, {}).get("points", 0)
                for race in self.historical_races
            )
            enhanced_data.at[idx, "Season_Points"] = season_points

            # Calculate recent form (average position in last 3 races)
            recent_races = [
                race for race in self.historical_races[-3:] if driver_id in race["results"]
            ]
            if recent_races:
                positions = [race["results"][driver_id]["position"] for race in recent_races]
                enhanced_data.at[idx, "Recent_Form"] = sum(positions) / len(positions)

                points = [race["results"][driver_id]["points"] for race in recent_races]
                enhanced_data.at[idx, "Points_Form"] = sum(points) / len(points)

    return enhanced_data


def _add_track_data(self, data):
    """Add track-specific data for each driver."""
    # Create a copy to avoid modifying the original
    enhanced_data = data.copy()

    # Initialize columns with default values
    enhanced_data["Track_Experience"] = 0
    enhanced_data["Track_Best_Position"] = 20
    enhanced_data["Track_Average_Position"] = 10
    enhanced_data["Track_Type"] = "permanent"  # Default track type
    enhanced_data["Track_Length"] = 5.0  # Default track length in km
    enhanced_data["Track_Corners"] = 15  # Default number of corners

    # Get current track information
    current_track = self.current_race_info.get("circuit_name", "unknown")

    # If we have historical data, use it to populate these columns
    if hasattr(self, "historical_races") and self.historical_races:
        for idx, row in enhanced_data.iterrows():
            driver_id = row["DriverId"]

            # Get driver's results at this track
            track_races = [
                race
                for race in self.historical_races
                if race.get("circuit_name") == current_track and driver_id in race["results"]
            ]

            if track_races:
                enhanced_data.at[idx, "Track_Experience"] = len(track_races)

                positions = [race["results"][driver_id]["position"] for race in track_races]
                enhanced_data.at[idx, "Track_Best_Position"] = min(positions)
                enhanced_data.at[idx, "Track_Average_Position"] = sum(positions) / len(positions)

    # Set track characteristics based on current track
    if current_track in self.track_data:
        track_info = self.track_data[current_track]
        enhanced_data["Track_Type"] = track_info.get("type", "permanent")
        enhanced_data["Track_Length"] = track_info.get("length", 5.0)
        enhanced_data["Track_Corners"] = track_info.get("corners", 15)

    return enhanced_data


def _add_reliability_data(self, data):
    """Add reliability data for each driver and team."""
    # Create a copy to avoid modifying the original
    enhanced_data = data.copy()

    # Initialize columns with default values
    enhanced_data["Driver_DNF_Rate"] = 0.1  # 10% default DNF rate
    enhanced_data["Team_DNF_Rate"] = 0.1

    # If we have historical data, use it to populate these columns
    if hasattr(self, "historical_races") and self.historical_races:
        # Calculate driver DNF rates
        for idx, row in enhanced_data.iterrows():
            driver_id = row["DriverId"]
            team_name = row["Team"]

            # Get driver's recent races (last 5)
            recent_races = [
                race for race in self.historical_races[-5:] if driver_id in race["results"]
            ]
            if recent_races:
                dnfs = sum(
                    1 for race in recent_races if race["results"][driver_id].get("status") == "DNF"
                )
                enhanced_data.at[idx, "Driver_DNF_Rate"] = dnfs / len(recent_races)

            # Get team's recent races (last 5)
            team_races = []
            for race in self.historical_races[-5:]:
                team_drivers = [
                    d_id
                    for d_id, result in race["results"].items()
                    if result.get("team") == team_name
                ]
                if team_drivers:
                    team_races.append((race, team_drivers))

            if team_races:
                team_dnfs = sum(
                    1
                    for race, drivers in team_races
                    for driver in drivers
                    if race["results"][driver].get("status") == "DNF"
                )
                total_entries = sum(len(drivers) for _, drivers in team_races)
                enhanced_data.at[idx, "Team_DNF_Rate"] = team_dnfs / total_entries

    return enhanced_data


def _add_car_performance_data(self, data):
    """Add car performance data for each team."""
    # Create a copy to avoid modifying the original
    enhanced_data = data.copy()

    # Initialize columns with default values
    enhanced_data["Team_Straight_Speed"] = 0.5  # Normalized between 0-1
    enhanced_data["Team_Corner_Speed"] = 0.5

    # Team performance data (could be derived from historical data or expert knowledge)
    team_performance = {
        "Red Bull Racing": {"straight": 0.95, "corner": 0.90},
        "Mercedes": {"straight": 0.85, "corner": 0.85},
        "Ferrari": {"straight": 0.90, "corner": 0.85},
        "McLaren": {"straight": 0.85, "corner": 0.90},
        "Aston Martin": {"straight": 0.80, "corner": 0.75},
        "Alpine F1 Team": {"straight": 0.75, "corner": 0.70},
        "Williams": {"straight": 0.70, "corner": 0.65},
        "Visa Cash App Racing Bulls F1 Team": {"straight": 0.75, "corner": 0.70},
        "Kick Sauber": {"straight": 0.65, "corner": 0.65},
        "Haas F1 Team": {"straight": 0.70, "corner": 0.70},
    }

    # Apply team performance data
    for idx, row in enhanced_data.iterrows():
        team_name = row["Team"]

        # Handle team name variations
        if team_name == "Red Bull":
            team_name = "Red Bull Racing"
        elif team_name == "Alpine":
            team_name = "Alpine F1 Team"
        elif team_name == "RB" or team_name == "Racing Bulls":
            team_name = "Visa Cash App Racing Bulls F1 Team"
        elif team_name == "Sauber":
            team_name = "Kick Sauber"
        elif team_name == "Haas":
            team_name = "Haas F1 Team"

        # Apply performance data if available
        if team_name in team_performance:
            enhanced_data.at[idx, "Team_Straight_Speed"] = team_performance[team_name]["straight"]
            enhanced_data.at[idx, "Team_Corner_Speed"] = team_performance[team_name]["corner"]

    return enhanced_data


def _add_weather_data(self, data):
    """Add weather data if available."""
    # Create a copy to avoid modifying the original
    enhanced_data = data.copy()

    # Initialize columns with default values
    enhanced_data["Track_Temperature"] = 25.0  # Default track temperature in °C
    enhanced_data["Air_Temperature"] = 20.0
    enhanced_data["Humidity"] = 50.0

    # If we have weather data for the current race, use it
    if hasattr(self, "current_weather") and self.current_weather:
        enhanced_data["Track_Temperature"] = self.current_weather.get("track_temp", 25.0)
        enhanced_data["Air_Temperature"] = self.current_weather.get("air_temp", 20.0)
        enhanced_data["Humidity"] = self.current_weather.get("humidity", 50.0)

    return enhanced_data


def _fill_missing_values(self, data):
    """Fill missing values with appropriate defaults."""
    # Create a copy to avoid modifying the original
    filled_data = data.copy()

    # Define default values for each column type
    numeric_defaults = {
        "Q1_Time": filled_data["Q1_Time"].mean() if "Q1_Time" in filled_data else 90.0,
        "Q2_Time": filled_data["Q2_Time"].mean() if "Q2_Time" in filled_data else 89.0,
        "Q3_Time": filled_data["Q3_Time"].mean() if "Q3_Time" in filled_data else 88.0,
        "Grid_Position": 10,
        "Last_Race_Position": 10,
        "Last_Race_Points": 0,
        "Season_Points": 0,
        "Season_Position": 10,
        "Team_Last_Race_Points": 0,
        "Team_Season_Points": 0,
        "Team_Season_Position": 5,
        "Track_Experience": 0,
        "Track_Best_Position": 10,
        "Track_Average_Position": 10,
        "Recent_Form": 10,
        "Points_Form": 0,
        "Team_DNF_Rate": 0.1,
        "Driver_DNF_Rate": 0.1,
        "Track_Temperature": 25.0,
        "Air_Temperature": 20.0,
        "Humidity": 50.0,
        "Track_Length": 5.0,
        "Track_Corners": 15,
        "Qualifying_Consistency": 0.5,
        "Race_Consistency": 0.5,
        "Team_Straight_Speed": 0.5,
        "Team_Corner_Speed": 0.5,
    }

    # Fill missing values
    for col, default in numeric_defaults.items():
        if col in filled_data.columns:
            filled_data[col] = filled_data[col].fillna(default)

    # Ensure categorical columns are filled
    if "Track_Type" in filled_data.columns:
        filled_data["Track_Type"] = filled_data["Track_Type"].fillna("permanent")

    return filled_data


def process_race_data(self, year: int, race: int) -> pd.DataFrame:
    """Process race data for a specific year and race."""
    results = self.load_session_data(year, race, 'R')
    if results is None:
        return None

    print("Results DataFrame columns:", results.columns.tolist())
    print("\nResults DataFrame head:")
    print(results.head())
    print("\nResults DataFrame dtypes:")
    print(results.dtypes)
    print("\nResults DataFrame info:")
    results.info()

    # Create a copy to avoid modifying the original
    race_df = results.copy()

    # Try different possible column names for grid position
    grid_pos_columns = ['GridPosition', 'grid', 'GridPos', 'StartingGrid', 'Position', 'Q1', 'Q2', 'Q3', 'QualifyingPosition']
    grid_pos_col = None
    for col in grid_pos_columns:
        if col in race_df.columns:
            grid_pos_col = col
            print(f"Using {col} for grid positions")
            break

    if grid_pos_col is None:
        print("No grid position column found in:", race_df.columns.tolist())
        return None

    # Ensure numeric types for relevant columns
    for col in [grid_pos_col, 'Position']:
        if col in race_df.columns:
            race_df[col] = pd.to_numeric(race_df[col], errors='coerce')

    return race_df


def process_actual_race_data(self, year: int, race: int) -> pd.DataFrame:
    """Process actual race data for a specific year and race."""
    # First, get the qualifying data to get grid positions
    quali_data = self.load_session_data(year, race, 'Q')
    if quali_data is None:
        print("Could not load qualifying data")
        return None

    # Get race results
    race_data = self.load_session_data(year, race, 'R')
    if race_data is None:
        print("Could not load race data")
        return None

    print("Race DataFrame columns:", race_data.columns.tolist())
    print("\nRace DataFrame head:")
    print(race_data.head())

    # Create a mapping of driver numbers to their qualifying positions
    quali_positions = {}
    if 'DriverNumber' in quali_data.columns and 'Position' in quali_data.columns:
        for _, row in quali_data.iterrows():
            quali_positions[str(row['DriverNumber'])] = row['Position']

    # Create a new DataFrame with the required columns
    result_df = pd.DataFrame()
    
    # Add required columns from race data
    columns_to_copy = ['DriverNumber', 'Position', 'Status', 'Points', 'Driver', 'Team']
    for col in columns_to_copy:
        if col in race_data.columns:
            result_df[col] = race_data[col]

    # Add grid positions from qualifying data
    result_df['GridPosition'] = result_df['DriverNumber'].astype(str).map(quali_positions)

    # Convert numeric columns
    numeric_columns = ['Position', 'GridPosition', 'Points']
    for col in numeric_columns:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')

    return result_df

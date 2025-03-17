import logging
import random
from datetime import timedelta
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .performance_adjustments import apply_all_adjustments

logger = logging.getLogger("F1Predictor.Models")


class RacePredictor:
    def __init__(self, model=None, feature_columns=None, total_laps=58):
        """Initialize RacePredictor.

        Args:
            model: The trained model to use for predictions
            feature_columns: List of feature columns used by the model
            total_laps: Total number of laps in the race (default: 58)
        """
        self.model = model
        self.feature_columns = feature_columns
        self.total_laps = total_laps
        self.logger = logging.getLogger(__name__)

    def predict_finishing_positions(self, race_features, historical_lap_times=None, race_info=None):
        """
        Predict finishing positions for all drivers

        Args:
            race_features: DataFrame or dictionary with race features from F1FeatureEngineer
            historical_lap_times: DataFrame with historical lap times for reference
            race_info: Dictionary containing race information including:
                - race_number: Current race number in the season
                - track_info: Dictionary with track characteristics
                - weather_conditions: Dictionary with weather information
                - is_sprint_weekend: Boolean indicating if it's a sprint weekend

        Returns:
            DataFrame with predicted finishing positions and race times
        """
        try:
            # Check if race_features is None, empty DataFrame, or empty dictionary
            if race_features is None or (isinstance(race_features, pd.DataFrame) and race_features.empty) or (isinstance(race_features, dict) and not race_features):
                logger.error("Empty race features")
                return pd.DataFrame()

            # Convert dictionary to DataFrame if needed
            if isinstance(race_features, dict):
                race_features = pd.DataFrame(race_features)

            # Ensure required columns exist
            required_columns = [
                "DriverId",
                "FullName",
                "TeamName",
                "GridPosition",
                "RacePaceScore",
                "ProjectedPosition",
                "DNFProbability",
            ]

            for col in required_columns:
                if col not in race_features.columns:
                    logger.error(f"Missing required column: {col}")
                    return pd.DataFrame()

            # Standardize team names if data processor is available
            if hasattr(race_features, 'data_processor'):
                race_features['TeamName'] = race_features['TeamName'].apply(race_features.data_processor.standardize_team_name)
            else:
                # Fallback team name mapping if no data processor
                team_name_mapping = {
                    'Racing Bulls': 'Visa Cash App Racing Bulls F1 Team',
                    'Visa Cash App RB': 'Visa Cash App Racing Bulls F1 Team',
                    'RB': 'Visa Cash App Racing Bulls F1 Team'
                }
                race_features['TeamName'] = race_features['TeamName'].map(lambda x: team_name_mapping.get(x, x))

            # Apply performance adjustments if race_info is provided
            if race_info:
                apply_all_adjustments(
                    race_features,
                    race_info.get('race_number', 1),
                    race_info.get('track_info', {}),
                    race_info.get('weather_conditions', {}),
                    race_info.get('is_sprint_weekend', False)
                )

            # Sort by projected position
            sorted_drivers = race_features.sort_values("ProjectedPosition").copy()

            # Determine which drivers will have DNFs
            dnf_drivers = []
            for _, driver in sorted_drivers.iterrows():
                if random.random() < driver["DNFProbability"]:
                    dnf_drivers.append(driver["DriverId"])

            logger.info(f"Predicted DNFs for {len(dnf_drivers)} drivers: {dnf_drivers}")

            # Simulate race with lap times for each driver
            base_lap_time = 90.0  # Default 1:30.000 lap time
            if historical_lap_times is not None and not historical_lap_times.empty:
                base_lap_time = historical_lap_times.mean()

            # Get the race distance
            if self.total_laps is None:
                self.total_laps = 58  # Default to a standard race length
                logger.info(f"Using default race length of {self.total_laps} laps")

            finishing_times = []
            for i, (_, driver) in enumerate(sorted_drivers.iterrows(), 1):
                driver_id = driver["DriverId"]

                if driver_id in dnf_drivers:
                    # Calculate a random lap for DNF (weighted towards later laps)
                    dnf_lap = int(random.triangular(1, self.total_laps, self.total_laps * 0.7))

                    finishing_times.append({
                        "DriverId": driver_id,
                        "FullName": driver["FullName"],
                        "TeamName": driver["TeamName"],
                        "GridPosition": driver["GridPosition"],
                        "Position": len(sorted_drivers) - len(dnf_drivers) + len(finishing_times) + 1,
                        "FinishTime": None,
                        "FinishStatus": f"DNF (Lap {dnf_lap})",
                        "CompletedLaps": dnf_lap,
                        "DNF": True,
                    })
                else:
                    # Calculate finish time based on pace score with increased variability
                    pace_factor = 1.0 - (driver["RacePaceScore"] * 0.2)  # 0.8 to 1.0 range
                    avg_lap_time = base_lap_time * pace_factor

                    # Increase lap time variation (8% variation, up from 5%)
                    lap_time_variation = 0.08
                    total_time = 0

                    # Track position effects
                    dirty_air_factor = 1.02  # 2% slower in dirty air
                    drs_factor = 0.98  # 2% faster with DRS
                    
                    # Tire compounds and strategies
                    tire_compounds = ['soft', 'medium', 'hard']
                    current_compound = random.choice(tire_compounds)
                    tire_age = 0
                    last_pit_lap = 0

                    # Simulate each lap
                    for lap in range(1, self.total_laps + 1):
                        # First lap effects
                        if lap == 1:
                            lap_time = avg_lap_time * 1.1  # 10% slower first lap
                        else:
                            # Random variation in lap time
                            variation = 1.0 + (random.random() * 2 - 1) * lap_time_variation
                            
                            # Enhanced tire degradation effect
                            tire_age += 1
                            tire_deg_factor = 1.0 + (tire_age / 20) * 0.05  # Up to 5% slower
                            
                            # Pit stop simulation
                            if (tire_age >= 20 and lap > last_pit_lap + 5) or tire_age >= 30:
                                current_compound = random.choice([c for c in tire_compounds if c != current_compound])
                                tire_age = 0
                                last_pit_lap = lap
                                total_time += random.uniform(20, 23)  # Pit stop time
                            
                            # Traffic and position effects
                            if 5 <= i <= 15:  # Midfield positions
                                variation *= dirty_air_factor
                                if random.random() < 0.7:  # 70% chance of DRS in midfield
                                    variation *= drs_factor
                            
                            # Weather effects (if provided in race_info)
                            if race_info and race_info.get('weather_conditions', {}).get('wet_track'):
                                variation *= random.uniform(1.0, 1.1)  # Up to 10% slower in wet
                            
                            lap_time = avg_lap_time * variation * tire_deg_factor

                        total_time += lap_time

                    finishing_times.append({
                        "DriverId": driver_id,
                        "FullName": driver["FullName"],
                        "TeamName": driver["TeamName"],
                        "GridPosition": driver["GridPosition"],
                        "Position": i,
                        "FinishTime": total_time,
                        "FinishStatus": "Finished",
                        "CompletedLaps": self.total_laps,
                        "DNF": False,
                    })

            # Create DataFrame and sort by finish time for non-DNF drivers
            results_df = pd.DataFrame(finishing_times)

            # Sort by finish time (DNF drivers will be at the end)
            finished_drivers = results_df[~results_df["DNF"]].sort_values("FinishTime").copy()
            dnf_drivers_df = results_df[results_df["DNF"]].copy()

            # Assign final positions
            for i, (idx, _) in enumerate(finished_drivers.iterrows(), 1):
                finished_drivers.loc[idx, "Position"] = i

            for i, (idx, _) in enumerate(dnf_drivers_df.iterrows(), len(finished_drivers) + 1):
                dnf_drivers_df.loc[idx, "Position"] = i

            # Combine and return
            final_results = pd.concat([finished_drivers, dnf_drivers_df])

            return final_results
        except Exception as e:
            logger.error(f"Error predicting finishing positions: {e}")
            return pd.DataFrame()

    def format_race_results(self, race_results):
        """
        Format race results for pretty printing

        Args:
            race_results: DataFrame with race results

        Returns:
            DataFrame with formatted race results
        """
        try:
            formatted_results = race_results.copy()

            # Format time gaps
            if "FinishTime" in formatted_results.columns:
                # Calculate gap to winner
                finished_drivers = formatted_results[~formatted_results["DNF"]]
                if not finished_drivers.empty:
                    winner_time = finished_drivers["FinishTime"].min()
                    for idx, row in finished_drivers.iterrows():
                        gap = row["FinishTime"] - winner_time
                        formatted_results.loc[idx, "GapToWinner"] = gap

            # Format time gaps
            for idx, row in formatted_results.iterrows():
                if "DNF" in row and not row["DNF"]:
                    if "GapToWinner" in row and pd.notna(row["GapToWinner"]):
                        gap_seconds = row["GapToWinner"]
                        if gap_seconds == 0:  # Winner
                            formatted_results.loc[idx, "FormattedGap"] = ""
                        else:
                            # Format gap with proper precision
                            if gap_seconds < 1:
                                # For gaps less than 1 second, show milliseconds
                                formatted_results.loc[idx, "FormattedGap"] = f"+{gap_seconds:.3f}s"
                            elif gap_seconds < 60:
                                # For gaps less than a minute, show seconds with 3 decimal places
                                formatted_results.loc[idx, "FormattedGap"] = f"+{gap_seconds:.3f}s"
                            else:
                                # For larger gaps, format as minutes:seconds.milliseconds
                                minutes, seconds = divmod(gap_seconds, 60)
                                formatted_results.loc[idx, "FormattedGap"] = f"+{int(minutes)}:{seconds:06.3f}"
                    else:
                        formatted_results.loc[idx, "FormattedGap"] = ""
                else:
                    formatted_results.loc[idx, "FormattedGap"] = "DNF"

            # Format race time
            for idx, row in formatted_results.iterrows():
                if (
                    "DNF" in row
                    and not row["DNF"]
                    and "FinishTime" in row
                    and pd.notna(row["FinishTime"])
                ):
                    total_seconds = row["FinishTime"]
                    # Convert to hours:minutes:seconds.milliseconds
                    hours, remainder = divmod(total_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    
                    # Format with proper precision
                    if hours > 0:
                        formatted_results.loc[idx, "FormattedRaceTime"] = (
                            f"{int(hours)}:{int(minutes):02d}:{seconds:06.3f}"
                        )
                    else:
                        formatted_results.loc[idx, "FormattedRaceTime"] = (
                            f"{int(minutes):02d}:{seconds:06.3f}"
                        )
                else:
                    formatted_results.loc[idx, "FormattedRaceTime"] = "DNF"

            # Calculate position changes
            if (
                "GridPosition" in formatted_results.columns
                and "Position" in formatted_results.columns
            ):
                formatted_results["PositionChange"] = formatted_results["GridPosition"].astype(
                    float
                ) - formatted_results["Position"].astype(float)

                # Format position changes
                for idx, row in formatted_results.iterrows():
                    if "DNF" in row and not row["DNF"]:
                        change = row["PositionChange"]
                        if change > 0:
                            formatted_results.loc[idx, "FormattedPositionChange"] = (
                                f"+{int(change)}"
                            )
                        elif change < 0:
                            formatted_results.loc[idx, "FormattedPositionChange"] = f"{int(change)}"
                        else:
                            formatted_results.loc[idx, "FormattedPositionChange"] = "="
                    else:
                        formatted_results.loc[idx, "FormattedPositionChange"] = ""

            return formatted_results
        except Exception as e:
            logger.error(f"Error formatting race results: {e}")
            return race_results

    def calculate_points(self, race_results):
        """
        Calculate championship points for each driver

        Args:
            race_results: DataFrame with race results

        Returns:
            DataFrame with added points column
        """
        # Standard F1 points system
        points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

        # Add fastest lap point (simplified - we'll give it to the winner for now)
        with_points = race_results.copy()

        # Initialize points column
        with_points["Points"] = 0

        # Assign points based on finishing position
        for idx, row in with_points.iterrows():
            if "DNF" in row and not row["DNF"] and "Position" in row and pd.notna(row["Position"]):
                position = int(row["Position"])
                if position in points_system:
                    with_points.loc[idx, "Points"] = points_system[position]

                    # Add fastest lap point to the winner (simplified)
                    if position == 1:
                        with_points.loc[idx, "Points"] += 1
                        with_points.loc[idx, "FastestLap"] = True
                    else:
                        with_points.loc[idx, "FastestLap"] = False
                else:
                    with_points.loc[idx, "Points"] = 0
                    with_points.loc[idx, "FastestLap"] = False
            else:
                # DNF drivers get no points
                with_points.loc[idx, "Points"] = 0
                with_points.loc[idx, "FastestLap"] = False

        return with_points

    def visualize_race_results(self, race_results, output_dir="results"):
        """
        Visualize race results with team colors
        
        Args:
            race_results: DataFrame with race results
            output_dir: Directory to save visualizations
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Team colors dictionary
            team_colors = {
                'Red Bull Racing': '#0600EF',
                'Mercedes': '#00D2BE',
                'Ferrari': '#DC0000',
                'McLaren': '#FF8700',
                'Aston Martin': '#006F62',
                'Alpine F1 Team': '#0090FF',
                'Williams': '#005AFF',
                'Visa Cash App Racing Bulls F1 Team': '#2B4562',  # Main team name
                'Kick Sauber': '#900000',  # Updated team name
                'Haas F1 Team': '#FFFFFF',
                'Alpine': '#0090FF',  # Alias
                'Sauber': '#900000',  # Alias for Kick Sauber
                'Haas': '#FFFFFF'  # Alias
            }
            
            # Team name mapping for standardization
            team_name_mapping = {
                'Racing Bulls': 'Visa Cash App Racing Bulls F1 Team',
                'Visa Cash App RB': 'Visa Cash App Racing Bulls F1 Team',
                'RB': 'Visa Cash App Racing Bulls F1 Team',
                'Alpine': 'Alpine F1 Team',
                'Sauber': 'Kick Sauber',
                'Haas': 'Haas F1 Team'
            }
            
            # Standardize team names
            race_results['TeamName'] = race_results['TeamName'].map(lambda x: team_name_mapping.get(x, x))
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            
            # Plot race results with team colors
            for _, result in race_results.iterrows():
                team_name = result['TeamName']
                color = team_colors.get(team_name, '#808080')  # Default to gray if team not found
                
                plt.bar(result['Position'], 1, color=color, alpha=0.7)
                plt.text(result['Position'], 0.5, f"{result['DriverId']}\n{team_name}", 
                        ha='center', va='center', color='white', fontweight='bold')
            
            plt.title('Race Results by Team')
            plt.xlabel('Position')
            plt.ylabel('')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            
            # Save plot
            output_file = os.path.join(output_dir, 'race_results.png')
            plt.savefig(output_file)
            plt.close()
            
            logger.info(f"Saved race results visualization to {output_file}")
            
        except Exception as e:
            logger.error(f"Error visualizing race results: {e}")

    def predict_and_visualize(self, race_features, title=None, output_file=None, race_info=None):
        """
        Predict race results and create visualization

        Args:
            race_features: DataFrame with race features
            title: Title for the visualization
            output_file: Path to save visualization
            race_info: Dictionary containing race information including:
                - race_number: Current race number in the season
                - track_info: Dictionary with track characteristics
                - weather_conditions: Dictionary with weather information
                - is_sprint_weekend: Boolean indicating if it's a sprint weekend

        Returns:
            DataFrame with formatted race results including points
        """
        try:
            # Make predictions
            race_results = self.predict_finishing_positions(race_features, race_info=race_info)
            if race_results.empty:
                return None

            # Format the results for display
            formatted_results = self.format_race_results(race_results)
            
            # Calculate points
            formatted_results_with_points = self.calculate_points(formatted_results)

            # Create visualization
            self.visualize_race_results(formatted_results_with_points, output_dir=output_file)

            return formatted_results_with_points  # Return formatted results with points
        except Exception as e:
            logger.error(f"Error in predict_and_visualize: {e}")
            return None

    def prepare_prediction_data(self, qualifying_data, driver_features, team_performance):
        """
        Prepare data for race position predictions.

        Args:
            qualifying_data (pd.DataFrame): Qualifying data for the race
            driver_features (pd.DataFrame): Driver features and statistics
            team_performance (pd.DataFrame): Team performance metrics

        Returns:
            pd.DataFrame: Prepared data for predictions
        """
        try:
            if qualifying_data is None or qualifying_data.empty:
                self.logger.warning("No qualifying data available for predictions")
                return None

            prediction_data = qualifying_data.merge(
                driver_features, on="DriverId", how="left"
            ).merge(team_performance, on="Constructor", how="left")

            prediction_data = prediction_data.fillna(prediction_data.mean())
            prediction_data = prediction_data.fillna(0)

            return prediction_data

        except Exception as err:
            self.logger.error(f"Error preparing prediction data: {str(err)}")
            return None

    def predict_race_positions(self, prediction_data):
        """
        Predict race positions using the trained model.

        Args:
            prediction_data (pd.DataFrame): Prepared data for predictions

        Returns:
            pd.DataFrame: Predicted race positions
        """
        try:
            if prediction_data is None or prediction_data.empty:
                self.logger.warning("No data available for predictions")
                return None

            predictions = self.model.predict(prediction_data[self.feature_columns])
            prediction_data["PredictedPosition"] = predictions

            cols = ["DriverId", "Driver", "Constructor", "PredictedPosition"]
            result = prediction_data[cols].sort_values("PredictedPosition")

            return result

        except Exception as err:
            self.logger.error(f"Error making predictions: {str(err)}")
            return None


def print_race_results(race_results):
    """
    Pretty print race results in F1-style format

    Args:
        race_results: DataFrame with race results
    """
    try:
        if race_results.empty:
            print("No race results to display")
            return

        # Select and order columns for display
        display_columns = [
            "Position",
            "DriverId",
            "FullName",
            "TeamName",
            "CompletedLaps",
            "FormattedRaceTime",
            "FormattedGap",
            "FormattedPositionChange",
            "Points",
        ]

        # Filter columns that exist in the DataFrame
        existing_columns = [col for col in display_columns if col in race_results.columns]

        # Define column widths for better alignment
        col_widths = {
            "Position": 3,
            "DriverId": 5,
            "FullName": 18,
            "TeamName": 24,
            "CompletedLaps": 5,
            "FormattedRaceTime": 18,
            "FormattedGap": 14,
            "FormattedPositionChange": 7,
            "Points": 4
        }

        # Print header with a more professional look
        print("\n╔═══════════════════════════════════════ RACE RESULTS ═══════════════════════════════════════╗\n")
        header = (
            f"{'POS':<{col_widths['Position']}} "
            f"{'NO':<{col_widths['DriverId']}} "
            f"{'DRIVER':<{col_widths['FullName']}} "
            f"{'TEAM':<{col_widths['TeamName']}} "
            f"{'LAPS':<{col_widths['CompletedLaps']}} "
            f"{'TIME/RETIRED':<{col_widths['FormattedRaceTime']}} "
            f"{'GAP':<{col_widths['FormattedGap']}} "
            f"{'CHANGE':<{col_widths['FormattedPositionChange']}} "
            f"{'PTS':<{col_widths['Points']}}"
        )
        print(header)
        print("═" * len(header))

        # Sort by position
        sorted_results = (
            race_results.sort_values("Position")
            if "Position" in race_results.columns
            else race_results
        )

        # Print each row
        for _, row in sorted_results.iterrows():
            pos = (
                str(int(row["Position"])) if "Position" in row and pd.notna(row["Position"]) else ""
            )
            driver_id = row["DriverId"] if "DriverId" in row else ""
            name = row["FullName"] if "FullName" in row else ""
            team = row["TeamName"] if "TeamName" in row else ""
            laps = str(int(row["CompletedLaps"])) if "CompletedLaps" in row else ""

            # Handle time or retired status
            if "DNF" in row and row["DNF"]:
                time_or_retired = row.get("FinishStatus", "DNF")
            else:
                time_or_retired = row.get("FormattedRaceTime", "")

            gap = row.get("FormattedGap", "")
            change = row.get("FormattedPositionChange", "")
            points = str(int(row["Points"])) if "Points" in row and row["Points"] > 0 else "0"

            # Print row with consistent spacing
            print(
                f"{pos:<{col_widths['Position']}} "
                f"{driver_id:<{col_widths['DriverId']}} "
                f"{name:<{col_widths['FullName']}} "
                f"{team:<{col_widths['TeamName']}} "
                f"{laps:<{col_widths['CompletedLaps']}} "
                f"{time_or_retired:<{col_widths['FormattedRaceTime']}} "
                f"{gap:<{col_widths['FormattedGap']}} "
                f"{change:<{col_widths['FormattedPositionChange']}} "
                f"{points:<{col_widths['Points']}}"
            )

        print("\n╚════════════════════════════════════════════════════════════════════════════════════════════╝")
        
        # Print fastest lap
        if "FastestLap" in race_results.columns:
            fastest_lap_driver = race_results[race_results["FastestLap"] == True]
            if not fastest_lap_driver.empty:
                fl_driver = fastest_lap_driver.iloc[0]
                print(f"\n🏁 Fastest Lap: {fl_driver['FullName']} ({fl_driver['DriverId']})")

    except Exception as e:
        logger.error(f"Error printing race results: {e}")
        print("Error formatting race results for display")

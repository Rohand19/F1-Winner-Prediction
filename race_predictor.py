import pandas as pd
import numpy as np
import random
import logging
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger("F1Predictor.Race")

class RacePredictor:
    def __init__(self, total_laps=None):
        """
        Initialize the race predictor
        
        Args:
            total_laps: Total number of laps in the race (default: None to auto-detect)
        """
        self.total_laps = total_laps
        random.seed(42)  # For reproducibility
        
    def predict_finishing_positions(self, race_features, historical_lap_times=None):
        """
        Predict finishing positions for all drivers
        
        Args:
            race_features: DataFrame with race features from F1FeatureEngineer
            historical_lap_times: DataFrame with historical lap times for reference
            
        Returns:
            DataFrame with predicted finishing positions and race times
        """
        try:
            if race_features is None or race_features.empty:
                logger.error("Empty race features DataFrame")
                return pd.DataFrame()
                
            # Ensure required columns exist
            required_columns = ["DriverId", "FullName", "TeamName", "GridPosition", 
                               "RacePaceScore", "ProjectedPosition", "DNFProbability"]
            
            for col in required_columns:
                if col not in race_features.columns:
                    logger.error(f"Missing required column: {col}")
                    return pd.DataFrame()
                
            # Sort by projected position
            sorted_drivers = race_features.sort_values("ProjectedPosition").copy()
            
            # Determine which drivers will have DNFs
            dnf_drivers = []
            for _, driver in sorted_drivers.iterrows():
                # Use the DNF probability to determine if a driver will DNF
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
                    # Calculate a random lap for DNF (between lap 1 and total_laps)
                    dnf_lap = random.randint(1, self.total_laps)
                    
                    finishing_times.append({
                        "DriverId": driver_id,
                        "FullName": driver["FullName"],
                        "TeamName": driver["TeamName"],
                        "GridPosition": driver["GridPosition"],
                        "Position": len(sorted_drivers) - len(dnf_drivers) + len(finishing_times) + 1,  # DNF positions are at the end
                        "FinishTime": None,
                        "FinishStatus": f"DNF (Lap {dnf_lap})",
                        "CompletedLaps": dnf_lap,
                        "DNF": True
                    })
                else:
                    # Calculate finish time based on pace score
                    pace_factor = 1.0 - (driver["RacePaceScore"] * 0.1)  # 0.9 to 1.0 range
                    avg_lap_time = base_lap_time * pace_factor
                    
                    # Add some randomness to lap times
                    lap_time_variation = 0.02  # 2% variation
                    total_time = 0
                    
                    for lap in range(1, self.total_laps + 1):
                        # First lap is typically slower
                        if lap == 1:
                            lap_time = avg_lap_time * 1.05
                        else:
                            # Random variation in lap time
                            variation = 1.0 + (random.random() * 2 - 1) * lap_time_variation
                            lap_time = avg_lap_time * variation
                            
                        total_time += lap_time
                    
                    finishing_times.append({
                        "DriverId": driver_id,
                        "FullName": driver["FullName"],
                        "TeamName": driver["TeamName"],
                        "GridPosition": driver["GridPosition"],
                        "Position": i,  # Initial position based on order
                        "FinishTime": total_time,
                        "FinishStatus": "Finished",
                        "CompletedLaps": self.total_laps,
                        "DNF": False
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
            if 'FinishTime' in formatted_results.columns:
                # Calculate gap to winner
                finished_drivers = formatted_results[~formatted_results['DNF']]
                if not finished_drivers.empty:
                    winner_time = finished_drivers['FinishTime'].min()
                    for idx, row in finished_drivers.iterrows():
                        gap = row['FinishTime'] - winner_time
                        formatted_results.loc[idx, 'GapToWinner'] = gap
            
            # Format time gaps
            for idx, row in formatted_results.iterrows():
                if 'DNF' in row and not row['DNF']:
                    if 'GapToWinner' in row and pd.notna(row['GapToWinner']):
                        gap_seconds = row['GapToWinner']
                        if gap_seconds == 0:  # Winner
                            formatted_results.loc[idx, 'FormattedGap'] = ""
                        else:
                            formatted_results.loc[idx, 'FormattedGap'] = f"+{gap_seconds:.3f}s"
                    else:
                        formatted_results.loc[idx, 'FormattedGap'] = ""
                else:
                    formatted_results.loc[idx, 'FormattedGap'] = "DNF"
            
            # Format race time
            for idx, row in formatted_results.iterrows():
                if 'DNF' in row and not row['DNF'] and 'FinishTime' in row and pd.notna(row['FinishTime']):
                    total_seconds = row['FinishTime']
                    # Convert to hours:minutes:seconds.milliseconds
                    time_delta = timedelta(seconds=total_seconds)
                    hours, remainder = divmod(time_delta.seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    formatted_results.loc[idx, 'FormattedRaceTime'] = f"{hours}:{minutes:02d}:{seconds:02d}.{time_delta.microseconds // 1000:03d}"
                else:
                    formatted_results.loc[idx, 'FormattedRaceTime'] = "DNF"
            
            # Calculate position changes
            if 'GridPosition' in formatted_results.columns and 'Position' in formatted_results.columns:
                formatted_results['PositionChange'] = formatted_results['GridPosition'].astype(float) - formatted_results['Position'].astype(float)
            
                # Format position changes
                for idx, row in formatted_results.iterrows():
                    if 'DNF' in row and not row['DNF']:
                        change = row['PositionChange']
                        if change > 0:
                            formatted_results.loc[idx, 'FormattedPositionChange'] = f"+{int(change)}"
                        elif change < 0:
                            formatted_results.loc[idx, 'FormattedPositionChange'] = f"{int(change)}"
                        else:
                            formatted_results.loc[idx, 'FormattedPositionChange'] = "="
                    else:
                        formatted_results.loc[idx, 'FormattedPositionChange'] = ""
            
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
        points_system = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
            6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }
        
        # Add fastest lap point (simplified - we'll give it to the winner for now)
        with_points = race_results.copy()
        
        # Initialize points column
        with_points["Points"] = 0
        
        # Assign points based on finishing position
        for idx, row in with_points.iterrows():
            if 'DNF' in row and not row['DNF'] and 'Position' in row and pd.notna(row['Position']):
                position = int(row['Position'])
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
    
    def visualize_race_results(self, race_results, title=None, output_file=None):
        """
        Create visualizations of race results
        
        Args:
            race_results: DataFrame with race results
            title: Title for the visualization
            output_file: File path to save the visualization
            
        Returns:
            None (displays or saves the visualization)
        """
        try:
            # Filter for only drivers who finished
            if 'DNF' not in race_results.columns:
                logger.warning("DNF column not found in race results")
                return None
                
            finished = race_results[~race_results['DNF']].copy()
            
            if finished.empty:
                logger.warning("No finished drivers to visualize")
                return None
            
            # Create figure with multiple subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1]})
            
            # Calculate time gaps for visualization
            if 'GapToWinner' not in finished.columns:
                # Calculate gap to winner if not already present
                winner_time = finished['FinishTime'].min()
                finished['GapToWinner'] = finished['FinishTime'] - winner_time
                
            finished['GapToWinnerFormatted'] = finished['GapToWinner'].fillna(0)
            
            # Plot time gaps to winner
            sns.barplot(
                x='GapToWinnerFormatted', 
                y='FullName', 
                hue='TeamName',
                data=finished.iloc[:10],  # Top 10 finishers
                ax=ax1
            )
            
            # Configure first subplot
            if title:
                ax1.set_title(f"{title} - Time Gaps to Winner", fontsize=16)
            else:
                ax1.set_title("Race Results - Time Gaps to Winner", fontsize=16)
                
            ax1.set_xlabel("Gap to Winner (seconds)", fontsize=12)
            ax1.set_ylabel("Driver", fontsize=12)
            
            # Add position and gap labels
            for i, (_, row) in enumerate(finished.iloc[:10].iterrows()):
                pos_text = f"{int(row['Position'])}."
                gap_text = f"{row['FormattedGap']}" if 'FormattedGap' in row else ""
                
                ax1.text(
                    -0.5, i, 
                    pos_text, 
                    va='center', 
                    ha='right', 
                    fontweight='bold'
                )
                
                # Add gap text at the end of each bar
                if i > 0:  # Skip winner
                    ax1.text(
                        row['GapToWinnerFormatted'] + 0.1, 
                        i, 
                        gap_text, 
                        va='center', 
                        ha='left'
                    )
            
            # Plot position changes
            if 'PositionChange' in race_results.columns:
                position_changes = race_results[~race_results['DNF']].copy()
                position_changes['PositionChange'] = position_changes['PositionChange'].astype(float)
                
                # Use a different color for positive/negative changes
                colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in position_changes['PositionChange']]
                
                sns.barplot(
                    x='PositionChange', 
                    y='FullName',
                    data=position_changes.iloc[:10],  # Top 10 finishers
                    palette=colors,
                    ax=ax2
                )
                
                # Configure second subplot
                ax2.set_title("Position Changes (Start vs. Finish)", fontsize=16)
                ax2.set_xlabel("Positions Gained/Lost", fontsize=12)
                ax2.set_ylabel("Driver", fontsize=12)
                
                # Add position change labels
                for i, (_, row) in enumerate(position_changes.iloc[:10].iterrows()):
                    change = row['PositionChange']
                    if change != 0:
                        change_text = f"+{int(change)}" if change > 0 else f"{int(change)}"
                        ax2.text(
                            change + (0.1 if change > 0 else -0.1), 
                            i, 
                            change_text, 
                            va='center', 
                            ha='left' if change > 0 else 'right',
                            fontweight='bold',
                            color='white'
                        )
            else:
                ax2.set_title("Position Changes Not Available", fontsize=16)
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {output_file}")
            
            return fig
        except Exception as e:
            logger.error(f"Error visualizing race results: {e}")
            return None
            
    def predict_and_visualize(self, race_features, title=None, output_file=None):
        """
        Full race prediction pipeline
        
        Args:
            race_features: DataFrame with race features from F1FeatureEngineer
            title: Title for the visualization
            output_file: File path to save the visualization
            
        Returns:
            DataFrame with race results
        """
        try:
            # Predict race results
            results = self.predict_finishing_positions(race_features)
            
            if results.empty:
                logger.error("Failed to predict race results")
                return pd.DataFrame()
            
            # Format results
            formatted_results = self.format_race_results(results)
            
            # Calculate points
            with_points = self.calculate_points(formatted_results)
            
            # Visualize results
            if output_file:
                fig = self.visualize_race_results(with_points, title, output_file)
                if fig is None:
                    logger.warning("Failed to create visualization")
            
            return with_points
        except Exception as e:
            logger.error(f"Error in predict_and_visualize: {e}")
            return pd.DataFrame()

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
            "Position", "DriverId", "FullName", "TeamName", 
            "CompletedLaps", "FormattedRaceTime", "FormattedGap", 
            "FormattedPositionChange", "Points"
        ]
        
        # Filter columns that exist in the DataFrame
        existing_columns = [col for col in display_columns if col in race_results.columns]
        
        # Print header
        print("\n=== RACE RESULTS ===\n")
        print(f"{'POS':<4}{'NO':<4}{'DRIVER':<16}{'TEAM':<24}{'LAPS':<6}{'TIME/RETIRED':<16}{'GAP':<10}{'CHANGE':<7}{'PTS':<4}")
        print("-" * 90)
        
        # Sort by position
        sorted_results = race_results.sort_values('Position') if 'Position' in race_results.columns else race_results
        
        # Print each row
        for _, row in sorted_results.iterrows():
            pos = str(int(row["Position"])) if 'Position' in row and pd.notna(row["Position"]) else ""
            driver_id = row["DriverId"]
            name = row["FullName"]
            team = row["TeamName"]
            laps = str(int(row["CompletedLaps"])) if 'CompletedLaps' in row else ""
            
            # Handle time or retired status
            if 'DNF' in row and row['DNF']:
                time_or_retired = row.get('FinishStatus', 'DNF')
            else:
                time_or_retired = row.get('FormattedRaceTime', '')
                
            gap = row.get('FormattedGap', '')
            change = row.get('FormattedPositionChange', '')
            points = str(int(row["Points"])) if "Points" in row and row["Points"] > 0 else "0"
            
            print(f"{pos:<4}{'':<4}{name:<16}{team:<24}{laps:<6}{time_or_retired:<16}{gap:<10}{change:<7}{points:<4}")
        
        # Print fastest lap
        if "FastestLap" in race_results.columns:
            fastest_lap_driver = race_results[race_results["FastestLap"] == True]
            if not fastest_lap_driver.empty:
                fl_driver = fastest_lap_driver.iloc[0]
                print(f"\nFastest Lap: {fl_driver['FullName']} ({fl_driver['DriverId']})")
                
    except Exception as e:
        logger.error(f"Error printing race results: {e}")
        print("Error formatting race results for display") 
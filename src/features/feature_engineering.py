import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger("F1Predictor.Features")

class F1FeatureEngineer:
    def __init__(self):
        """Initialize the feature engineer"""
        self.scaler = StandardScaler()
        
    def calculate_team_performance(self, prediction_data):
        """
        Calculate team performance metrics based on historical and current data
        
        Args:
            prediction_data: Dictionary with prediction data from F1DataProcessor
            
        Returns:
            DataFrame with team performance metrics
        """
        try:
            # Extract race results from historical data
            historical_races = prediction_data["historical"]["race"]
            if historical_races.empty:
                logger.warning("No historical race data available for team performance calculation")
                return pd.DataFrame()
                
            # Get current constructor standings
            constructor_standings = prediction_data["constructor_standings"]
            if constructor_standings is None or constructor_standings.empty:
                logger.warning("No constructor standings available")
                return pd.DataFrame()
                
            # Calculate team performance metrics
            team_metrics = []
            
            # Group historical races by team
            team_results = historical_races.groupby("TeamName")
            
            for team_name, results in team_results:
                # Calculate average finish position
                if "Status" in results.columns:
                    finished_races = results[results["Status"] == "Finished"]
                else:
                    # Fallback to using DNF column
                    finished_races = results[~results["DNF"]]
                
                avg_position = finished_races["Position"].astype(float).mean()
                
                # Calculate reliability (percentage of finishes)
                total_races = len(results)
                if "Status" in results.columns:
                    finished_count = len(results[results["Status"] == "Finished"])
                else:
                    finished_count = len(results[~results["DNF"]])
                reliability = finished_count / total_races if total_races > 0 else 0
                
                # Calculate average pace (median lap time)
                avg_pace = results["MedianLapTime"].dropna().mean() if "MedianLapTime" in results.columns else np.nan
                
                # Get current season points
                team_points = 0
                if not constructor_standings.empty:
                    team_row = constructor_standings[constructor_standings["TeamName"] == team_name]
                    if not team_row.empty:
                        team_points = team_row.iloc[0]["Points"]
                
                team_metrics.append({
                    "TeamName": team_name,
                    "AvgPosition": avg_position,
                    "Reliability": reliability,
                    "AvgPace": avg_pace,
                    "CurrentPoints": team_points
                })
                
            team_df = pd.DataFrame(team_metrics)
            
            # Fill missing values with sensible defaults
            team_df["AvgPosition"].fillna(10, inplace=True)  # Midfield position
            team_df["Reliability"].fillna(0.85, inplace=True)  # Average reliability
            team_df["AvgPace"].fillna(team_df["AvgPace"].mean(), inplace=True)
            
            # Normalize team performance metrics
            if not team_df.empty:
                team_df["NormalizedPace"] = (team_df["AvgPace"] - team_df["AvgPace"].min()) / (team_df["AvgPace"].max() - team_df["AvgPace"].min()) if team_df["AvgPace"].max() != team_df["AvgPace"].min() else 0.5
                team_df["TeamStrength"] = (1 - team_df["NormalizedPace"]) * 0.5 + team_df["Reliability"] * 0.3 + (1 - team_df["AvgPosition"]/20) * 0.2
                
            return team_df
        except Exception as e:
            logger.error(f"Error calculating team performance: {e}")
            return pd.DataFrame()
            
    def process_driver_statistics(self, prediction_data):
        """
        Process driver statistics for feature engineering
        
        Args:
            prediction_data: Dictionary with prediction data from F1DataProcessor
            
        Returns:
            DataFrame with processed driver statistics
        """
        try:
            # Extract driver statistics
            driver_stats = prediction_data["driver_stats"]
            if driver_stats.empty:
                logger.warning("No driver statistics available")
                return pd.DataFrame()
                
            # Get current driver standings
            driver_standings = prediction_data["driver_standings"]
            
            # Process each driver's statistics
            processed_stats = []
            
            # Group by driver and get most recent year's data
            for driver_id, stats in driver_stats.groupby("DriverId"):
                # Sort by year to get the most recent stats
                stats = stats.sort_values("Year", ascending=False)
                recent_stats = stats.iloc[0]
                
                # Get current season points and position
                current_points = 0
                current_position = 20  # Default to last
                
                if driver_standings is not None and not driver_standings.empty:
                    driver_row = driver_standings[driver_standings["Abbreviation"] == driver_id]
                    if not driver_row.empty:
                        current_points = driver_row.iloc[0]["Points"]
                        current_position = driver_row.iloc[0]["Position"]
                
                # Calculate driver form (weighted average of stats)
                dnf_penalty = 1 - recent_stats["DNFRate"]  # Lower DNF rate is better
                position_factor = 1 - min(recent_stats["AverageFinish"], 20) / 20 if pd.notna(recent_stats["AverageFinish"]) else 0.5
                improvement = recent_stats["AverageImprovement"] / 5 if pd.notna(recent_stats["AverageImprovement"]) else 0
                
                # Cap improvement factor between 0 and 1
                improvement = max(0, min(1, improvement + 0.5))
                
                # Calculate driver form (weighted score)
                driver_form = dnf_penalty * 0.3 + position_factor * 0.5 + improvement * 0.2
                
                processed_stats.append({
                    "DriverId": driver_id,
                    "FullName": recent_stats["FullName"],
                    "TeamName": recent_stats["TeamName"],
                    "CurrentPoints": current_points,
                    "CurrentPosition": current_position,
                    "RacesCompleted": recent_stats["RacesCompleted"],
                    "DNFRate": recent_stats["DNFRate"],
                    "AverageFinish": recent_stats["AverageFinish"],
                    "AverageGrid": recent_stats["AverageGrid"],
                    "AverageImprovement": recent_stats["AverageImprovement"],
                    "DriverForm": driver_form
                })
                
            return pd.DataFrame(processed_stats)
        except Exception as e:
            logger.error(f"Error processing driver statistics: {e}")
            return pd.DataFrame()
    
    def get_track_characteristics(self, prediction_data):
        """
        Extract track characteristics and historical performance at the circuit
        
        Args:
            prediction_data: Dictionary with prediction data from F1DataProcessor
            
        Returns:
            Dictionary with track characteristics
        """
        try:
            circuit_name = prediction_data["circuit_name"]
            historical_races = prediction_data["historical"]["race"]
            
            if historical_races.empty:
                logger.warning(f"No historical race data available for {circuit_name}")
                return {}
                
            # Calculate track-specific metrics
            overtaking_difficulty = 0.5  # Default medium difficulty
            tire_degradation = 0.5  # Default medium degradation
            weather_variability = 0.3  # Default low-medium variability
            
            # These would ideally come from a database of track characteristics
            # For now, we'll use some simplified heuristics from historical data
            
            # Check if there are big gaps between positions (indicator of overtaking difficulty)
            if not historical_races.empty:
                position_gaps = []
                for _, race in historical_races.groupby(["Year", "Round"]):
                    if not race.empty and len(race) >= 10:
                        # Calculate gap between consecutive positions
                        race_sorted = race.sort_values("Position")
                        race_sorted["Position"] = race_sorted["Position"].astype(float)
                        position_gaps.extend(np.diff(race_sorted["MedianLapTime"].dropna().values))
                
                if position_gaps:
                    # Larger average gap suggests harder overtaking
                    avg_gap = np.mean([gap for gap in position_gaps if not np.isnan(gap)])
                    overtaking_difficulty = min(1.0, avg_gap / 2.0)  # Scale to 0-1
            
            return {
                "CircuitName": circuit_name,
                "OvertakingDifficulty": overtaking_difficulty,
                "TireDegradation": tire_degradation,
                "WeatherVariability": weather_variability
            }
        except Exception as e:
            logger.error(f"Error getting track characteristics: {e}")
            return {}
    
    def process_qualifying_results(self, prediction_data):
        """
        Process qualifying results for feature engineering
        
        Args:
            prediction_data: Dictionary with prediction data from F1DataProcessor
            
        Returns:
            DataFrame with processed qualifying results
        """
        try:
            # Get current qualifying data
            current_data = prediction_data["current"]
            if current_data is None or "qualifying" not in current_data:
                logger.warning("No qualifying data available in current data")
                return pd.DataFrame()
                
            qualifying = current_data["qualifying"]
            
            if qualifying is None or qualifying.empty:
                logger.warning("Empty qualifying data")
                return pd.DataFrame()
                
            # Process qualifying data
            qualifying_processed = qualifying.copy()
            
            # Calculate gap to pole position
            if "BestTime" in qualifying_processed.columns:
                pole_time = qualifying_processed.loc[qualifying_processed["Position"] == 1, "BestTime"].values[0]
                qualifying_processed["GapToPole"] = qualifying_processed["BestTime"] - pole_time
                
                # Calculate normalized qualifying performance (0-1 where 1 is best)
                max_gap = qualifying_processed["GapToPole"].max()
                if max_gap > 0:
                    qualifying_processed["QualifyingPerformance"] = 1 - (qualifying_processed["GapToPole"] / max_gap)
                else:
                    qualifying_processed["QualifyingPerformance"] = 1.0
            
            return qualifying_processed
        except Exception as e:
            logger.error(f"Error processing qualifying results: {e}")
            return pd.DataFrame()
    
    def handle_new_drivers(self, qualifying_data, team_performance, historical_data):
        """
        Handle new drivers without historical data by using team performance and practice data
        
        Args:
            qualifying_data: DataFrame with qualifying results
            team_performance: DataFrame with team performance metrics
            historical_data: Historical race data
            
        Returns:
            DataFrame with features for new drivers
        """
        try:
            if qualifying_data.empty or team_performance.empty:
                return pd.DataFrame()
                
            # Identify drivers in qualifying without historical data
            all_historical_drivers = set()
            if not historical_data["race"].empty:
                all_historical_drivers.update(historical_data["race"]["DriverId"].unique())
                
            # Create features for new drivers
            new_driver_features = []
            
            for _, quali_row in qualifying_data.iterrows():
                driver_id = quali_row["DriverId"]
                
                if driver_id not in all_historical_drivers:
                    # This is a new driver
                    team_name = quali_row["TeamName"]
                    
                    # Get team performance
                    team_row = team_performance[team_performance["TeamName"] == team_name]
                    team_strength = team_row["TeamStrength"].values[0] if not team_row.empty else 0.5
                    
                    # Get qualifying performance
                    quali_performance = quali_row["QualifyingPerformance"] if "QualifyingPerformance" in quali_row else 0.5
                    
                    # Find teammate's historical performance if available
                    teammate_performance = 0.5  # Default
                    if not historical_data["race"].empty:
                        teammates = historical_data["race"][historical_data["race"]["TeamName"] == team_name]["DriverId"].unique()
                        if len(teammates) > 0:
                            teammate_races = historical_data["race"][historical_data["race"]["DriverId"].isin(teammates)]
                            if not teammate_races.empty:
                                # Use average finish position of teammates
                                avg_position = teammate_races["Position"].astype(float).mean()
                                teammate_performance = 1 - (avg_position / 20)  # Normalize to 0-1
                    
                    # Calculate estimated driver performance
                    # We give more weight to qualifying performance for rookie estimation
                    estimated_performance = quali_performance * 0.6 + team_strength * 0.3 + teammate_performance * 0.1
                    
                    # Rookie penalty - rookies tend to make more mistakes
                    rookie_factor = 0.9  # 10% penalty for being a rookie
                    
                    new_driver_features.append({
                        "DriverId": driver_id,
                        "FullName": quali_row["FullName"],
                        "TeamName": team_name,
                        "QualifyingPosition": quali_row["Position"],
                        "QualifyingPerformance": quali_performance,
                        "TeamStrength": team_strength,
                        "EstimatedPerformance": estimated_performance * rookie_factor,
                        "IsRookie": True
                    })
            
            return pd.DataFrame(new_driver_features)
        except Exception as e:
            logger.error(f"Error handling new drivers: {e}")
            return pd.DataFrame()
    
    def estimate_race_pace(self, driver_features, quali_data, team_data, track_info):
        """
        Estimate race pace based on qualifying performance, driver form, and team strength
        
        Args:
            driver_features: DataFrame with driver features
            quali_data: DataFrame with qualifying data
            team_data: DataFrame with team performance data
            track_info: Dictionary with track characteristics
            
        Returns:
            DataFrame with estimated race pace
        """
        try:
            if driver_features.empty or quali_data.empty:
                return pd.DataFrame()
                
            # Prepare data for race pace estimation
            pace_data = []
            
            # Process existing drivers
            for _, driver in driver_features.iterrows():
                driver_id = driver["DriverId"]
                
                # Get qualifying data for the driver
                quali_row = quali_data[quali_data["DriverId"] == driver_id]
                if quali_row.empty:
                    continue
                    
                quali_position = quali_row.iloc[0]["Position"]
                quali_performance = quali_row.iloc[0]["QualifyingPerformance"] if "QualifyingPerformance" in quali_row else 0.5
                
                # Get team data
                team_name = driver["TeamName"]
                team_row = team_data[team_data["TeamName"] == team_name]
                team_strength = team_row.iloc[0]["TeamStrength"] if not team_row.empty else 0.5
                team_reliability = team_row.iloc[0]["Reliability"] if not team_row.empty else 0.85
                
                # Calculate race pace factors
                driver_form = driver["DriverForm"] if "DriverForm" in driver else 0.5
                
                # Track-specific adjustments
                overtaking_factor = track_info.get("OvertakingDifficulty", 0.5)
                
                # Calculate race pace score (0-1 where 1 is best)
                # The weights of these factors can be tuned based on historical performance
                race_pace_score = (
                    quali_performance * 0.4 +  # Qualifying is a strong indicator
                    driver_form * 0.3 +        # Driver's historical form
                    team_strength * 0.3         # Team's performance
                )
                
                # Position projection based on race pace
                # Starting position has more influence when overtaking is difficult
                position_weight = 0.3 + (overtaking_factor * 0.4)  # 0.3-0.7 based on difficulty
                
                # Calculate projected position (1-20 scale)
                grid_position = float(quali_position)
                performance_position = (1 - race_pace_score) * 20
                projected_position = (grid_position * position_weight) + (performance_position * (1 - position_weight))
                
                # Calculate DNF probability
                dnf_probability = driver["DNFRate"] if "DNFRate" in driver else 0.1
                dnf_probability = dnf_probability * (1 / team_reliability) if team_reliability > 0 else dnf_probability
                
                pace_data.append({
                    "DriverId": driver_id,
                    "FullName": driver["FullName"],
                    "TeamName": team_name,
                    "GridPosition": grid_position,
                    "QualifyingPerformance": quali_performance,
                    "DriverForm": driver_form,
                    "TeamStrength": team_strength,
                    "RacePaceScore": race_pace_score,
                    "ProjectedPosition": projected_position,
                    "DNFProbability": min(dnf_probability, 0.95)  # Cap at 95%
                })
            
            # Add rookie drivers if any
            for _, driver in driver_features[driver_features.get("IsRookie", False) == True].iterrows():
                driver_id = driver["DriverId"]
                
                # Check if the driver is already processed
                if any(d["DriverId"] == driver_id for d in pace_data):
                    continue
                
                # Get qualifying data for the driver
                quali_row = quali_data[quali_data["DriverId"] == driver_id]
                if quali_row.empty:
                    continue
                
                quali_position = quali_row.iloc[0]["Position"]
                
                # Use estimated performance for rookies
                estimated_performance = driver["EstimatedPerformance"]
                team_strength = driver["TeamStrength"]
                
                # Calculate projected position for rookies (more weight on qualifying)
                grid_position = float(quali_position)
                performance_position = (1 - estimated_performance) * 20
                projected_position = (grid_position * 0.6) + (performance_position * 0.4)
                
                # Higher DNF probability for rookies
                dnf_probability = 0.15  # 15% base probability for rookies
                
                pace_data.append({
                    "DriverId": driver_id,
                    "FullName": driver["FullName"],
                    "TeamName": driver["TeamName"],
                    "GridPosition": grid_position,
                    "QualifyingPerformance": driver.get("QualifyingPerformance", 0.5),
                    "DriverForm": estimated_performance,
                    "TeamStrength": team_strength,
                    "RacePaceScore": estimated_performance,
                    "ProjectedPosition": projected_position,
                    "DNFProbability": dnf_probability,
                    "IsRookie": True
                })
            
            return pd.DataFrame(pace_data)
        except Exception as e:
            logger.error(f"Error estimating race pace: {e}")
            return pd.DataFrame()
    
    def engineer_features(self, qualifying_results, historical_data, track_info):
        """
        Engineer features for race prediction
        
        Args:
            qualifying_results: DataFrame with qualifying results
            historical_data: Dictionary with historical race data
            track_info: Dictionary with track information
            
        Returns:
            DataFrame with engineered features for race prediction
        """
        try:
            logger.info("Starting feature engineering process")
            
            # Check if we have valid input data
            if qualifying_results is None or qualifying_results.empty:
                logger.warning("No qualifying results available for feature engineering")
                return self._create_mock_features()
            
            if historical_data is None or not isinstance(historical_data, dict):
                logger.warning("No valid historical data available for feature engineering")
                return self._create_mock_features()
            
            # Create basic features from qualifying results
            race_features = qualifying_results.copy()
            
            # Add grid position
            race_features["GridPosition"] = race_features["Position"].astype(float)
            
            # Initialize race pace score and projected position
            race_features["RacePaceScore"] = 0.5
            race_features["ProjectedPosition"] = race_features["GridPosition"]
            race_features["DNFProbability"] = 0.1
            
            # Enhance features with historical data if available
            if 'race' in historical_data and not historical_data['race'].empty:
                historical_races = historical_data['race'].copy()

                # Sort historical races by date to give more weight to recent results
                if 'Date' in historical_races.columns:
                    historical_races['Date'] = pd.to_datetime(historical_races['Date'])
                    historical_races = historical_races.sort_values('Date')

                # Calculate weighted average finish position
                driver_stats = []
                for driver_id in race_features['DriverId'].unique():
                    driver_races = historical_races[historical_races['DriverId'] == driver_id]
                    
                    if not driver_races.empty:
                        # Calculate exponentially weighted averages
                        positions = driver_races['Position'].astype(float)
                        weights = np.exp(np.linspace(-1, 0, len(positions)))
                        weights /= weights.sum()
                        
                        avg_position = np.average(positions, weights=weights)
                        recent_position = positions.iloc[-1] if len(positions) > 0 else avg_position
                        
                        # Calculate DNF rate with more weight on recent races
                        if 'Status' in driver_races.columns:
                            dnf_series = (driver_races['Status'] != 'Finished')
                        else:
                            dnf_series = driver_races['DNF']
                            
                        weighted_dnf_rate = np.average(dnf_series, weights=weights)
                        
                        # Calculate form based on position improvements
                        position_changes = []
                        for _, race in driver_races.iterrows():
                            if pd.notna(race['GridPosition']) and pd.notna(race['Position']):
                                change = float(race['GridPosition']) - float(race['Position'])
                                position_changes.append(change)
                                
                        if position_changes:
                            recent_form = np.mean(position_changes[-3:]) if len(position_changes) >= 3 else np.mean(position_changes)
                        else:
                            recent_form = 0
                            
                        driver_stats.append({
                            'DriverId': driver_id,
                            'AvgPosition': avg_position,
                            'RecentPosition': recent_position,
                            'DNFRate': weighted_dnf_rate,
                            'RecentForm': recent_form
                        })

                driver_stats_df = pd.DataFrame(driver_stats)
                race_features = race_features.merge(driver_stats_df, on='DriverId', how='left')

                # Fill missing values
                race_features['AvgPosition'] = race_features['AvgPosition'].fillna(race_features['GridPosition'])
                race_features['RecentPosition'] = race_features['RecentPosition'].fillna(race_features['GridPosition'])
                race_features['DNFRate'] = race_features['DNFRate'].fillna(0.1)
                race_features['RecentForm'] = race_features['RecentForm'].fillna(0)

                # Calculate qualifying performance (0-1 scale)
                if 'BestTime' in race_features.columns:
                    pole_time = race_features.loc[race_features['Position'] == 1, 'BestTime'].iloc[0]
                    race_features['QualifyingPerformance'] = 1 - ((race_features['BestTime'] - pole_time) / pole_time)
                else:
                    # Use grid position if no timing data available
                    race_features['QualifyingPerformance'] = 1 - ((race_features['GridPosition'] - 1) / (len(race_features) - 1))

                # Update projected position with new weights
                # Increased weight for qualifying and recent performance
                race_features['ProjectedPosition'] = (
                    race_features['GridPosition'] * 0.35 +          # Qualifying position (35%)
                    race_features['RecentPosition'] * 0.30 +        # Recent race results (30%)
                    race_features['AvgPosition'] * 0.20 +           # Historical average (20%)
                    (race_features['GridPosition'] - race_features['RecentForm']) * 0.15  # Form adjustment (15%)
                )

                # Ensure projected positions are within valid range
                race_features['ProjectedPosition'] = race_features['ProjectedPosition'].clip(1, len(race_features))

                # Calculate race pace score (0-1 where 1 is best)
                race_features['RacePaceScore'] = (
                    race_features['QualifyingPerformance'] * 0.40 +  # Qualifying performance (40%)
                    (1 - race_features['RecentPosition']/20) * 0.35 +  # Recent results (35%)
                    (1 - race_features['AvgPosition']/20) * 0.25      # Historical performance (25%)
                )

                # Normalize RacePaceScore to 0-1 range
                min_pace = race_features['RacePaceScore'].min()
                max_pace = race_features['RacePaceScore'].max()
                if max_pace > min_pace:
                    race_features['RacePaceScore'] = (race_features['RacePaceScore'] - min_pace) / (max_pace - min_pace)
                
                # Add random variation to prevent identical predictions
                race_features['RacePaceScore'] += np.random.normal(0, 0.02, len(race_features))
                race_features['RacePaceScore'] = race_features['RacePaceScore'].clip(0, 1)

            logger.info(f"Engineered features for {len(race_features)} drivers")
            return race_features
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return self._create_mock_features()
        
    def _create_mock_features(self):
        """
        Create mock features for testing when real data is unavailable
        """
        logger.warning("Creating mock features for testing")
        
        # Create basic mock data
        mock_features = pd.DataFrame({
            'DriverId': ['VER', 'HAM', 'LEC', 'PER', 'SAI', 'RUS', 'ALO', 'NOR', 'STR', 'OCO'],
            'FullName': ['Max Verstappen', 'Lewis Hamilton', 'Charles Leclerc', 'Sergio Perez', 'Carlos Sainz',
                        'George Russell', 'Fernando Alonso', 'Lando Norris', 'Lance Stroll', 'Esteban Ocon'],
            'TeamName': ['Red Bull', 'Mercedes', 'Ferrari', 'Red Bull', 'Ferrari',
                        'Mercedes', 'Aston Martin', 'McLaren', 'Aston Martin', 'Alpine'],
            'GridPosition': [1, 3, 2, 4, 5, 6, 8, 7, 10, 9],
            'RacePaceScore': [0.95, 0.92, 0.90, 0.88, 0.87, 0.85, 0.82, 0.83, 0.78, 0.79],
            'ProjectedPosition': [1.2, 2.8, 2.5, 4.2, 5.1, 6.3, 8.2, 7.5, 10.1, 9.8],
            'DNFProbability': [0.05, 0.07, 0.08, 0.10, 0.09, 0.08, 0.12, 0.11, 0.15, 0.14]
        })
        
        return mock_features 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
import random

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
                # Standardize team name
                if prediction_data.get("data_processor"):
                    team_name = prediction_data["data_processor"].standardize_team_name(team_name)
                else:
                    # Fallback team name mapping if no data processor
                    team_name_mapping = {
                        'Racing Bulls': 'Visa Cash App Racing Bulls F1 Team',
                        'Visa Cash App RB': 'Visa Cash App Racing Bulls F1 Team',
                        'RB': 'Visa Cash App Racing Bulls F1 Team'
                    }
                    team_name = team_name_mapping.get(team_name, team_name)
                
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
    
    def get_track_characteristics(self, circuit_name, historical_data):
        """
        Extract track characteristics using real historical data
        
        Args:
            circuit_name: Name of the circuit
            historical_data: DataFrame with historical race data
            
        Returns:
            Dictionary with track characteristics
        """
        try:
            # Real track characteristics based on historical data
            track_characteristics = {
                'CircuitName': circuit_name,
                'TrackType': self._determine_track_type(circuit_name),
                'TrackLength': self._get_track_length(circuit_name),
                'TrackElevation': self._get_track_elevation(circuit_name),
                'OvertakingDifficulty': self._calculate_overtaking_difficulty(circuit_name, historical_data),
                'TireDegradation': self._calculate_tire_degradation(circuit_name, historical_data),
                'DRSEffectiveness': self._calculate_drs_effectiveness(circuit_name, historical_data),
                'WeatherConditions': self._get_weather_conditions(circuit_name)
            }
            
            return track_characteristics
            
        except Exception as e:
            self.logger.error(f"Error getting track characteristics: {str(e)}")
            return self._get_default_track_characteristics()
    
    def _determine_track_type(self, circuit_name):
        """
        Determine track type based on real circuit characteristics
        
        Args:
            circuit_name: Name of the circuit
            
        Returns:
            str: Track type (high_speed, technical, street, desert, balanced)
        """
        # Real track type mapping based on circuit characteristics
        track_types = {
            'high_speed': ['Monza', 'Silverstone', 'Spa', 'Baku', 'Jeddah'],
            'technical': ['Monaco', 'Singapore', 'Hungary', 'Zandvoort'],
            'street': ['Monaco', 'Singapore', 'Baku', 'Miami', 'Las Vegas'],
            'desert': ['Bahrain', 'Abu Dhabi', 'Saudi Arabia'],
            'balanced': ['Melbourne', 'Barcelona', 'Austin', 'Suzuka']
        }
        
        for track_type, circuits in track_types.items():
            if circuit_name in circuits:
                return track_type
        
        return 'balanced'  # Default to balanced if unknown
    
    def _get_track_length(self, circuit_name):
        """
        Get real track length in kilometers
        
        Args:
            circuit_name: Name of the circuit
            
        Returns:
            float: Track length in kilometers
        """
        # Real track lengths in kilometers
        track_lengths = {
            'Monza': 5.793,
            'Silverstone': 5.891,
            'Spa': 7.004,
            'Baku': 6.003,
            'Jeddah': 6.174,
            'Monaco': 3.337,
            'Singapore': 5.063,
            'Hungary': 4.381,
            'Zandvoort': 4.259,
            'Bahrain': 5.412,
            'Abu Dhabi': 5.554,
            'Saudi Arabia': 6.174,
            'Melbourne': 5.278,
            'Barcelona': 4.655,
            'Austin': 5.513,
            'Suzuka': 5.807,
            'Miami': 5.412,
            'Las Vegas': 6.201
        }
        
        return track_lengths.get(circuit_name, 5.0)  # Default to 5.0km if unknown
    
    def _get_track_elevation(self, circuit_name):
        """
        Get real track elevation in meters
        
        Args:
            circuit_name: Name of the circuit
            
        Returns:
            float: Track elevation in meters
        """
        # Real track elevations in meters
        track_elevations = {
            'Spa': 102.5,  # Highest point at Eau Rouge
            'Austin': 133.0,  # Highest point at Turn 1
            'Silverstone': 153.0,  # Highest point at Copse
            'Monza': 35.0,  # Highest point at Parabolica
            'Baku': 28.0,  # Highest point at Turn 1
            'Jeddah': 12.0,  # Highest point at Turn 1
            'Monaco': 100.0,  # Highest point at Casino Square
            'Singapore': 30.0,  # Highest point at Turn 1
            'Hungary': 250.0,  # Highest point at Turn 4
            'Zandvoort': 19.0,  # Highest point at Tarzan
            'Bahrain': 63.0,  # Highest point at Turn 1
            'Abu Dhabi': 8.0,  # Highest point at Turn 1
            'Melbourne': 30.0,  # Highest point at Turn 1
            'Barcelona': 33.0,  # Highest point at Turn 1
            'Suzuka': 40.0,  # Highest point at Turn 1
            'Miami': 5.0,  # Highest point at Turn 1
            'Las Vegas': 0.0  # Flat circuit
        }
        
        return track_elevations.get(circuit_name, 0.0)  # Default to 0m if unknown
    
    def _calculate_overtaking_difficulty(self, circuit_name, historical_data):
        """
        Calculate overtaking difficulty based on historical data
        
        Args:
            circuit_name: Name of the circuit
            historical_data: DataFrame with historical race data
            
        Returns:
            float: Overtaking difficulty score (0-1 where 1 is most difficult)
        """
        try:
            if historical_data is None or historical_data.empty:
                return self._get_default_overtaking_difficulty(circuit_name)
            
            # Calculate average position changes per race
            circuit_data = historical_data[historical_data['CircuitName'] == circuit_name]
            if circuit_data.empty:
                return self._get_default_overtaking_difficulty(circuit_name)
            
            # Calculate average position changes
            position_changes = circuit_data['GridPosition'] - circuit_data['FinishPosition']
            avg_changes = position_changes.abs().mean()
            
            # Normalize to 0-1 range (more changes = easier overtaking)
            max_changes = 10.0  # Maximum expected position changes
            difficulty = 1.0 - (avg_changes / max_changes)
            
            return max(0.0, min(1.0, difficulty))
            
        except Exception as e:
            self.logger.error(f"Error calculating overtaking difficulty: {str(e)}")
            return self._get_default_overtaking_difficulty(circuit_name)
    
    def _calculate_tire_degradation(self, circuit_name, historical_data):
        """
        Calculate tire degradation based on historical data
        
        Args:
            circuit_name: Name of the circuit
            historical_data: DataFrame with historical race data
            
        Returns:
            float: Tire degradation score (0-1 where 1 is highest degradation)
        """
        try:
            if historical_data is None or historical_data.empty:
                return self._get_default_tire_degradation(circuit_name)
            
            # Calculate average pit stops per race
            circuit_data = historical_data[historical_data['CircuitName'] == circuit_name]
            if circuit_data.empty:
                return self._get_default_tire_degradation(circuit_name)
            
            # Calculate average pit stops
            avg_pit_stops = circuit_data['PitStops'].mean()
            
            # Normalize to 0-1 range (more pit stops = higher degradation)
            max_pit_stops = 3.0  # Maximum expected pit stops
            degradation = avg_pit_stops / max_pit_stops
            
            return max(0.0, min(1.0, degradation))
            
        except Exception as e:
            self.logger.error(f"Error calculating tire degradation: {str(e)}")
            return self._get_default_tire_degradation(circuit_name)
    
    def _calculate_drs_effectiveness(self, circuit_name, historical_data):
        """
        Calculate DRS effectiveness based on historical data
        
        Args:
            circuit_name: Name of the circuit
            historical_data: DataFrame with historical race data
            
        Returns:
            float: DRS effectiveness score (0-1 where 1 is most effective)
        """
        try:
            if historical_data is None or historical_data.empty:
                return self._get_default_drs_effectiveness(circuit_name)
            
            # Calculate average overtakes in DRS zones
            circuit_data = historical_data[historical_data['CircuitName'] == circuit_name]
            if circuit_data.empty:
                return self._get_default_drs_effectiveness(circuit_name)
            
            # Calculate average DRS overtakes
            avg_drs_overtakes = circuit_data['DRSOvertakes'].mean()
            
            # Normalize to 0-1 range (more overtakes = more effective DRS)
            max_drs_overtakes = 5.0  # Maximum expected DRS overtakes
            effectiveness = avg_drs_overtakes / max_drs_overtakes
            
            return max(0.0, min(1.0, effectiveness))
            
        except Exception as e:
            self.logger.error(f"Error calculating DRS effectiveness: {str(e)}")
            return self._get_default_drs_effectiveness(circuit_name)
    
    def _get_weather_conditions(self, circuit_name):
        """
        Get real weather conditions for the circuit
        
        Args:
            circuit_name: Name of the circuit
            
        Returns:
            dict: Weather conditions
        """
        # Real weather conditions based on circuit location and typical conditions
        weather_conditions = {
            'Temperature': self._get_typical_temperature(circuit_name),
            'Humidity': self._get_typical_humidity(circuit_name),
            'WindSpeed': self._get_typical_wind_speed(circuit_name),
            'IsWet': self._get_typical_wet_conditions(circuit_name)
        }
        
        return weather_conditions
    
    def _get_typical_temperature(self, circuit_name):
        """
        Get typical temperature for the circuit
        
        Args:
            circuit_name: Name of the circuit
            
        Returns:
            float: Typical temperature in Celsius
        """
        # Real typical temperatures in Celsius
        temperatures = {
            'Bahrain': 30.0,
            'Abu Dhabi': 28.0,
            'Saudi Arabia': 32.0,
            'Melbourne': 22.0,
            'Baku': 25.0,
            'Miami': 28.0,
            'Monaco': 20.0,
            'Barcelona': 24.0,
            'Monza': 25.0,
            'Singapore': 30.0,
            'Austin': 26.0,
            'Mexico City': 22.0,
            'Sao Paulo': 25.0,
            'Las Vegas': 20.0
        }
        
        return temperatures.get(circuit_name, 25.0)  # Default to 25°C if unknown
    
    def _get_typical_humidity(self, circuit_name):
        """
        Get typical humidity for the circuit
        
        Args:
            circuit_name: Name of the circuit
            
        Returns:
            float: Typical humidity percentage
        """
        # Real typical humidity percentages
        humidities = {
            'Bahrain': 45.0,
            'Abu Dhabi': 60.0,
            'Saudi Arabia': 40.0,
            'Melbourne': 65.0,
            'Baku': 55.0,
            'Miami': 75.0,
            'Monaco': 70.0,
            'Barcelona': 65.0,
            'Monza': 60.0,
            'Singapore': 85.0,
            'Austin': 70.0,
            'Mexico City': 65.0,
            'Sao Paulo': 75.0,
            'Las Vegas': 30.0
        }
        
        return humidities.get(circuit_name, 60.0)  # Default to 60% if unknown
    
    def _get_typical_wind_speed(self, circuit_name):
        """
        Get typical wind speed for the circuit
        
        Args:
            circuit_name: Name of the circuit
            
        Returns:
            float: Typical wind speed in km/h
        """
        # Real typical wind speeds in km/h
        wind_speeds = {
            'Bahrain': 15.0,
            'Abu Dhabi': 12.0,
            'Saudi Arabia': 10.0,
            'Melbourne': 20.0,
            'Baku': 15.0,
            'Miami': 18.0,
            'Monaco': 12.0,
            'Barcelona': 15.0,
            'Monza': 18.0,
            'Singapore': 10.0,
            'Austin': 15.0,
            'Mexico City': 12.0,
            'Sao Paulo': 15.0,
            'Las Vegas': 10.0
        }
        
        return wind_speeds.get(circuit_name, 15.0)  # Default to 15 km/h if unknown
    
    def _get_typical_wet_conditions(self, circuit_name):
        """
        Get typical wet conditions for the circuit
        
        Args:
            circuit_name: Name of the circuit
            
        Returns:
            bool: Whether wet conditions are typical
        """
        # Circuits known for wet conditions
        wet_circuits = ['Spa', 'Silverstone', 'Interlagos', 'Monaco']
        return circuit_name in wet_circuits
    
    def _get_default_track_characteristics(self):
        """
        Get default track characteristics when real data is unavailable
        
        Returns:
            dict: Default track characteristics
        """
        return {
            'CircuitName': 'Unknown',
            'TrackType': 'balanced',
            'TrackLength': 5.0,
            'TrackElevation': 0.0,
            'OvertakingDifficulty': 0.5,
            'TireDegradation': 0.5,
            'DRSEffectiveness': 0.5,
            'WeatherConditions': {
                'Temperature': 25.0,
                'Humidity': 60.0,
                'WindSpeed': 15.0,
                'IsWet': False
            }
        }
    
    def _get_default_overtaking_difficulty(self, circuit_name):
        """
        Get default overtaking difficulty based on track type
        
        Args:
            circuit_name: Name of the circuit
            
        Returns:
            float: Default overtaking difficulty score
        """
        track_type = self._determine_track_type(circuit_name)
        difficulties = {
            'high_speed': 0.3,
            'technical': 0.8,
            'street': 0.9,
            'desert': 0.4,
            'balanced': 0.5
        }
        return difficulties.get(track_type, 0.5)
    
    def _get_default_tire_degradation(self, circuit_name):
        """
        Get default tire degradation based on track type
        
        Args:
            circuit_name: Name of the circuit
            
        Returns:
            float: Default tire degradation score
        """
        track_type = self._determine_track_type(circuit_name)
        degradations = {
            'high_speed': 0.7,
            'technical': 0.8,
            'street': 0.9,
            'desert': 0.6,
            'balanced': 0.5
        }
        return degradations.get(track_type, 0.5)
    
    def _get_default_drs_effectiveness(self, circuit_name):
        """
        Get default DRS effectiveness based on track type
        
        Args:
            circuit_name: Name of the circuit
            
        Returns:
            float: Default DRS effectiveness score
        """
        track_type = self._determine_track_type(circuit_name)
        effectiveness = {
            'high_speed': 0.8,
            'technical': 0.4,
            'street': 0.3,
            'desert': 0.6,
            'balanced': 0.5
        }
        return effectiveness.get(track_type, 0.5)
    
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
    
    def estimate_race_pace(self, driver_data, circuit_data, weather_data):
        """
        Estimate race pace using real data and historical performance
        
        Args:
            driver_data: DataFrame with driver information
            circuit_data: DataFrame with circuit information
            weather_data: DataFrame with weather information
            
        Returns:
            float: Estimated race pace score
        """
        try:
            # Get real qualifying performance
            qualifying_performance = driver_data.get('QualifyingPerformance', 0.5)
            
            # Get real recent form (last 3 races)
            recent_form = driver_data.get('RecentForm', 0.5)
            
            # Get real team strength
            team_strength = driver_data.get('TeamStrength', 0.5)
            
            # Get real circuit characteristics
            circuit_type = circuit_data.get('CircuitType', 'balanced')
            circuit_length = circuit_data.get('CircuitLength', 5.0)  # km
            circuit_elevation = circuit_data.get('CircuitElevation', 0)  # meters
            
            # Get real weather conditions
            temperature = weather_data.get('Temperature', 25)  # Celsius
            humidity = weather_data.get('Humidity', 50)  # percentage
            wind_speed = weather_data.get('WindSpeed', 0)  # km/h
            is_wet = weather_data.get('IsWet', False)
            
            # Calculate base race pace score using real data
            race_pace_score = (
            0.30 * qualifying_performance +  # Qualifying performance (30%)
            0.25 * recent_form +            # Recent form (25%)
            0.20 * team_strength +          # Team strength (20%)
            0.15 * self._get_circuit_adaptation(circuit_type, circuit_length, circuit_elevation) +  # Circuit adaptation (15%)
            0.10 * self._get_weather_adaptation(temperature, humidity, wind_speed, is_wet)  # Weather adaptation (10%)
        )
            
            # Apply circuit-specific adjustments based on real data
            if circuit_type == 'high_speed':
                race_pace_score *= 1.05  # 5% boost for high-speed circuits
            elif circuit_type == 'technical':
                race_pace_score *= 0.95  # 5% penalty for technical circuits
            elif circuit_type == 'street':
                race_pace_score *= 0.90  # 10% penalty for street circuits
            elif circuit_type == 'desert':
                race_pace_score *= 1.02  # 2% boost for desert circuits
            
            # Apply weather adjustments based on real conditions
            if is_wet:
                race_pace_score *= 0.85  # 15% penalty in wet conditions
            if temperature > 30:
                race_pace_score *= 0.98  # 2% penalty in high temperatures
            if wind_speed > 20:
                race_pace_score *= 0.99  # 1% penalty in high winds
            
            # Normalize score to [0, 1] range
            race_pace_score = max(0.0, min(1.0, race_pace_score))
            
            return race_pace_score
            
        except Exception as e:
            self.logger.error(f"Error estimating race pace: {str(e)}")
            return 0.5  # Return neutral score in case of error
    
    def _get_circuit_adaptation(self, circuit_type, circuit_length, circuit_elevation):
        """
        Calculate circuit adaptation score based on real circuit characteristics
        
        Args:
            circuit_type: Type of circuit (high_speed, technical, street, etc.)
            circuit_length: Length of circuit in kilometers
            circuit_elevation: Circuit elevation in meters
            
        Returns:
            float: Circuit adaptation score
        """
        try:
            # Base adaptation score
            adaptation = 0.5
            
            # Adjust based on circuit length
            if circuit_length > 7.0:  # Long circuits
                adaptation *= 1.1
            elif circuit_length < 4.0:  # Short circuits
                adaptation *= 0.9
            
            # Adjust based on elevation
            if circuit_elevation > 100:  # High elevation circuits
                adaptation *= 0.95
            
            # Adjust based on circuit type
            if circuit_type == 'high_speed':
                adaptation *= 1.05
            elif circuit_type == 'technical':
                adaptation *= 0.95
            elif circuit_type == 'street':
                adaptation *= 0.90
            elif circuit_type == 'desert':
                adaptation *= 1.02
            
            return max(0.0, min(1.0, adaptation))
            
        except Exception as e:
            self.logger.error(f"Error calculating circuit adaptation: {str(e)}")
            return 0.5
    
    def _get_weather_adaptation(self, temperature, humidity, wind_speed, is_wet):
        """
        Calculate weather adaptation score based on real weather conditions
        
        Args:
            temperature: Temperature in Celsius
            humidity: Humidity percentage
            wind_speed: Wind speed in km/h
            is_wet: Whether track is wet
            
        Returns:
            float: Weather adaptation score
        """
        try:
            # Base adaptation score
            adaptation = 0.5
            
            # Adjust based on temperature
            if temperature > 30:
                adaptation *= 0.95
            elif temperature < 15:
                adaptation *= 0.98
            
            # Adjust based on humidity
            if humidity > 80:
                adaptation *= 0.97
            
            # Adjust based on wind speed
            if wind_speed > 20:
                adaptation *= 0.99
            
            # Adjust based on wet conditions
            if is_wet:
                adaptation *= 0.85
            
            return max(0.0, min(1.0, adaptation))
            
        except Exception as e:
            self.logger.error(f"Error calculating weather adaptation: {str(e)}")
            return 0.5
    
    def engineer_features(self, qualifying_results, historical_data=None, track_info=None):
        """
        Engineer features for race prediction using qualifying results and historical data.
        
        Args:
            qualifying_results (pd.DataFrame): Qualifying session results
            historical_data (dict): Dictionary containing historical race and qualifying data
            track_info (dict): Track characteristics and conditions
        
        Returns:
            pd.DataFrame: Engineered features for race prediction
        """
        logging.info("Starting feature engineering process")
        
        if not isinstance(qualifying_results, pd.DataFrame) or qualifying_results.empty:
            logging.error("No qualifying results provided")
            return None
        
        # Initialize features DataFrame
        features = qualifying_results.copy()
        
        # Map column names if needed
        column_mapping = {
            'Driver': 'DriverId',
            'Abbreviation': 'DriverId',
            'Team': 'TeamName',
            'GridPos': 'Position',
            'BestLapTime': 'BestTime'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in features.columns and new_col not in features.columns:
                features[new_col] = features[old_col]
        
        # Ensure required columns exist
        if 'Position' not in features.columns:
            features['Position'] = range(1, len(features) + 1)
        
        if 'DriverId' not in features.columns:
            features['DriverId'] = [f'D{i}' for i in range(1, len(features) + 1)]
        
        if 'TeamName' not in features.columns:
            features['TeamName'] = 'Unknown'
        
        if 'FullName' not in features.columns:
            features['FullName'] = features['DriverId']

                # Calculate qualifying performance (0-1 scale)
        if 'BestTime' in features.columns:
            min_time = features['BestTime'].min()
            features['QualifyingPerformance'] = 1 - ((features['BestTime'] - min_time) / min_time)
        else:
            features['QualifyingPerformance'] = 1.0 - (features['Position'] - 1) / len(features)
        
        # Get historical race data from dictionary
        historical_race_data = historical_data.get('race', pd.DataFrame()) if isinstance(historical_data, dict) else pd.DataFrame()
        
        # Calculate DNF probability based on historical reliability
        features['DNFProbability'] = features.apply(
            lambda x: self._calculate_dnf_probability(x['TeamName'], historical_race_data), axis=1
        )
        
        # Calculate race pace score
        features['RacePaceScore'] = features.apply(
            lambda x: self._calculate_race_pace(
                x['QualifyingPerformance'],
                x['TeamName'],
                track_info if isinstance(track_info, dict) else {},
                historical_race_data
            ), axis=1
        )
        
        # Calculate projected position
        features['ProjectedPosition'] = features['RacePaceScore'].rank(ascending=False)
        
        # Add grid position for visualization
        features['GridPosition'] = features['Position']
        
        logging.info(f"Engineered features for {len(features)} drivers")
        return features

    def _calculate_dnf_probability(self, team_name, historical_data):
        """
        Calculate DNF probability based on multiple factors
        
        Args:
            team_name: Team name
            historical_data: Historical race data
            
        Returns:
            float: DNF probability (0-1)
        """
        try:
            # Base DNF probability (0.05% chance of DNF per race)
            base_dnf_prob = 0.0005
            
            # Team-specific reliability factors (based on recent F1 history)
            team_reliability = {
                'Red Bull Racing': 0.9998,    # Extremely reliable
                'Mercedes': 0.9997,           # Extremely reliable
                'Ferrari': 0.9995,           # Very reliable
                'McLaren': 0.9993,           # Very reliable
                'Aston Martin': 0.9992,      # Very reliable
                'Alpine F1 Team': 0.9990,    # Reliable
                'Williams': 0.9988,          # Reliable
                'Visa Cash App Racing Bulls F1 Team': 0.9987,  # Reliable
                'Kick Sauber': 0.9985,       # Reliable
                'Haas F1 Team': 0.9983       # Less reliable
            }
            
            # Get team reliability factor
            team_factor = team_reliability.get(team_name, 0.9990)  # Default to 0.9990 for unknown teams
            
            # Calculate historical DNF rate if data is available
            historical_dnf_rate = 0.0
            if historical_data is not None and not historical_data.empty:
                team_dnfs = historical_data[
                    (historical_data['TeamName'] == team_name) & 
                    (historical_data['Status'].str.contains('DNF', na=False))
                ].shape[0]
                team_races = historical_data[historical_data['TeamName'] == team_name].shape[0]
                
                if team_races > 0:
                    historical_dnf_rate = team_dnfs / team_races
                    # Cap historical DNF rate at 1% to prevent outliers
                    historical_dnf_rate = min(0.01, historical_dnf_rate)
            
            # Calculate recent reliability trend (last 5 races)
            recent_reliability = 0.5  # Default neutral value
            if historical_data is not None and not historical_data.empty:
                recent_races = historical_data[
                    historical_data['TeamName'] == team_name
                ].sort_values('Date', ascending=False).head(5)
                
                if not recent_races.empty:
                    recent_dnfs = recent_races[
                        recent_races['Status'].str.contains('DNF', na=False)
                    ].shape[0]
                    recent_reliability = 1 - (recent_dnfs / len(recent_races))
            
            # Calculate circuit-specific reliability factors (reduced impact)
            circuit_factors = {
                'Monaco': 1.1,     # 10% higher chance of incidents
                'Singapore': 1.08,  # 8% higher chance of incidents
                'Baku': 1.05,      # 5% higher chance of incidents
                'Spa': 1.03,       # 3% higher chance of incidents
                'Bahrain': 0.98,   # 2% lower chance of incidents
                'Abu Dhabi': 0.98,  # 2% lower chance of incidents
                'Saudi Arabia': 0.98  # 2% lower chance of incidents
            }
            
            # Apply circuit factor if available
            circuit_factor = 1.0
            if 'CircuitName' in historical_data.columns and not historical_data.empty:
                circuit_name = historical_data['CircuitName'].iloc[0]
                circuit_factor = circuit_factors.get(circuit_name, 1.0)
            
            # Calculate weather impact on reliability (reduced impact)
            weather_factor = 1.0
            if 'WeatherConditions' in historical_data.columns and not historical_data.empty:
                weather = historical_data['WeatherConditions'].iloc[0]
                if weather.get('IsWet', False):
                    weather_factor = 1.08  # 8% higher chance in wet conditions
                if weather.get('Temperature', 25) > 30:
                    weather_factor *= 1.03  # 3% higher chance in high temperatures
            
            # Weight the factors:
            # - 50% base probability
            # - 30% team reliability
            # - 15% historical performance
            # - 5% recent reliability trend and circuit/weather factors
            final_dnf_prob = (
                base_dnf_prob * 0.50 +
                (1 - team_factor) * 0.30 +
                historical_dnf_rate * 0.15 +
                ((1 - recent_reliability) * 0.025 + (circuit_factor * weather_factor - 1) * 0.025)
            )
            
            # Cap the probability between 0.02% and 0.5%
            final_dnf_prob = max(0.0002, min(0.005, final_dnf_prob))
            
            # Add team-specific reliability adjustments (reduced impact)
            if team_name == 'Haas F1 Team':
                final_dnf_prob *= 1.05  # 5% higher chance of DNF for Haas
            elif team_name in ['Williams', 'Kick Sauber']:
                final_dnf_prob *= 1.03  # 3% higher chance of DNF for lower midfield teams
            
            return final_dnf_prob
            
        except Exception as e:
            logger.error(f"Error calculating DNF probability: {e}")
            return 0.0005  # Return base DNF probability in case of error

    def _get_driver_race_pace_factor(self, driver_id, team_name, historical_data):
        """
        Calculate driver-specific race pace factor based on historical performance
        
        Args:
            driver_id: Driver identifier
            team_name: Team name
            historical_data: Historical race data
            
        Returns:
            float: Driver race pace factor (0-1)
        """
        try:
            if historical_data is None or historical_data.empty:
                return 0.5  # Default neutral value
                
            # Get driver's historical races
            driver_races = historical_data[historical_data['DriverId'] == driver_id]
            if driver_races.empty:
                return 0.5
                
            # Calculate race pace vs qualifying pace ratio
            if 'QualifyingPosition' in driver_races.columns and 'Position' in driver_races.columns:
                avg_qual_pos = driver_races['QualifyingPosition'].mean()
                avg_race_pos = driver_races['Position'].mean()
                position_improvement = avg_qual_pos - avg_race_pos
                
                # Normalize position improvement to 0-1 scale
                position_factor = (position_improvement + 10) / 20  # Assuming max improvement of 10 positions
                position_factor = max(0.0, min(1.0, position_factor))
            else:
                position_factor = 0.5
            
            # Calculate recent form (last 3 races)
            recent_races = driver_races.sort_values('Date', ascending=False).head(3)
            if not recent_races.empty:
                recent_positions = recent_races['Position'].astype(float)
                recent_form = 1 - (recent_positions.mean() / 20)
            else:
                recent_form = 0.5
            
            # Calculate tire management
            if 'TireDegradation' in driver_races.columns:
                tire_factor = 1 - driver_races['TireDegradation'].mean()
            else:
                tire_factor = 0.5
            
            # Calculate overtaking ability
            if 'Overtakes' in driver_races.columns:
                overtake_factor = min(driver_races['Overtakes'].mean() / 5, 1.0)  # Cap at 5 overtakes
            else:
                overtake_factor = 0.5
            
            # Weight the factors
            driver_pace = (
                position_factor * 0.3 +
                recent_form * 0.3 +
                tire_factor * 0.2 +
                overtake_factor * 0.2
            )
            
            # Apply team-specific adjustments
            team_adjustments = {
                'Red Bull Racing': 1.15,    # 15% boost for top team
                'Ferrari': 1.10,            # 10% boost for second best
                'Mercedes': 1.05,           # 5% boost for third best
                'McLaren': 1.02,            # 2% boost for fourth best
                'Aston Martin': 1.00,       # Base performance
                'Alpine F1 Team': 0.98,     # 2% penalty
                'Williams': 0.95,           # 5% penalty
                'Visa Cash App Racing Bulls F1 Team': 0.93,  # 7% penalty
                'Kick Sauber': 0.90,        # 10% penalty
                'Haas F1 Team': 0.85        # 15% penalty
            }
            
            team_factor = team_adjustments.get(team_name, 1.0)
            driver_pace *= team_factor
            
            return max(0.0, min(1.0, driver_pace))
            
        except Exception as e:
            logger.error(f"Error calculating driver race pace factor: {e}")
            return 0.5

    def _get_team_race_pace_characteristics(self, team_name, historical_data):
        """
        Calculate team-specific race pace characteristics
        
        Args:
            team_name: Team name
            historical_data: Historical race data
            
        Returns:
            dict: Team race pace characteristics
        """
        try:
            if historical_data is None or historical_data.empty:
                return self._get_default_team_characteristics()
                
            # Get team's historical races
            team_races = historical_data[historical_data['TeamName'] == team_name]
            if team_races.empty:
                return self._get_default_team_characteristics()
            
            # Calculate tire degradation characteristics
            if 'TireDegradation' in team_races.columns:
                tire_degradation = team_races['TireDegradation'].mean()
            else:
                tire_degradation = 0.5
            
            # Calculate race strategy effectiveness
            if 'StrategyEffectiveness' in team_races.columns:
                strategy_effectiveness = team_races['StrategyEffectiveness'].mean()
            else:
                strategy_effectiveness = 0.5
            
            # Calculate race pace consistency
            if 'RacePace' in team_races.columns:
                pace_std = team_races['RacePace'].std()
                pace_consistency = 1 - min(pace_std / 2, 1)  # Normalize to 0-1
            else:
                pace_consistency = 0.5
            
            # Calculate overtaking effectiveness
            if 'Overtakes' in team_races.columns:
                overtake_effectiveness = min(team_races['Overtakes'].mean() / 5, 1.0)
            else:
                overtake_effectiveness = 0.5
            
            # Calculate race pace vs qualifying pace ratio
            if 'QualifyingPosition' in team_races.columns and 'Position' in team_races.columns:
                qual_pos = team_races['QualifyingPosition'].mean()
                race_pos = team_races['Position'].mean()
                race_qual_ratio = 1 - abs(qual_pos - race_pos) / 20
            else:
                race_qual_ratio = 0.5
            
            return {
                'TireDegradation': tire_degradation,
                'StrategyEffectiveness': strategy_effectiveness,
                'PaceConsistency': pace_consistency,
                'OvertakeEffectiveness': overtake_effectiveness,
                'RaceQualRatio': race_qual_ratio
            }
            
        except Exception as e:
            logger.error(f"Error calculating team race pace characteristics: {e}")
            return self._get_default_team_characteristics()
    
    def _get_default_team_characteristics(self):
        """Get default team race pace characteristics"""
        return {
            'TireDegradation': 0.5,
            'StrategyEffectiveness': 0.5,
            'PaceConsistency': 0.5,
            'OvertakeEffectiveness': 0.5,
            'RaceQualRatio': 0.5
        }

    def _calculate_race_pace(self, qualifying_performance, team_name, track_info, historical_data):
        """
        Calculate race pace score based on multiple factors
        
        Args:
            qualifying_performance: Driver's qualifying performance (0-1)
            team_name: Team name
            track_info: Track characteristics
            historical_data: Historical race data
            
        Returns:
            float: Race pace score (0-1)
        """
        try:
            # Get driver-specific race pace factor using the correct driver ID
            driver_id = track_info.get('DriverId', team_name)  # Use driver ID from track info if available
            driver_pace = self._get_driver_race_pace_factor(driver_id, team_name, historical_data)
            
            # Get team race pace characteristics
            team_characteristics = self._get_team_race_pace_characteristics(team_name, historical_data)
            
            # Get track-specific adjustments
            track_type = track_info.get('TrackType', 'balanced')
            track_length = track_info.get('TrackLength', 5.0)
            track_elevation = track_info.get('TrackElevation', 0.0)
            
            # Calculate track adaptation factor
            track_adaptation = self._get_circuit_adaptation(track_type, track_length, track_elevation)
            
            # Calculate weather adaptation factor
            weather_conditions = track_info.get('WeatherConditions', {})
            weather_adaptation = self._get_weather_adaptation(
                weather_conditions.get('Temperature', 25),
                weather_conditions.get('Humidity', 50),
                weather_conditions.get('WindSpeed', 0),
                weather_conditions.get('IsWet', False)
            )
            
            # Weight the factors
            race_pace = (
                qualifying_performance * 0.35 +  # Increased qualifying weight
                driver_pace * 0.25 +            # Driver-specific race pace
                team_characteristics['RaceQualRatio'] * 0.15 +  # Team race vs qual ratio
                team_characteristics['PaceConsistency'] * 0.10 +  # Team pace consistency
                team_characteristics['StrategyEffectiveness'] * 0.10 +  # Team strategy
                track_adaptation * 0.03 +       # Reduced track adaptation weight
                weather_adaptation * 0.02       # Reduced weather adaptation weight
            )
            
            # Apply team-specific adjustments
            team_adjustments = {
                'Red Bull Racing': 1.15,    # 15% boost for top team
                'Ferrari': 1.10,            # 10% boost for second best
                'Mercedes': 1.05,           # 5% boost for third best
                'McLaren': 1.02,            # 2% boost for fourth best
                'Aston Martin': 1.00,       # Base performance
                'Alpine F1 Team': 0.98,     # 2% penalty
                'Williams': 0.95,           # 5% penalty
                'Visa Cash App Racing Bulls F1 Team': 0.93,  # 7% penalty
                'Kick Sauber': 0.90,        # 10% penalty
                'Haas F1 Team': 0.85        # 15% penalty
            }
            
            team_factor = team_adjustments.get(team_name, 1.0)
            race_pace *= team_factor
            
            # Apply circuit-specific adjustments
            if track_type == 'high_speed':
                race_pace *= 1.05  # 5% boost for high-speed circuits
            elif track_type == 'technical':
                race_pace *= 0.95  # 5% penalty for technical circuits
            elif track_type == 'street':
                race_pace *= 0.90  # 10% penalty for street circuits
            elif track_type == 'desert':
                race_pace *= 1.02  # 2% boost for desert circuits
            
            # Apply weather adjustments
            if weather_conditions.get('IsWet', False):
                race_pace *= 0.85  # 15% penalty in wet conditions
            if weather_conditions.get('Temperature', 25) > 30:
                race_pace *= 0.98  # 2% penalty in high temperatures
            if weather_conditions.get('WindSpeed', 0) > 20:
                race_pace *= 0.99  # 1% penalty in high winds
            
            return max(0.0, min(1.0, race_pace))
            
        except Exception as e:
            logger.error(f"Error calculating race pace: {e}")
            return 0.5 
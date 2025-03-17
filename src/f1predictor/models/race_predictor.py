import pandas as pd
import numpy as np
import random
import logging
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import timedelta, datetime

from .performance_adjustments import apply_all_adjustments

logger = logging.getLogger("F1Predictor.Models")


class RacePredictor:
    def __init__(self, model=None, feature_columns=None, total_laps=57):
        """Initialize RacePredictor.

        Args:
            model: The trained model to use for predictions
            feature_columns: List of feature columns used by the model
            total_laps: Total number of laps in the race (default: 57)
        """
        self.model = model
        self.feature_columns = feature_columns
        self.total_laps = total_laps
        self.logger = logging.getLogger(__name__)
        self.base_lap_time = 5.412 * 17.5  # Bahrain GP track length * approximate lap time
        self.track_effect = 0.02
        
        # Track-specific characteristics
        self.track_data = {
            'Bahrain': {
                'overtaking_difficulty': 0.6,  # Medium difficulty (0-1 scale, higher = harder)
                'tire_deg': 0.85,              # High tire degradation
                'track_type': 'desert',        # Track type
                'pit_loss': 23.0,              # Time lost in pit lane in seconds
                'drs_effect': 0.6              # DRS effectiveness (0-1 scale)
            },
            'Saudi Arabia': {
                'overtaking_difficulty': 0.8,  # Difficult to overtake
                'tire_deg': 0.6,               # Lower tire degradation
                'track_type': 'street',
                'pit_loss': 24.5,
                'drs_effect': 0.75             # Stronger DRS effect on long straights
            },
            'Australia': {
                'overtaking_difficulty': 0.7,
                'tire_deg': 0.7,
                'track_type': 'street/permanent',
                'pit_loss': 22.0,
                'drs_effect': 0.5
            },
            'Japan': {
                'overtaking_difficulty': 0.65,
                'tire_deg': 0.75,
                'track_type': 'permanent',
                'pit_loss': 21.0,
                'drs_effect': 0.5
            },
            'China': {
                'overtaking_difficulty': 0.55,
                'tire_deg': 0.8,
                'track_type': 'permanent',
                'pit_loss': 22.0,
                'drs_effect': 0.65
            },
            'Miami': {
                'overtaking_difficulty': 0.6,
                'tire_deg': 0.7,
                'track_type': 'street',
                'pit_loss': 22.5,
                'drs_effect': 0.6
            },
            'Imola': {
                'overtaking_difficulty': 0.85,
                'tire_deg': 0.65,
                'track_type': 'permanent',
                'pit_loss': 21.0,
                'drs_effect': 0.45
            },
            'Monaco': {
                'overtaking_difficulty': 0.95,
                'tire_deg': 0.5,
                'track_type': 'street',
                'pit_loss': 25.0,
                'drs_effect': 0.2
            },
            'Spain': {
                'overtaking_difficulty': 0.7,
                'tire_deg': 0.75,
                'track_type': 'permanent',
                'pit_loss': 21.5,
                'drs_effect': 0.55
            },
            'Canada': {
                'overtaking_difficulty': 0.5,
                'tire_deg': 0.8,
                'track_type': 'street/permanent',
                'pit_loss': 22.0,
                'drs_effect': 0.7
            }
        }
        
        # Default track data if specific track not found
        self.default_track_data = {
            'overtaking_difficulty': 0.65,
            'tire_deg': 0.7,
            'track_type': 'permanent',
            'pit_loss': 22.0,
            'drs_effect': 0.5
        }

    def predict_finishing_positions(self, qualifying_data: pd.DataFrame, track_name='Bahrain', weather_conditions=None) -> pd.DataFrame:
        """Predict race finishing positions with enhanced reliability and incident handling.
        
        Args:
            qualifying_data: DataFrame with qualifying data
            track_name: Name of the track for the race
            weather_conditions: Dict with weather data (temp, humidity, rain_chance, etc.)
        """
        # Make a copy of qualifying data to ensure we don't modify it
        qualifying_copy = qualifying_data.copy()
        
        # Initialize results DataFrame with complete driver information
        results = pd.DataFrame(index=qualifying_copy.index)
        
        # Add important columns
        results['Position'] = qualifying_copy['Grid_Position']
        results['GridPosition'] = qualifying_copy['Grid_Position']
        results['LapsCompleted'] = 0
        results['TotalTime'] = 0.0
        results['Status'] = 'Running'
        results['Points'] = 0
        results['Gap'] = None
        results['StartPosition'] = None
        results['PitStops'] = 0
        
        # Add DriverId if available
        if 'DriverId' in qualifying_copy.columns:
            results['DriverId'] = qualifying_copy['DriverId']
        elif 'Abbreviation' in qualifying_copy.columns:
            results['DriverId'] = qualifying_copy['Abbreviation']
        else:
            results['DriverId'] = ["D" + str(i+1) for i in range(len(results))]
        
        # Add Driver and Team information
        if 'Driver' in qualifying_copy.columns:
            results['Driver'] = qualifying_copy['Driver']
        elif 'FullName' in qualifying_copy.columns:
            results['Driver'] = qualifying_copy['FullName']
        else:
            results['Driver'] = ["Driver " + str(i+1) for i in range(len(results))]
            
        if 'Team' in qualifying_copy.columns:
            results['Team'] = qualifying_copy['Team']
        elif 'TeamName' in qualifying_copy.columns:
            results['Team'] = qualifying_copy['TeamName']
        else:
            results['Team'] = ["Team " + str(i+1) for i in range(len(results))]
        
        # Print driver information for debugging
        print("\nDrivers in race simulation:")
        for idx, row in results.iterrows():
            print(f"  {row['GridPosition']}: {row['Driver']} ({row['Team']})")
        print()
        
        # Get track-specific data
        track_data = self.track_data.get(track_name, self.default_track_data)
        
        # Initialize weather conditions if not provided
        if weather_conditions is None:
            weather_conditions = {
                'track_temp': 35.0,        # Default track temperature
                'air_temp': 25.0,          # Default air temperature
                'humidity': 60.0,          # Default humidity
                'rain_chance': 0.0,        # Default chance of rain (0.0 = 0%, 1.0 = 100%)
                'wind_speed': 5.0,         # Default wind speed in km/h
                'changing_conditions': False # Static conditions by default
            }
        
        # Calculate base lap times and tire degradation with track/weather factors
        base_lap_times = self._calculate_base_lap_times(qualifying_copy, track_name)
        tire_degradation = self._calculate_tire_degradation(qualifying_copy, track_data['tire_deg'])
        
        # Initialize race state
        current_lap = 1
        total_laps = self.total_laps
        safety_car_laps = []
        virtual_safety_car_laps = []
        
        # Initialize tire strategies for each driver
        tire_strategies = {}
        for idx in results.index:
            # Random strategy choice (1-stop, 2-stop, or 3-stop)
            strategy_probs = [0.3, 0.6, 0.1]  # Probabilities for 1, 2, 3 stops
            n_stops = np.random.choice([1, 2, 3], p=strategy_probs)
            
            # Calculate stop laps - distribute evenly with some randomness
            stop_laps = []
            for i in range(n_stops):
                # Base lap is evenly distributed
                base_lap = int((i + 1) * total_laps / (n_stops + 1))
                # Add some randomness (-5 to +5 laps)
                lap = min(total_laps - 5, max(15, base_lap + np.random.randint(-5, 6)))
                stop_laps.append(lap)
            
            # Sort stop laps
            stop_laps.sort()
            
            # Store strategy
            tire_strategies[idx] = {
                'n_stops': n_stops,
                'stop_laps': stop_laps,
                'current_tire_age': 0,
                'tire_compound': 'medium'  # Starting compound
            }
            
            # Top 10 typically start on different tires than others
            if results.at[idx, 'GridPosition'] <= 10:
                tire_strategies[idx]['tire_compound'] = 'soft'
            
        # Team reliability factors - higher values mean better reliability
        team_reliability = {
            'Red Bull Racing': 0.95,
            'Ferrari': 0.92,
            'Mercedes': 0.93,
            'McLaren': 0.90,
            'Aston Martin': 0.88,
            'Alpine': 0.85,
            'Williams': 0.82,
            'RB': 0.88,
            'Kick Sauber': 0.80,
            'Haas F1 Team': 0.85,
            # Include aliases
            'Red Bull': 0.95,
            'Mercedes-AMG': 0.93,
            'Haas': 0.85,
            'Aston Martin Aramco': 0.88,
            'Alpine F1 Team': 0.85,
            'Sauber': 0.80
        }
        
        # Driver skill factors (higher is better)
        driver_skill = {
            'Max Verstappen': 0.95,
            'Lewis Hamilton': 0.93,
            'Fernando Alonso': 0.92,
            'Charles Leclerc': 0.92,
            'Lando Norris': 0.91,
            'Carlos Sainz': 0.90,
            'George Russell': 0.90,
            'Sergio Perez': 0.89,
            'Oscar Piastri': 0.87,
            'Yuki Tsunoda': 0.85,
            'Daniel Ricciardo': 0.86,
            'Alexander Albon': 0.85,
            'Lance Stroll': 0.84,
            'Kevin Magnussen': 0.84,
            'Nico Hulkenberg': 0.84,
            'Esteban Ocon': 0.84,
            'Pierre Gasly': 0.84,
            'Valtteri Bottas': 0.84,
            'Guanyu Zhou': 0.82,
            'Logan Sargeant': 0.80
        }
        
        # Wet weather performance factors (higher = better in rain)
        wet_weather_skill = {
            'Max Verstappen': 0.95,       # Exceptional in wet
            'Lewis Hamilton': 0.94,       # Exceptional in wet
            'Fernando Alonso': 0.93,      # Exceptional in wet
            'Carlos Sainz': 0.88,
            'Charles Leclerc': 0.86,
            'Lando Norris': 0.87,
            'George Russell': 0.87,
            'Sergio Perez': 0.85,
            'Oscar Piastri': 0.82,
            'Lance Stroll': 0.86,         # Relatively good in wet
            'Sebastian Vettel': 0.90,
            'Daniel Ricciardo': 0.85,
            'Alexander Albon': 0.84,
            'Kevin Magnussen': 0.85,
            'Nico Hulkenberg': 0.85,
            'Esteban Ocon': 0.84,
            'Pierre Gasly': 0.83,
            'Valtteri Bottas': 0.85,
            'Guanyu Zhou': 0.80,
            'Logan Sargeant': 0.78
        }
        
        # Base probability of incident per lap (lower for reliable simulation)
        base_incident_prob = 0.001  # 0.1% chance per lap
        
        # Weather conditions can evolve during the race if changing_conditions is True
        if weather_conditions['changing_conditions']:
            # Create a simple weather evolution
            weather_timeline = {}
            
            # Start with current conditions
            current_rain_chance = weather_conditions['rain_chance']
            
            # Define some key points for weather evolution
            weather_timeline[1] = current_rain_chance  # Starting condition
            
            # Random points where weather might change
            change_points = sorted(np.random.choice(
                range(10, total_laps-5), 
                size=min(3, np.random.randint(1, 4)), 
                replace=False
            ))
            
            for lap in change_points:
                # Generate a new rain chance that evolves from the previous one
                new_rain_chance = min(1.0, max(0.0, current_rain_chance + np.random.uniform(-0.3, 0.3)))
                weather_timeline[lap] = new_rain_chance
                current_rain_chance = new_rain_chance
            
            weather_timeline[total_laps] = current_rain_chance  # Ending condition
        else:
            # Static weather
            weather_timeline = {1: weather_conditions['rain_chance']}
        
        # Initialize driver-specific race state
        driver_states = {}
        for idx in results.index:
            driver_states[idx] = {
                'position': results.at[idx, 'GridPosition'],
                'current_lap': 0,
                'in_pit': False,
                'pit_penalty': 0.0,  # Time penalty for pit stop
                'drs_enabled': False,
                'tire_age': 0,
                'energy_management': np.random.uniform(0.7, 1.0),  # How well managing energy
                'race_pace': np.random.uniform(0.8, 1.0),  # Race pace compared to qualifying
                'overtaking_opportunity': 0.0  # Probability of attempting an overtake
            }
        
        # Simulate race with more realistic reliability and strategy
        while current_lap <= total_laps:
            # Update weather conditions for this lap
            current_rain_chance = 0.0
            for lap in sorted(weather_timeline.keys()):
                if current_lap >= lap:
                    current_rain_chance = weather_timeline[lap]
            
            is_wet_lap = current_rain_chance > 0.2  # Consider it wet if rain chance > 20%
            rain_intensity = 0.0
            if is_wet_lap:
                rain_intensity = min(1.0, current_rain_chance * 1.5)  # Scale up for effect
            
            # Update tire degradation and pit stops
            for idx, row in results.iterrows():
                if row['Status'] == 'Running':
                    # Get driver's current state
                    state = driver_states[idx]
                    strategy = tire_strategies[idx]
                    
                    # Check if it's time for a pit stop
                    if current_lap in strategy['stop_laps'] and not state['in_pit']:
                        # Enter pit
                        state['in_pit'] = True
                        
                        # Apply pit stop time penalty
                        pit_time = track_data['pit_loss']
                        
                        # Add some team-specific pit stop variation (±1.5s)
                        team = row['Team']
                        team_pit_factor = np.random.normal(0, 1.5)
                        
                        # Top teams have slightly faster stops on average
                        if team in ['Red Bull Racing', 'Ferrari', 'Mercedes', 'McLaren']:
                            team_pit_factor -= 0.5
                        elif team in ['Alpine', 'Kick Sauber', 'Haas F1 Team']:
                            team_pit_factor += 0.5
                        
                        pit_time += team_pit_factor
                        
                        # Apply the penalty
                        state['pit_penalty'] = pit_time
                        
                        # Update pit stop count
                        results.at[idx, 'PitStops'] += 1
                        
                        # Reset tire age
                        strategy['current_tire_age'] = 0
                        
                        # Switch compound based on race stage
                        race_progress = current_lap / total_laps
                        if race_progress < 0.3:
                            strategy['tire_compound'] = 'medium'
                        elif race_progress < 0.7:
                            strategy['tire_compound'] = 'hard' if np.random.random() < 0.7 else 'medium'
                        else:
                            strategy['tire_compound'] = 'soft' if np.random.random() < 0.6 else 'medium'
                        
                        # Wet conditions use wet or intermediate tires
                        if is_wet_lap:
                            if rain_intensity > 0.6:
                                strategy['tire_compound'] = 'wet'
                            else:
                                strategy['tire_compound'] = 'intermediate'
                    
                    # Calculate lap time based on various factors
                    if state['in_pit']:
                        # Driver is in pit this lap
                        results.at[idx, 'TotalTime'] += state['pit_penalty']
                        state['in_pit'] = False  # Exit pit
                        state['pit_penalty'] = 0.0
                    else:
                        # Regular lap
                        # Get base lap time
                        lap_time = base_lap_times[idx]
                        
                        # Apply tire degradation based on compound and age
                        tire_compound = strategy['tire_compound']
                        tire_age = strategy['current_tire_age']
                        
                        # Different compounds have different degradation profiles
                        if tire_compound == 'soft':
                            deg_factor = tire_degradation[idx] * 1.5  # Higher degradation
                            max_optimal_age = 15  # Optimal performance till this lap
                        elif tire_compound == 'medium':
                            deg_factor = tire_degradation[idx] * 1.0  # Medium degradation
                            max_optimal_age = 25
                        elif tire_compound == 'hard':
                            deg_factor = tire_degradation[idx] * 0.7  # Lower degradation
                            max_optimal_age = 35
                        elif tire_compound == 'intermediate':
                            if rain_intensity > 0.3:  # Good for light rain
                                deg_factor = tire_degradation[idx] * 0.9
                                max_optimal_age = 20
                            else:  # High degradation in dry conditions
                                deg_factor = tire_degradation[idx] * 2.0
                                max_optimal_age = 10
                        elif tire_compound == 'wet':
                            if rain_intensity > 0.7:  # Good for heavy rain
                                deg_factor = tire_degradation[idx] * 0.8
                                max_optimal_age = 25
                            else:  # High degradation in light rain
                                deg_factor = tire_degradation[idx] * 1.8
                                max_optimal_age = 15
                        
                        # Apply non-linear degradation (increases exponentially after optimal age)
                        if tire_age <= max_optimal_age:
                            # Linear phase
                            tire_effect = deg_factor * tire_age * 0.01
                        else:
                            # Exponential phase
                            base_deg = deg_factor * max_optimal_age * 0.01
                            exp_deg = deg_factor * (1 + (tire_age - max_optimal_age) * 0.02)
                            tire_effect = base_deg + exp_deg
                        
                        lap_time *= (1 + tire_effect)
                        
                        # Apply weather effects
                        if is_wet_lap:
                            # Get driver's wet weather skill
                            driver = row['Driver']
                            wet_skill = wet_weather_skill.get(driver, 0.85)
                            
                            # Apply penalty based on rain intensity and skill
                            # Wet conditions slow everyone down, but skilled drivers less so
                            wet_penalty = rain_intensity * 0.12 * (2 - wet_skill)
                            lap_time *= (1 + wet_penalty)
                            
                            # Increase chance of incidents in wet conditions
                            if np.random.random() < rain_intensity * 0.05:
                                # Minor error in wet - add small time penalty
                                lap_time += np.random.uniform(0.5, 3.0)
                        
                        # Apply DRS effect if eligible (not first 2 laps, not in wet)
                        if current_lap > 2 and not is_wet_lap:
                            # Check if driver is within 1 second of the car ahead
                            pos = int(row['Position'])
                            cars_ahead = results[(results['Position'] < pos) & (results['Status'] == 'Running')]
                            if not cars_ahead.empty:
                                # Find the nearest car ahead
                                nearest_ahead = cars_ahead.sort_values('Position', ascending=False).iloc[0]
                                time_gap = results.at[idx, 'TotalTime'] - nearest_ahead['TotalTime']
                                
                                # If within DRS range (1 second)
                                if time_gap < 1.0:
                                    # Apply DRS benefit
                                    drs_benefit = track_data['drs_effect'] * 0.01  # Convert to percentage
                                    lap_time *= (1 - drs_benefit)
                                    state['drs_enabled'] = True
                                else:
                                    state['drs_enabled'] = False
                        
                        # Apply random variation (smaller for more controlled simulation)
                        lap_time *= np.random.normal(1, 0.005)  # 0.5% standard deviation
                        
                        # Update total time and laps
                        results.at[idx, 'TotalTime'] += lap_time
                        results.at[idx, 'LapsCompleted'] = current_lap
                        
                        # Update tire age
                        strategy['current_tire_age'] += 1
            
            # Simulate race incidents with more realistic reliability
            if current_lap > 1:
                for idx, row in results.iterrows():
                    if row['Status'] == 'Running':
                        # Get team and driver reliability factors
                        team = row['Team']
                        driver = row['Driver']
                        
                        team_factor = team_reliability.get(team, 0.85)  # Default if team not found
                        driver_factor = driver_skill.get(driver, 0.85)  # Default if driver not found
                        
                        # Calculate incident probability - multiplying by (1-reliability) makes reliable teams have fewer incidents
                        incident_prob = base_incident_prob * (1 - team_factor) * (1 - driver_factor) * 10
                        
                        # Positions 1-3 have lower DNF chance, back markers slightly higher
                        position_factor = 1.0
                        if int(row['Position']) <= 3:
                            position_factor = 0.7  # 30% less likely for top 3
                        elif int(row['Position']) >= 15:
                            position_factor = 1.2  # 20% more likely for back markers
                        
                        incident_prob *= position_factor
                        
                        # First few laps have slightly higher incident chance
                        if current_lap < 5:
                            incident_prob *= 1.5
                        
                        # Increase incidents in wet conditions
                        if is_wet_lap:
                            incident_prob *= (1 + rain_intensity)
                        
                        # Increase incidents on high tire wear
                        strategy = tire_strategies[idx]
                        if strategy['current_tire_age'] > 35:
                            incident_prob *= (1 + (strategy['current_tire_age'] - 35) * 0.03)
                        
                        # Simulate incidents
                        if np.random.random() < incident_prob:
                            # Determine incident type - much less likely to be DNF
                            incident_type = np.random.choice(['DNF', 'Damage', 'Spin'], p=[0.25, 0.35, 0.4])
                            
                            if incident_type == 'DNF':
                                results.at[idx, 'Status'] = 'DNF'
                                results.at[idx, 'LapsCompleted'] = current_lap
                            elif incident_type == 'Damage':
                                # Add time penalty
                                results.at[idx, 'TotalTime'] += np.random.uniform(10.0, 30.0)
                            elif incident_type == 'Spin':
                                # Add smaller time penalty
                                results.at[idx, 'TotalTime'] += np.random.uniform(3.0, 8.0)
            
            # More realistic safety car and virtual safety car
            if current_lap % 15 == 0 and np.random.random() < 0.3:  # Lower probability of safety car
                # Check if any car DNF'd on this lap
                dnf_this_lap = False
                for idx, row in results.iterrows():
                    if row['Status'] == 'DNF' and row['LapsCompleted'] == current_lap:
                        dnf_this_lap = True
                        break
                
                # Higher chance of safety car if there was a DNF
                if dnf_this_lap and np.random.random() < 0.7:
                    safety_car_laps.append(current_lap)
                    # Safety car lasts 3-5 laps
                    sc_duration = np.random.randint(3, 6)
                    for sc_lap in range(current_lap, min(current_lap + sc_duration, total_laps + 1)):
                        safety_car_laps.append(sc_lap)
                    
                    # Add safety car time to all running cars
                    for jdx in results.index:
                        if results.at[jdx, 'Status'] == 'Running':
                            # Safety car bunches up the field, so everyone loses time but gaps are reduced
                            results.at[jdx, 'TotalTime'] += 5.0  # Small addition to all cars
                # Virtual safety car is more common than full safety car
                elif np.random.random() < 0.4:
                    virtual_safety_car_laps.append(current_lap)
                    # VSC lasts 1-3 laps
                    vsc_duration = np.random.randint(1, 4)
                    for vsc_lap in range(current_lap, min(current_lap + vsc_duration, total_laps + 1)):
                        virtual_safety_car_laps.append(vsc_lap)
                    
                    # Add VSC time but smaller effect than full safety car
                    for jdx in results.index:
                        if results.at[jdx, 'Status'] == 'Running':
                            results.at[jdx, 'TotalTime'] += 3.0
            
            # Update positions based on total time
            running_cars = results[results['Status'] == 'Running']
            if not running_cars.empty:
                positions = running_cars['TotalTime'].rank().astype(int)
                for idx in running_cars.index:
                    results.at[idx, 'Position'] = positions[idx]
            
            current_lap += 1
        
        # Calculate points based on finishing positions
        points_map = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
            6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }
        
        # Award fastest lap point (randomly among top 10)
        top_10_finishers = results[(results['Status'] == 'Running') & (results['Position'] <= 10)]
        if not top_10_finishers.empty:
            fastest_lap_idx = top_10_finishers.sample(1).index[0]
            
            # Award points for finishing position
            for idx, row in results.iterrows():
                if row['Status'] == 'Running':
                    position = int(row['Position'])
                    results.at[idx, 'Points'] = points_map.get(position, 0)
                    
                    # Add fastest lap point
                    if idx == fastest_lap_idx:
                        results.at[idx, 'Points'] += 1
                        results.at[idx, 'FastestLap'] = True
                    else:
                        results.at[idx, 'FastestLap'] = False
        
        # Calculate gaps to winner
        min_time = results[results['Status'] == 'Running']['TotalTime'].min()
        for idx, row in results.iterrows():
            if row['Status'] == 'Running':
                results.at[idx, 'Gap'] = row['TotalTime'] - min_time
        
        # Record start positions for position change visualization
        results['StartPosition'] = results['GridPosition']
        
        # Sort by position
        results = results.sort_values('Position')
        
        return results

    def _format_time_gap(self, gap_seconds):
        """Format time gap in a human-readable format."""
        if gap_seconds == 0:
            return ""
        elif gap_seconds < 60:
            return f"+{gap_seconds:.3f}"
        else:
            minutes = int(gap_seconds // 60)
            seconds = gap_seconds % 60
            return f"+{minutes}:{seconds:06.3f}"

    def format_race_results(self, race_results):
        """
        Format race results for pretty printing

        Args:
            race_results: DataFrame with race results

        Returns:
            DataFrame with formatted race results
        """
        try:
            if race_results is None or race_results.empty:
                self.logger.warning("No race results to format")
                return None

            # Create a formatted results DataFrame
            formatted_results = pd.DataFrame(index=race_results.index)
            
            # Fill NaN values before formatting
            race_results = race_results.fillna({
                'Position': 0,
                'FinishTime': 0.0,
                'DriverId': 'UNK',
                'FullName': 'Unknown',
                'TeamName': 'Unknown',
                'GridPosition': 0,
                'Status': 'DNF'
            })
            
            # Convert to appropriate types
            formatted_results['Position'] = race_results['Position'].astype(int)
            formatted_results['FinishTime'] = race_results['FinishTime'].astype(float)
            formatted_results['DriverId'] = race_results['DriverId']
            formatted_results['FullName'] = race_results['FullName']
            formatted_results['TeamName'] = race_results['TeamName']
            formatted_results['GridPosition'] = race_results['GridPosition'].astype(int)
            formatted_results['Status'] = race_results['Status']
            
            # Sort by position
            formatted_results = formatted_results.sort_values('Position')
            
            # Calculate gaps
            min_time = formatted_results[formatted_results['FinishTime'] < float('inf')]['FinishTime'].min()
            formatted_results['Gap'] = formatted_results['FinishTime'] - min_time
            
            # Calculate points
            points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
            formatted_results['Points'] = formatted_results['Position'].map(lambda x: points_map.get(x, 0))
            
            # Format time gaps
            def format_gap(gap, status):
                if status == 'DNF':
                    return 'DNF'
                if pd.isna(gap) or gap == 0:
                    return ''
                if gap == float('inf'):
                    return 'DNF'
                minutes = int(gap // 60)
                seconds = gap % 60
                if minutes > 0:
                    return f"+{minutes}:{seconds:06.3f}"
                return f"+{seconds:.3f}"
            
            formatted_results['FormattedGap'] = formatted_results.apply(
                lambda x: format_gap(x['Gap'], x['Status']), axis=1)
            
            # Calculate position changes
            formatted_results['Change'] = formatted_results['GridPosition'] - formatted_results['Position']
            formatted_results['FormattedPositionChange'] = formatted_results['Change'].apply(
                lambda x: f"+{x}" if x > 0 else str(x) if x < 0 else ""
            )
            
            return formatted_results
        
        except Exception as e:
            self.logger.error(f"Error formatting race results: {str(e)}")
            return None

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
            
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f'race_results_{timestamp}.png')
            
            # Save plot
            plt.savefig(output_file)
            plt.close()
            
            logger.info(f"Saved race results visualization to {output_file}")
            
        except Exception as e:
            logger.error(f"Error visualizing race results: {e}")
            plt.close()  # Ensure figure is closed even if error occurs

    def predict_and_visualize(self, race_features, race_info=None, output_file=None):
        """
        Predict race results and create visualizations.
        
        Args:
            race_features (pd.DataFrame): Features for race prediction
            race_info (dict, optional): Additional race information
            output_file (str, optional): Path to save visualizations
            
        Returns:
            pd.DataFrame: Race results
        """
        try:
            # Predict race results
            results = self.predict_finishing_positions(race_features)
            
            # Calculate points based on finishing positions
            points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
            results["Points"] = results["Position"].map(lambda x: points_map.get(x, 0))
            
            # Ensure StartPosition is present
            if "StartPosition" not in results.columns:
                if "GridPosition" in race_features.columns:
                    results["StartPosition"] = race_features["GridPosition"].values
                else:
                    results["StartPosition"] = race_features["Position"].values
            
            # Print race results
            print_race_results(results)
            
            if output_file:
                # Create visualizations
                plt.figure(figsize=(12, 6))
                
                # Grid vs Finish Position
                plt.subplot(1, 2, 1)
                plt.scatter(results["StartPosition"], results["Position"])
                plt.plot([1, 20], [1, 20], 'r--')  # Diagonal line
                plt.xlabel("Grid Position")
                plt.ylabel("Finish Position")
                plt.title("Grid vs Finish Position")
                plt.grid(True)
                
                # Team Performance
                plt.subplot(1, 2, 2)
                team_points = results.groupby("Team")["Points"].sum().sort_values(ascending=False)
                team_points.plot(kind="bar")
                plt.title("Team Performance")
                plt.xlabel("Team")
                plt.ylabel("Points")
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save visualizations
                plt.savefig(output_file)
                plt.close()
            
            return results
        except Exception as e:
            print(f"Error in predict_and_visualize: {str(e)}")
            raise

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

    def simulate_race(self, drivers, race_info):
        # Debug logging
        self.logger.info(f"Drivers type: {type(drivers)}")
        if isinstance(drivers, pd.DataFrame):
            self.logger.info(f"Drivers columns: {drivers.columns}")
            self.logger.info(f"Drivers index: {drivers.index}")
            self.logger.info(f"First row: {drivers.iloc[0]}")
        
        # Convert drivers to list if DataFrame
        if isinstance(drivers, pd.DataFrame):
            driver_list = drivers.index.tolist()
        else:
            driver_list = list(drivers)
        
        self.logger.info(f"Driver list: {driver_list}")
        
        # Initialize race state
        n_drivers = len(driver_list)
        positions = list(range(n_drivers))
        times = {i: 0.0 for i in positions}
        
        # Simulate each lap
        for lap in range(self.total_laps):
            for i in positions:
                # Base variation
                variation = 1.0 + (random.random() * 2 - 1) * self.track_effect
                times[i] += self.base_lap_time * variation
            
            # Update positions
            positions = sorted(positions, key=lambda x: times[x])
        
        # Create final results
        final_positions = {driver_list[i]: pos + 1 for pos, i in enumerate(positions)}
        final_times = {driver_list[i]: times[i] for i in positions}
        
        return final_positions, final_times

    def _calculate_lap_time(self, driver_id, state, race_state, race_features):
        """
        Calculate lap time for a driver based on various factors.
        
        Args:
            driver_id: Driver ID
            state: Driver's current state
            race_state: Current race state
            race_features: Race features DataFrame
            
        Returns:
            float: Lap time in seconds
        """
        # Get driver's race pace score
        race_pace = race_features.loc[race_features['DriverId'] == driver_id, 'RacePaceScore'].iloc[0]
        
        # Calculate base lap time
        lap_time = self.base_lap_time
        
        # Apply car performance effect (up to 1.5s per lap)
        performance_factor = (1 - race_pace) * 1.5
        lap_time += performance_factor
        
        # Apply tire degradation effect (up to 1.5% per lap)
        tire_age = state['tire_age']
        tire_effect = (tire_age / 20) * 0.015 * lap_time
        lap_time += tire_effect
        
        # Apply fuel load effect (starts at +0.3s, decreases each lap)
        fuel_effect = 0.3 * (self.total_laps - race_state['current_lap']) / self.total_laps
        lap_time += fuel_effect
        
        # Apply energy management effect (up to 2% variation)
        energy_effect = (1 - state['energy_management']) * 0.02 * lap_time
        lap_time += energy_effect
        
        # Apply DRS effect (1.5% faster when enabled)
        if race_state['current_lap'] > 3 and race_state['current_lap'] < self.total_laps - 1:
            lap_time *= 0.985
        
        # Apply safety car effect (35% slower)
        if race_state['safety_car']:
            lap_time *= 1.35
        
        # Apply VSC effect (25% slower)
        if race_state['vsc']:
            lap_time *= 1.25
        
        # Apply track evolution effect (up to 1% faster)
        track_evolution = race_state['track_evolution']
        lap_time *= (1 - (track_evolution - 1) * 0.01)
        
        # Add random variation (±0.2%)
        random_factor = random.uniform(0.998, 1.002)
        lap_time *= random_factor
        
        return lap_time

    def _calculate_base_lap_times(self, qualifying_data: pd.DataFrame, track_name='Bahrain') -> pd.Series:
        """Calculate base lap times for each driver based on qualifying data."""
        base_times = pd.Series(index=qualifying_data.index)
        
        # Get track-specific data
        track_data = self.track_data.get(track_name, self.default_track_data)
        
        # 2024 track-specific baseline lap times in seconds (representative of race pace)
        track_baselines = {
            'Bahrain': 93.0,           # ~1:33 lap time
            'Saudi Arabia': 90.0,      # ~1:30 lap time
            'Australia': 80.0,         # ~1:20 lap time
            'Japan': 92.0,             # ~1:32 lap time
            'China': 94.0,             # ~1:34 lap time
            'Miami': 89.0,             # ~1:29 lap time
            'Imola': 77.0,             # ~1:17 lap time
            'Monaco': 74.0,            # ~1:14 lap time
            'Canada': 75.0,            # ~1:15 lap time
            'Spain': 82.0,             # ~1:22 lap time
            'Austria': 65.0,           # ~1:05 lap time
            'Great Britain': 88.0,     # ~1:28 lap time
            'Hungary': 77.0,           # ~1:17 lap time
            'Belgium': 106.0,          # ~1:46 lap time
            'Netherlands': 72.0,       # ~1:12 lap time
            'Italy': 82.0,             # ~1:22 lap time
            'Azerbaijan': 103.0,       # ~1:43 lap time
            'Singapore': 98.0,         # ~1:38 lap time
            'United States': 94.0,     # ~1:34 lap time
            'Mexico': 78.0,            # ~1:18 lap time
            'Brazil': 71.0,            # ~1:11 lap time
            'Las Vegas': 103.0,        # ~1:43 lap time
            'Qatar': 83.0,             # ~1:23 lap time
            'Abu Dhabi': 87.0          # ~1:27 lap time
        }
        
        baseline_time = track_baselines.get(track_name, 90.0)
        
        # 2024 team performance factors with track type specializations
        # Format: {'team_name': {'overall': 0.0, 'street': 0.0, 'permanent': 0.0, 'desert': 0.0}}
        # Higher values = slower (multiplier)
        team_performance = {
            'Red Bull Racing': {
                'overall': 1.000,   # Benchmark (fastest car)
                'street': 1.000,    # Strong everywhere
                'permanent': 1.000,
                'desert': 1.000,
                'high_speed': 0.998, # Slightly better on high-speed tracks
                'low_speed': 1.002   # Slightly worse on low-speed tracks
            },
            'Ferrari': {
                'overall': 1.002,    # 0.2% slower overall
                'street': 1.003,     # Not as strong on street circuits
                'permanent': 1.001,  # Better on permanent tracks
                'desert': 1.002,
                'high_speed': 1.003, # Not as strong on high-speed tracks
                'low_speed': 1.000   # Good on low-speed tracks
            },
            'McLaren': {
                'overall': 1.003,    # 0.3% slower overall
                'street': 1.004,     # Not as strong on street circuits
                'permanent': 1.002,  # Better on permanent tracks
                'desert': 1.003,
                'high_speed': 1.001, # Good on high-speed tracks
                'low_speed': 1.005   # Weaker on low-speed tracks
            },
            'Mercedes': {
                'overall': 1.005,    # 0.5% slower overall
                'street': 1.006,     # Struggles a bit on street circuits
                'permanent': 1.004,  # Better on permanent tracks
                'desert': 1.005,
                'high_speed': 1.004, # Decent on high-speed tracks
                'low_speed': 1.006   # Not as good on low-speed tracks
            },
            'Aston Martin': {
                'overall': 1.010,    # 1.0% slower overall
                'street': 1.012,     # Not as strong on street circuits
                'permanent': 1.008,  # Better on permanent tracks
                'desert': 1.010,
                'high_speed': 1.011, # Not as good on high-speed tracks
                'low_speed': 1.008   # Better on low-speed tracks
            },
            'RB': {
                'overall': 1.018,    # 1.8% slower overall
                'street': 1.016,     # Better on street circuits
                'permanent': 1.020,  # Not as strong on permanent tracks
                'desert': 1.018,
                'high_speed': 1.015, # Good on high-speed tracks
                'low_speed': 1.020   # Struggles on low-speed tracks
            },
            'Williams': {
                'overall': 1.022,    # 2.2% slower overall
                'street': 1.024,     # Not as strong on street circuits
                'permanent': 1.020,  # Better on permanent tracks
                'desert': 1.022,
                'high_speed': 1.018, # Good on high-speed tracks
                'low_speed': 1.025   # Struggles on low-speed tracks
            },
            'Haas F1 Team': {
                'overall': 1.025,    # 2.5% slower overall
                'street': 1.028,     # Not as strong on street circuits
                'permanent': 1.023,  # Better on permanent tracks
                'desert': 1.025,
                'high_speed': 1.023, # Decent on high-speed tracks
                'low_speed': 1.027   # Not as good on low-speed tracks
            },
            'Alpine': {
                'overall': 1.030,    # 3.0% slower overall
                'street': 1.032,     # Not as strong on street circuits
                'permanent': 1.028,  # Better on permanent tracks
                'desert': 1.030,
                'high_speed': 1.028, # Better on high-speed tracks
                'low_speed': 1.032   # Not as good on low-speed tracks
            },
            'Kick Sauber': {
                'overall': 1.035,    # 3.5% slower overall
                'street': 1.038,     # Not as strong on street circuits
                'permanent': 1.033,  # Better on permanent tracks
                'desert': 1.035,
                'high_speed': 1.035, # Average on high-speed tracks
                'low_speed': 1.035   # Average on low-speed tracks
            }
        }
        
        # Add aliases
        team_performance['Red Bull'] = team_performance['Red Bull Racing']
        team_performance['Haas'] = team_performance['Haas F1 Team']
        team_performance['Sauber'] = team_performance['Kick Sauber']
        team_performance['Aston Martin Aramco'] = team_performance['Aston Martin']
        team_performance['Alpine F1 Team'] = team_performance['Alpine']
        
        # Track type for performance specialization
        track_type = track_data.get('track_type', 'permanent')
        
        # Simplify track types for team performance lookup
        if 'street' in track_type:
            perf_track_type = 'street'
        elif track_type == 'desert':
            perf_track_type = 'desert'
        else:
            perf_track_type = 'permanent'
        
        # High-speed vs low-speed categorization
        high_speed_tracks = ['Monza', 'Silverstone', 'Spa', 'Saudi Arabia', 'Azerbaijan']
        low_speed_tracks = ['Monaco', 'Singapore', 'Hungary', 'Abu Dhabi']
        
        if track_name in high_speed_tracks:
            speed_type = 'high_speed'
        elif track_name in low_speed_tracks:
            speed_type = 'low_speed'
        else:
            speed_type = 'overall'  # Default to overall
        
        # Driver skill factors updated to include track specialization
        # Format: {'driver_name': {'overall': 0.0, 'street': 0.0, 'permanent': 0.0}}
        # Lower values = faster (percentage off team baseline)
        driver_skill = {
            'Max Verstappen': {
                'overall': 0.997,    # 0.3% faster than team baseline
                'street': 0.998,     # Slightly less advantage on street
                'permanent': 0.996,  # Great on permanent tracks
                'wet': 0.995,        # Exceptional in wet conditions
                'high_speed': 0.996, # Great at high-speed tracks
                'low_speed': 0.998   # Good but not as exceptional on low-speed tracks
            },
            'Lewis Hamilton': {
                'overall': 0.998,    # 0.2% faster than team baseline
                'street': 0.997,     # Better on street circuits
                'permanent': 0.998,  # Good on permanent tracks
                'wet': 0.996,        # Excellent in wet conditions
                'high_speed': 0.998, # Good at high-speed tracks
                'low_speed': 0.997   # Better at low-speed technical tracks
            },
            'Fernando Alonso': {
                'overall': 0.998,    # 0.2% faster than team baseline
                'street': 0.996,     # Exceptional on street circuits
                'permanent': 0.999,  # Good on permanent tracks
                'wet': 0.996,        # Excellent in wet conditions
                'high_speed': 0.999, # Good at high-speed tracks
                'low_speed': 0.996   # Exceptional at low-speed technical tracks
            },
            'Charles Leclerc': {
                'overall': 0.998,    # 0.2% faster than team baseline
                'street': 0.996,     # Exceptional on street circuits (Monaco specialist)
                'permanent': 0.999,  # Good on permanent tracks
                'wet': 0.999,        # Average in wet conditions
                'high_speed': 0.998, # Good at high-speed tracks
                'low_speed': 0.997   # Strong at low-speed technical tracks
            },
            'Lando Norris': {
                'overall': 0.998,    # 0.2% faster than team baseline
                'street': 0.998,     # Good on street circuits
                'permanent': 0.997,  # Better on permanent tracks
                'wet': 0.997,        # Good in wet conditions
                'high_speed': 0.997, # Strong at high-speed tracks
                'low_speed': 0.999   # Less strong at low-speed tracks
            },
            'Carlos Sainz': {
                'overall': 0.999,    # 0.1% faster than team baseline
                'street': 0.998,     # Good on street circuits
                'permanent': 0.999,  # Average on permanent tracks
                'wet': 0.998,        # Good in wet conditions
                'high_speed': 0.999, # Average at high-speed tracks
                'low_speed': 0.998   # Better at low-speed tracks
            },
            'George Russell': {
                'overall': 0.999,    # 0.1% faster than team baseline
                'street': 0.999,     # Average on street circuits
                'permanent': 0.998,  # Better on permanent tracks
                'wet': 0.998,        # Good in wet conditions
                'high_speed': 0.998, # Good at high-speed tracks
                'low_speed': 0.999   # Average at low-speed tracks
            },
            'Sergio Perez': {
                'overall': 1.002,    # 0.2% slower than team baseline
                'street': 0.999,     # Strong on street circuits
                'permanent': 1.004,  # Weaker on permanent tracks
                'wet': 1.003,        # Struggles in wet conditions
                'high_speed': 1.003, # Not as strong at high-speed tracks
                'low_speed': 0.999   # Good at low-speed tracks
            }
        }
        
        # Default driver skills for missing drivers
        default_driver_skill = {
            'overall': 1.000,     # Team baseline
            'street': 1.000,      # Team baseline
            'permanent': 1.000,   # Team baseline
            'wet': 1.000,         # Team baseline
            'high_speed': 1.000,  # Team baseline
            'low_speed': 1.000    # Team baseline
        }
        
        # Check if qualifying time columns exist
        has_q3 = 'Q3_Time' in qualifying_data.columns
        has_q1 = 'Q1_Time' in qualifying_data.columns
        
        # Determine gap between qualifying and race pace (typically 3-5% slower in race)
        quali_to_race_factor = 1.04  # 4% slower in race than qualifying
        
        # Calculate base times
        for idx in qualifying_data.index:
            # Get driver and team info
            driver = qualifying_data.loc[idx, 'Driver'] if 'Driver' in qualifying_data.columns else 'Unknown'
            team = qualifying_data.loc[idx, 'Team'] if 'Team' in qualifying_data.columns else 'Unknown'
            
            # Get team-specific performance factor based on track type
            team_factor = 1.02  # Default 2% off pace if team unknown
            if team in team_performance:
                # Get track-specific performance if available, otherwise use overall
                team_data = team_performance[team]
                if perf_track_type in team_data:
                    team_track_factor = team_data[perf_track_type]
                else:
                    team_track_factor = team_data['overall']
                
                # Apply speed type specialization if available
                if speed_type in team_data:
                    team_speed_factor = team_data[speed_type]
                else:
                    team_speed_factor = team_data['overall']
                
                # Combined factor (weighted average: 70% track type, 30% speed type)
                team_factor = (team_track_factor * 0.7) + (team_speed_factor * 0.3)
            
            # Get driver-specific performance factor
            driver_factor = 1.00  # Default to neutral if driver unknown
            if driver in driver_skill:
                # Get track-specific performance if available, otherwise use overall
                driver_data = driver_skill[driver]
                if perf_track_type in driver_data:
                    driver_track_factor = driver_data[perf_track_type]
                else:
                    driver_track_factor = driver_data['overall']
                
                # Apply speed type specialization if available
                if speed_type in driver_data:
                    driver_speed_factor = driver_data[speed_type]
                else:
                    driver_speed_factor = driver_data['overall']
                
                # Combined factor (weighted average: 70% track type, 30% speed type)
                driver_factor = (driver_track_factor * 0.7) + (driver_speed_factor * 0.3)
            else:
                # Default driver factors
                driver_factor = default_driver_skill['overall']
            
            # Get qualifying position for calculation
            grid_pos = qualifying_data.loc[idx, 'Grid_Position'] if 'Grid_Position' in qualifying_data.columns else 10
            
            # If we have qualifying time data, use it as baseline
            if (has_q3 or has_q1) and (
                (has_q3 and idx in qualifying_data.index and pd.notna(qualifying_data.loc[idx, 'Q3_Time'])) or
                (has_q1 and idx in qualifying_data.index and pd.notna(qualifying_data.loc[idx, 'Q1_Time']))
            ):
                # Prefer Q3 > Q2 > Q1 times
                if has_q3 and idx in qualifying_data.index and pd.notna(qualifying_data.loc[idx, 'Q3_Time']):
                    q_time = qualifying_data.loc[idx, 'Q3_Time']
                elif 'Q2_Time' in qualifying_data.columns and idx in qualifying_data.index and pd.notna(qualifying_data.loc[idx, 'Q2_Time']):
                    q_time = qualifying_data.loc[idx, 'Q2_Time']
                else:
                    q_time = qualifying_data.loc[idx, 'Q1_Time']
                
                # Convert to race pace with some randomness
                random_factor = np.random.normal(1.0, 0.002)  # Small random variation (0.2%)
                
                # Apply track difficulty factor for qualifying to race conversion
                # Tracks where qualifying pace is less representative of race pace
                high_disparity_tracks = ['Monaco', 'Singapore']
                if track_name in high_disparity_tracks:
                    quali_to_race_factor *= 1.02  # Extra 2% gap between quali and race
                
                base_times[idx] = q_time * quali_to_race_factor * random_factor
            else:
                # No qualifying data, use baseline approach based on track, team and driver
                # Calculate time with diminishing gap based on position
                # Top teams have smaller gaps, back markers have larger gaps
                position_factor = 1.0 + (0.002 * grid_pos)  # 0.2% penalty per position
                
                # Some randomness in the baseline
                random_factor = np.random.normal(1.0, 0.003)  # 0.3% standard deviation
                
                # Calculate final time
                base_times[idx] = baseline_time * team_factor * driver_factor * position_factor * random_factor
        
        # Apply fuel load effect (heavier at start of race)
        for idx in base_times.index:
            # First stint is typically 0.5-1.0% slower due to full fuel
            base_times[idx] *= 1.007  # 0.7% penalty for full fuel
        
        return base_times

    def _calculate_tire_degradation(self, qualifying_data: pd.DataFrame, tire_deg_factor: float) -> pd.Series:
        """Calculate tire degradation rates for each driver based on qualifying data and historical performance.
        
        Args:
            qualifying_data: DataFrame with qualifying data
            tire_deg_factor: Track-specific tire degradation factor (higher = more degradation)
        """
        degradation = pd.Series(index=qualifying_data.index, data=0.001)  # Default minimal degradation
        
        # 2024 team-specific tire degradation factors (higher values = more degradation)
        team_tire_factors = {
            'Red Bull Racing': 0.70,  # Good tire management
            'Ferrari': 0.80,  # Has shown some tire issues
            'Mercedes': 0.75,  # Better tire management
            'McLaren': 0.72,  # Improved tire management
            'Aston Martin': 0.82,  # Struggles with tire life sometimes
            'Alpine': 0.85,  # Often has degradation issues
            'Williams': 0.80,  # Mid-range tire management
            'RB': 0.78,  # Decent tire performance
            'Kick Sauber': 0.87,  # Struggles with long runs
            'Haas F1 Team': 0.85,  # Known for tire management issues
            # Aliases
            'Red Bull': 0.70,
            'Haas': 0.85,
            'Aston Martin Aramco': 0.82,
            'Alpine F1 Team': 0.85,
            'Sauber': 0.87
        }
        
        # Driver-specific tire management factors (higher means better management)
        driver_tire_skill = {
            'Max Verstappen': 0.90,  # Excellent tire management
            'Lewis Hamilton': 0.92,  # Known for tire preservation
            'Fernando Alonso': 0.91,  # Veteran with excellent tire skills
            'Charles Leclerc': 0.85,  # Good but can be aggressive
            'Lando Norris': 0.87,  # Improved tire management
            'Carlos Sainz': 0.88,  # Good tire management
            'George Russell': 0.86,  # Good tire management
            'Sergio Perez': 0.89,  # Known for tire preservation
            'Oscar Piastri': 0.83,  # Still developing skills
            'Yuki Tsunoda': 0.82,  # Can be aggressive on tires
            'Daniel Ricciardo': 0.85,  # Good tire management
            'Alexander Albon': 0.84,  # Decent tire management
            'Lance Stroll': 0.83,  # Average tire management
            'Kevin Magnussen': 0.82,  # Can be hard on tires
            'Nico Hulkenberg': 0.85,  # Experienced, good management
            'Esteban Ocon': 0.84,  # Decent tire management
            'Pierre Gasly': 0.84,  # Decent tire management
            'Valtteri Bottas': 0.86,  # Good tire preserver
            'Guanyu Zhou': 0.81,  # Still developing skills
            'Logan Sargeant': 0.80   # Still developing skills
        }
        
        # Driving style factors for tire wear (higher means harder on tires)
        driving_style = {
            'Max Verstappen': 0.85,  # Smooth, easy on tires
            'Lewis Hamilton': 0.80,  # Very smooth, excellent tire management
            'Fernando Alonso': 0.82,  # Veteran smoothness
            'Charles Leclerc': 0.90,  # More aggressive, harder on tires
            'Lando Norris': 0.87,  # Fairly smooth
            'Carlos Sainz': 0.88,  # Balanced style
            'George Russell': 0.85,  # Technical, smooth style
            'Sergio Perez': 0.82,  # Gentle on tires
            'Oscar Piastri': 0.88,  # Still developing smoothness
            'Yuki Tsunoda': 0.92,  # Aggressive style
            'Daniel Ricciardo': 0.90,  # Can be hard on tires with late braking
            'Alexander Albon': 0.87,  # Fairly balanced
            'Lance Stroll': 0.89,  # Can be aggressive
            'Kevin Magnussen': 0.93,  # Aggressive, hard on tires
            'Nico Hulkenberg': 0.88,  # Balanced style
            'Esteban Ocon': 0.88,  # Balanced style
            'Pierre Gasly': 0.87,  # Fairly balanced
            'Valtteri Bottas': 0.84,  # Smooth, technical
            'Guanyu Zhou': 0.89,  # Still developing smoothness
            'Logan Sargeant': 0.91   # Still learning optimal technique
        }
        
        # Apply track-specific tire degradation multiplier
        # Higher values = more degradation
        track_tire_multiplier = tire_deg_factor  # From track data
        
        # Calculate degradation based on multiple factors
        for idx in degradation.index:
            # Base degradation rate (slightly lower than before)
            base_rate = 0.001  # 0.1% per lap is the baseline
            
            # Get driver and team
            driver = qualifying_data.loc[idx, 'Driver'] if 'Driver' in qualifying_data.columns else 'Unknown'
            team = qualifying_data.loc[idx, 'Team'] if 'Team' in qualifying_data.columns else 'Unknown'
            
            # Apply team-specific factor
            team_factor = team_tire_factors.get(team, 0.85)  # Default if team not found
            base_rate *= team_factor
            
            # Apply driver-specific skill factor
            driver_factor = driver_tire_skill.get(driver, 0.85)  # Default if driver not found
            base_rate *= (2 - driver_factor)  # Invert the scale so higher skill means lower degradation
            
            # Apply driving style factor (aggressive drivers wear tires more)
            style_factor = driving_style.get(driver, 0.88)  # Default if driver not found
            base_rate *= style_factor
            
            # Apply track-specific degradation multiplier
            base_rate *= track_tire_multiplier
            
            # Adjust based on grid position - cars at front tend to have cleaner air
            grid_pos = qualifying_data.loc[idx, 'Grid_Position'] if 'Grid_Position' in qualifying_data.columns else 10
            if grid_pos <= 5:
                base_rate *= 0.9  # 10% less degradation for front runners
            elif grid_pos >= 15:
                base_rate *= 1.1  # 10% more degradation for backmarkers due to dirty air
            
            # Probabilistic variation to account for setup differences
            random_factor = np.random.normal(1.0, 0.1)  # 10% standard deviation
            base_rate *= random_factor
            
            degradation[idx] = base_rate
        
        return degradation


def print_race_results(results_df):
    """
    Print race results in a formatted table

    Args:
        results_df: Pandas DataFrame with race results
    """
    try:
        if results_df is None or results_df.empty:
            print("No race results to display.")
            return
            
        # Fill NaN values for display
        display_df = results_df.copy()
        display_df = display_df.fillna('N/A')
        
        # Sort by position to ensure correct order
        display_df = display_df.sort_values('Position')
        
        # Default total laps for formatting
        total_laps = 58  # Default for Bahrain GP
        max_laps = display_df['LapsCompleted'].max()
        if not pd.isna(max_laps) and max_laps != 'N/A':
            total_laps = max(total_laps, int(max_laps))
        
        # Calculate position changes (if available)
        if 'GridPosition' in display_df.columns and 'Position' in display_df.columns:
            display_df['Change'] = display_df['GridPosition'] - display_df['Position']
            display_df['Change'] = display_df['Change'].map(lambda x: f"+{int(x)}" if x > 0 else str(int(x)) if x < 0 else "")
        else:
            display_df['Change'] = ""
        
        # Format gaps
        if 'Gap' in display_df.columns:
            def format_gap(gap, status):
                if status == 'DNF':
                    return 'DNF'
                if gap == 'N/A':
                    return ''
                if pd.isna(gap) or gap == 0:
                    return ''
                
                try:
                    gap_float = float(gap)
                    minutes = int(gap_float // 60)
                    seconds = gap_float % 60
                    if minutes > 0:
                        return f"+{minutes}m{seconds:.3f}s"
                    return f"+{seconds:.3f}s"
                except:
                    return str(gap)
                    
            display_df['FormattedGap'] = display_df.apply(
                lambda x: format_gap(x['Gap'], x['Status']), axis=1)
            
        # Print header
        print("\n╔═══════════════════════════════════════ RACE RESULTS ═══════════════════════════════════════╗\n")
        
        # Header
        header = f"{'POS':<4} {'NO':<5} {'DRIVER':<18} {'TEAM':<25} {'LAPS':<6} {'TIME/RETIRED':<16} {'GAP':<16} {'CHANGE':<8} {'PTS':<4}"
        print(header)
        print("═" * 100)
        
        # Print each row
        for _, row in display_df.iterrows():
            pos = int(row['Position']) if not pd.isna(row['Position']) and row['Position'] != 'N/A' else '-'
            driver_id = row['DriverId'] if not pd.isna(row['DriverId']) and row['DriverId'] != 'N/A' else '-'
            driver = row['Driver'] if not pd.isna(row['Driver']) and row['Driver'] != 'N/A' else 'Unknown Driver'
            team = row['Team'] if not pd.isna(row['Team']) and row['Team'] != 'N/A' else 'Unknown Team'
            
            laps = int(row['LapsCompleted']) if not pd.isna(row['LapsCompleted']) and row['LapsCompleted'] != 'N/A' else 0
            
            status = row['Status'] if not pd.isna(row['Status']) and row['Status'] != 'N/A' else 'Unknown'
            time_str = f"{laps}/{total_laps}" if status == 'DNF' else f"{row['TotalTime']:.3f}"
            
            gap = row['FormattedGap'] if 'FormattedGap' in row else ''
            change = row['Change'] if 'Change' in row else ''
            
            points = int(row['Points']) if not pd.isna(row['Points']) and row['Points'] != 'N/A' else 0
            
            line = f"{pos:<4} {driver_id:<5} {driver[:18]:<18} {team[:25]:<25} {laps:<6} {time_str:<16} {gap:<16} {change:<8} {points:<4}"
            print(line)
        
        print("\n╚═══════════════════════════════════════ END RESULTS ═══════════════════════════════════════╝\n")
    
    except Exception as e:
        print(f"Error formatting race results for display: {str(e)}")
        print("Raw Results:")
        print(results_df)

"""
Performance adjustment functions for F1 race predictions.
Handles dynamic weights, car development, track characteristics, and other factors.
"""

import numpy as np
from typing import Dict, Any

def calculate_dynamic_weights(race_number: int) -> Dict[str, float]:
    """
    Calculate weights based on race number in the season.
    Early season: Higher weight on qualifying
    Mid season: Higher weight on recent form
    Late season: Highest weight on recent form
    """
    if race_number <= 3:  # Early season
        return {
            'qualifying': 0.40,
            'recent': 0.35,
            'historical': 0.15,
            'form': 0.10
        }
    elif race_number <= 10:  # Mid season
        return {
            'qualifying': 0.30,
            'recent': 0.40,
            'historical': 0.20,
            'form': 0.10
        }
    else:  # Late season
        return {
            'qualifying': 0.25,
            'recent': 0.45,
            'historical': 0.20,
            'form': 0.10
        }

def add_development_factor(race_features, race_number: int) -> None:
    """
    Add development rate factor based on team resources and historical patterns.
    
    Args:
        race_features: DataFrame containing race features
        race_number: Current race number in the season
    """
    team_development_rates = {
        'Red Bull Racing': 1.002,  # Strong development rate
        'Ferrari': 1.0015,         # Solid development
        'McLaren': 1.0015,         # Good development
        'Mercedes': 1.001,         # Good development         
        'Aston Martin': 1.0008,    # Moderate development
        'Alpine F1 Team': 1.0005,  # Limited development
        'Williams': 1.0003,        # Limited development
        'Visa Cash App RB': 1.0003,# Limited development
        'Stake F1 Team': 1.0002,   # Minimal development
        'Haas F1 Team': 1.0002     # Minimal development
    }
    
    for idx, row in race_features.iterrows():
        dev_factor = team_development_rates.get(row['TeamName'], 1.0)
        # Apply development factor progressively through season
        race_features.loc[idx, 'RacePaceScore'] *= (dev_factor ** (race_number - 1))

def enhance_track_specific_performance(race_features, track_info: Dict[str, Any]) -> None:
    """
    Adjust performance based on track characteristics.
    
    Args:
        race_features: DataFrame containing race features
        track_info: Dictionary containing track information
    """
    track_type_factors = {
        'high_speed': {
            'Red Bull Racing': 1.02,  # Excellent high-speed performance
            'McLaren': 1.015,         # Strong in high-speed
            'Mercedes': 1.01,         # Good high-speed
            'Ferrari': 1.008,         # Decent high-speed
            'Aston Martin': 1.005     # Moderate high-speed
        },
        'street_circuit': {
            'Ferrari': 1.02,          # Strong street circuit performance
            'Aston Martin': 1.015,    # Good street circuit
            'Mercedes': 1.01,         # Decent street circuit
            'Red Bull Racing': 1.008, # Adaptable to street circuits
            'McLaren': 1.005          # Moderate street circuit
        },
        'technical': {
            'Mercedes': 1.02,         # Strong technical track performance
            'Ferrari': 1.015,         # Good technical
            'Red Bull Racing': 1.01,  # Adaptable to technical
            'McLaren': 1.008,         # Decent technical
            'Aston Martin': 1.005     # Moderate technical
        }
    }
    
    track_type = track_info.get('track_type', 'balanced')
    if track_type in track_type_factors:
        for idx, row in race_features.iterrows():
            factor = track_type_factors[track_type].get(row['TeamName'], 1.0)
            race_features.loc[idx, 'RacePaceScore'] *= factor

def add_weather_performance(race_features, weather_conditions: Dict[str, Any]) -> None:
    """
    Adjust performance based on weather conditions.
    
    Args:
        race_features: DataFrame containing race features
        weather_conditions: Dictionary containing weather information
    """
    wet_weather_specialists = {
        'VER': 1.03,  # Verstappen strong in all conditions
        'HAM': 1.025,   # Hamilton excellent in wet
        'ALO': 1.02,   # Alonso experienced in varying conditions
        'SAI': 1.015,  # Sainz good in wet
        'NOR': 1.015,  # Norris good in wet
        'RUS': 1.01,   # Russell adaptable
        'LEC': 1.01,   # Leclerc adaptable
        'PIA': 1.008,  # Piastri showing promise
        'ALB': 1.008   # Albon consistent
    }
    
    if weather_conditions.get('wet_track', False):
        for idx, row in race_features.iterrows():
            wet_factor = wet_weather_specialists.get(row['DriverId'], 0.98)
            race_features.loc[idx, 'RacePaceScore'] *= wet_factor

def adjust_for_sprint_format(race_features, is_sprint_weekend: bool) -> None:
    """
    Adjust predictions for sprint race weekends.
    
    Args:
        race_features: DataFrame containing race features
        is_sprint_weekend: Boolean indicating if it's a sprint weekend
    """
    if is_sprint_weekend:
        tire_management_factors = {
            'Red Bull Racing': 1.015,  # Excellent tire management
            'Ferrari': 1.01,           # Good tire management
            'Mercedes': 1.008,         # Decent tire management
            'McLaren': 1.005,          # Moderate tire management
            'Aston Martin': 1.003      # Fair tire management
        }
        
        for idx, row in race_features.iterrows():
            factor = tire_management_factors.get(row['TeamName'], 1.0)
            race_features.loc[idx, 'RacePaceScore'] *= factor

def model_teammate_dynamics(race_features) -> None:
    """
    Model intra-team performance differences.
    
    Args:
        race_features: DataFrame containing race features
    """
    for team in race_features['TeamName'].unique():
        team_drivers = race_features[race_features['TeamName'] == team]
        if len(team_drivers) == 2:
            # Calculate relative performance between teammates
            qual_diff = abs(team_drivers.iloc[0]['QualifyingPerformance'] - 
                          team_drivers.iloc[1]['QualifyingPerformance'])
            
            # Adjust race pace based on qualifying gap
            race_features.loc[team_drivers.index, 'RacePaceScore'] *= (1 + qual_diff * 0.1)

def refine_dnf_probabilities(race_features, track_info: Dict[str, Any]) -> None:
    """
    More sophisticated DNF probability calculation.
    
    Args:
        race_features: DataFrame containing race features
        track_info: Dictionary containing track information
    """
    base_dnf_prob = race_features['DNFProbability'].copy()
    
    # Adjust for track difficulty
    track_difficulty = track_info.get('difficulty', 0.5)  # 0-1 scale
    race_features['DNFProbability'] = base_dnf_prob * (1 + track_difficulty * 0.2)
    
    # 2024 team reliability factors (based on current season data)
    team_reliability = {
        'Red Bull Racing': 0.95,    # Very reliable
        'Ferrari': 0.93,            # Good reliability
        'Mercedes': 0.94,           # Good reliability
        'McLaren': 0.92,            # Decent reliability
        'Aston Martin': 0.91,       # Moderate reliability
        'Alpine F1 Team': 0.89,     # Some reliability issues
        'Williams': 0.90,           # Moderate reliability
        'Visa Cash App RB': 0.88,   # Some reliability concerns
        'Stake F1 Team': 0.87,      # More reliability issues
        'Haas F1 Team': 0.86        # Most reliability issues
    }
    
    for idx, row in race_features.iterrows():
        team_factor = team_reliability.get(row['TeamName'], 0.9)
        race_features.loc[idx, 'DNFProbability'] *= (1 - team_factor)

def apply_all_adjustments(race_features, race_number: int, track_info: Dict[str, Any], 
                         weather_conditions: Dict[str, Any], is_sprint_weekend: bool) -> None:
    """
    Apply all performance adjustments in the correct order.
    
    Args:
        race_features: DataFrame containing race features
        race_number: Current race number in the season
        track_info: Dictionary containing track information
        weather_conditions: Dictionary containing weather information
        is_sprint_weekend: Boolean indicating if it's a sprint weekend
    """
    # 1. Apply development factor (season progression)
    add_development_factor(race_features, race_number)
    
    # 2. Apply track-specific adjustments
    enhance_track_specific_performance(race_features, track_info)
    
    # 3. Apply weather adjustments
    add_weather_performance(race_features, weather_conditions)
    
    # 4. Apply sprint format adjustments
    adjust_for_sprint_format(race_features, is_sprint_weekend)
    
    # 5. Model teammate dynamics
    model_teammate_dynamics(race_features)
    
    # 6. Refine DNF probabilities
    refine_dnf_probabilities(race_features, track_info)
    
    # 7. Apply dynamic weights based on season progression
    weights = calculate_dynamic_weights(race_number)
    
    # Update projected positions with dynamic weights
    race_features['ProjectedPosition'] = (
        race_features['GridPosition'] * weights['qualifying'] +
        race_features['RecentPosition'] * weights['recent'] +
        race_features['AvgPosition'] * weights['historical'] +
        (race_features['GridPosition'] - race_features['RecentForm']) * weights['form']
    ) 
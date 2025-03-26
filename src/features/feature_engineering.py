import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
import random
import math

logger = logging.getLogger("F1Predictor.Features")


class F1FeatureEngineer:
    def __init__(self):
        """Initialize the feature engineer with enhanced parameters"""
        self.scaler = StandardScaler()
        
        # Enhanced track characteristics weights with more granular factors
        self.track_weights = {
            'high_speed': {
                'straight_speed': 0.65,
                'corner_speed': 0.35,
                'tire_management': 0.40,
                'brake_management': 0.20,
                'overtaking_difficulty': 0.25,
                'track_evolution': 0.35
            },
            'technical': {
                'straight_speed': 0.25,
                'corner_speed': 0.75,
                'tire_management': 0.50,
                'brake_management': 0.60,
                'overtaking_difficulty': 0.45,
                'track_evolution': 0.30
            },
            'street': {
                'straight_speed': 0.35,
                'corner_speed': 0.65,
                'tire_management': 0.60,
                'brake_management': 0.70,
                'overtaking_difficulty': 0.55,
                'track_evolution': 0.45
            },
            'mixed': {
                'straight_speed': 0.50,
                'corner_speed': 0.50,
                'tire_management': 0.45,
                'brake_management': 0.45,
                'overtaking_difficulty': 0.35,
                'track_evolution': 0.40
            }
        }
        
        # Enhanced weather impact factors with more detailed conditions
        self.weather_impact = {
            'temperature': {
                'tire_performance': 0.50,
                'engine_performance': 0.40,
                'driver_performance': 0.40,
                'track_evolution': 0.30,
                'brake_performance': 0.25
            },
            'humidity': {
                'engine_performance': 0.50,
                'tire_performance': 0.40,
                'brake_performance': 0.40,
                'track_grip': 0.30,
                'driver_comfort': 0.25
            },
            'wind': {
                'straight_speed': 0.60,
                'corner_stability': 0.60,
                'aero_efficiency': 0.40,
                'tire_temperatures': 0.25
            },
            'rain': {
                'visibility': 0.45,
                'track_grip': 0.55,
                'tire_strategy': 0.50,
                'driver_skill_wet': 0.65,
                'brake_performance': 0.40
            },
            'track_temp': {
                'tire_degradation': 0.60,
                'tire_graining': 0.50,
                'grip_level': 0.55,
                'surface_evolution': 0.45
            }
        }

        # Enhanced driver characteristics with more detailed factors
        self.driver_characteristics = {
            'Verstappen': {
                'wet_performance': 1.20,
                'tire_management': 1.18,
                'race_craft': 1.22,
                'qualifying_pace': 1.20,
                'consistency': 1.20,
                'adaptability': 1.20,
                'defensive_skill': 1.18,
                'aggressive_style': 1.15,
                'recovery_ability': 1.25,
                'track_knowledge': 1.18
            },
            'Hamilton': {
                'wet_performance': 1.18,
                'tire_management': 1.18,
                'race_craft': 1.15,
                'qualifying_pace': 1.12,
                'consistency': 1.15,
                'adaptability': 1.15,
                'defensive_skill': 1.12,
                'aggressive_style': 1.10,
                'recovery_ability': 1.15,
                'track_knowledge': 1.18
            },
            'Leclerc': {
                'wet_performance': 1.10,
                'tire_management': 1.08,
                'race_craft': 1.12,
                'qualifying_pace': 1.18,
                'consistency': 1.08,
                'adaptability': 1.12,
                'defensive_skill': 1.10,
                'aggressive_style': 1.15,
                'recovery_ability': 1.10,
                'track_knowledge': 1.12
            },
            'Norris': {
                'wet_performance': 1.15,
                'tire_management': 1.12,
                'race_craft': 1.15,
                'qualifying_pace': 1.15,
                'consistency': 1.12,
                'adaptability': 1.15,
                'defensive_skill': 1.12,
                'aggressive_style': 1.12,
                'recovery_ability': 1.12,
                'track_knowledge': 1.10
            },
            'Piastri': {
                'wet_performance': 1.10,
                'tire_management': 1.10,
                'race_craft': 1.12,
                'qualifying_pace': 1.15,
                'consistency': 1.10,
                'adaptability': 1.12,
                'defensive_skill': 1.08,
                'aggressive_style': 1.12,
                'recovery_ability': 1.08,
                'track_knowledge': 1.05
            },
            'Sainz': {
                'wet_performance': 1.12,
                'tire_management': 1.15,
                'race_craft': 1.15,
                'qualifying_pace': 1.12,
                'consistency': 1.15,
                'adaptability': 1.12,
                'defensive_skill': 1.15,
                'aggressive_style': 1.10,
                'recovery_ability': 1.12,
                'track_knowledge': 1.12
            },
            'Perez': {
                'wet_performance': 1.10,
                'tire_management': 1.15,
                'race_craft': 1.12,
                'qualifying_pace': 1.08,
                'consistency': 1.08,
                'adaptability': 1.10,
                'defensive_skill': 1.15,
                'aggressive_style': 1.05,
                'recovery_ability': 1.10,
                'track_knowledge': 1.12
            },
            'Russell': {
                'wet_performance': 1.12,
                'tire_management': 1.10,
                'race_craft': 1.12,
                'qualifying_pace': 1.15,
                'consistency': 1.12,
                'adaptability': 1.12,
                'defensive_skill': 1.10,
                'aggressive_style': 1.12,
                'recovery_ability': 1.10,
                'track_knowledge': 1.10
            },
            'Alonso': {
                'wet_performance': 1.15,
                'tire_management': 1.15,
                'race_craft': 1.18,
                'qualifying_pace': 1.12,
                'consistency': 1.12,
                'adaptability': 1.15,
                'defensive_skill': 1.18,
                'aggressive_style': 1.12,
                'recovery_ability': 1.15,
                'track_knowledge': 1.20
            }
        }

        # Enhanced circuit characteristics with more comprehensive properties
        self.circuit_characteristics = {
            'Bahrain': {
                'overtaking_zones': 3,
                'tire_deg_factor': 1.25,
                'track_evolution': 1.18,
                'weather_variability': 1.15,
                'straight_speed_importance': 0.65,
                'corner_speed_importance': 0.75,
                'brake_wear': 1.20,
                'engine_stress': 1.15,
                'sector_characteristics': {
                    'S1': {'type': 'technical', 'corners': 4, 'key_feature': 'heavy_braking'},
                    'S2': {'type': 'high_speed', 'corners': 6, 'key_feature': 'flowing_corners'},
                    'S3': {'type': 'mixed', 'corners': 5, 'key_feature': 'traction_zones'}
                },
                'surface_abrasion': 'high',
                'altitude': 'low',
                'grip_evolution': 'medium'
            },
            'Saudi Arabia': {
                'overtaking_zones': 3,
                'tire_deg_factor': 1.15,
                'track_evolution': 1.20,
                'weather_variability': 1.10,
                'straight_speed_importance': 0.70,
                'corner_speed_importance': 0.65,
                'brake_wear': 1.15,
                'engine_stress': 1.20,
                'sector_characteristics': {
                    'S1': {'type': 'high_speed', 'corners': 7, 'key_feature': 'walls'},
                    'S2': {'type': 'technical', 'corners': 8, 'key_feature': 'precision'},
                    'S3': {'type': 'high_speed', 'corners': 5, 'key_feature': 'slipstream'}
                },
                'surface_abrasion': 'medium',
                'altitude': 'low',
                'grip_evolution': 'high'
            },
            'Australia': {
                'overtaking_zones': 2,
                'tire_deg_factor': 1.10,
                'track_evolution': 1.15,
                'weather_variability': 1.25,
                'straight_speed_importance': 0.60,
                'corner_speed_importance': 0.70,
                'brake_wear': 1.10,
                'engine_stress': 1.10,
                'sector_characteristics': {
                    'S1': {'type': 'mixed', 'corners': 5, 'key_feature': 'chicanes'},
                    'S2': {'type': 'high_speed', 'corners': 4, 'key_feature': 'flowing'},
                    'S3': {'type': 'technical', 'corners': 7, 'key_feature': 'stop_start'}
                },
                'surface_abrasion': 'medium',
                'altitude': 'low',
                'grip_evolution': 'medium'
            },
            'Japan': {
                'overtaking_zones': 2,
                'tire_deg_factor': 1.20,
                'track_evolution': 1.10,
                'weather_variability': 1.30,
                'straight_speed_importance': 0.50,
                'corner_speed_importance': 0.80,
                'brake_wear': 1.15,
                'engine_stress': 1.25,
                'sector_characteristics': {
                    'S1': {'type': 'high_speed', 'corners': 5, 'key_feature': 'esses'},
                    'S2': {'type': 'high_speed', 'corners': 3, 'key_feature': 'spoon_curve'},
                    'S3': {'type': 'mixed', 'corners': 4, 'key_feature': '130r'}
                },
                'surface_abrasion': 'medium',
                'altitude': 'low',
                'grip_evolution': 'medium'
            },
            'China': {
                'overtaking_zones': 3,
                'tire_deg_factor': 1.15,
                'track_evolution': 1.20,
                'weather_variability': 1.20,
                'straight_speed_importance': 0.60,
                'corner_speed_importance': 0.65,
                'brake_wear': 1.10,
                'engine_stress': 1.15,
                'sector_characteristics': {
                    'S1': {'type': 'mixed', 'corners': 4, 'key_feature': 'long_straight'},
                    'S2': {'type': 'technical', 'corners': 5, 'key_feature': 'snail_corner'},
                    'S3': {'type': 'high_speed', 'corners': 7, 'key_feature': 'long_right_hander'}
                },
                'surface_abrasion': 'medium',
                'altitude': 'low',
                'grip_evolution': 'high'
            },
            'Miami': {
                'overtaking_zones': 3,
                'tire_deg_factor': 1.20,
                'track_evolution': 1.25,
                'weather_variability': 1.20,
                'straight_speed_importance': 0.65,
                'corner_speed_importance': 0.60,
                'brake_wear': 1.20,
                'engine_stress': 1.15,
                'sector_characteristics': {
                    'S1': {'type': 'high_speed', 'corners': 4, 'key_feature': 'long_straight'},
                    'S2': {'type': 'technical', 'corners': 6, 'key_feature': 'twisty_complex'},
                    'S3': {'type': 'mixed', 'corners': 5, 'key_feature': 'stadium_section'}
                },
                'surface_abrasion': 'high',
                'altitude': 'low',
                'grip_evolution': 'very_high'
            },
            'Monaco': {
                'overtaking_zones': 1,
                'tire_deg_factor': 1.05,
                'track_evolution': 1.30,
                'weather_variability': 1.15,
                'straight_speed_importance': 0.30,
                'corner_speed_importance': 0.90,
                'brake_wear': 1.30,
                'engine_stress': 1.05,
                'sector_characteristics': {
                    'S1': {'type': 'technical', 'corners': 6, 'key_feature': 'casino'},
                    'S2': {'type': 'technical', 'corners': 5, 'key_feature': 'tunnel'},
                    'S3': {'type': 'technical', 'corners': 8, 'key_feature': 'swimming_pool'}
                },
                'surface_abrasion': 'low',
                'altitude': 'low',
                'grip_evolution': 'very_high'
            }
        }
        
        # Default characteristics for tracks not specifically defined
        self.default_circuit_characteristics = {
            'overtaking_zones': 2,
            'tire_deg_factor': 1.15,
            'track_evolution': 1.15,
            'weather_variability': 1.15,
            'straight_speed_importance': 0.55,
            'corner_speed_importance': 0.65,
            'brake_wear': 1.15,
            'engine_stress': 1.15,
            'sector_characteristics': {
                'S1': {'type': 'mixed', 'corners': 5, 'key_feature': 'generic'},
                'S2': {'type': 'mixed', 'corners': 5, 'key_feature': 'generic'},
                'S3': {'type': 'mixed', 'corners': 5, 'key_feature': 'generic'}
            },
            'surface_abrasion': 'medium',
            'altitude': 'low',
            'grip_evolution': 'medium'
        }

        # Enhanced team performance characteristics with updated 2024 data
        self.team_characteristics = {
            'Red Bull Racing': {
                'aero_efficiency': 1.18,
                'power_unit': 1.15,
                'tire_management': 1.15,
                'race_pace': 1.20,
                'qualifying_pace': 1.18,
                'development_rate': 1.15,
                'high_speed': 1.18,
                'low_speed': 1.15,
                'wet_performance': 1.18,
                'reliability': 1.15,
                'technical_tracks': 1.15,
                'street_circuits': 1.18,
                'strategy_execution': 1.20,
                'setup_optimization': 1.18,
                'race_start': 1.15
            },
            'Ferrari': {
                'aero_efficiency': 1.15,
                'power_unit': 1.15,
                'tire_management': 1.12,
                'race_pace': 1.15,
                'qualifying_pace': 1.18,
                'development_rate': 1.12,
                'high_speed': 1.15,
                'low_speed': 1.12,
                'wet_performance': 1.12,
                'reliability': 1.12,
                'technical_tracks': 1.15,
                'street_circuits': 1.15,
                'strategy_execution': 1.10,
                'setup_optimization': 1.15,
                'race_start': 1.15
            },
            'McLaren': {
                'aero_efficiency': 1.15,
                'power_unit': 1.12,
                'tire_management': 1.10,
                'race_pace': 1.12,
                'qualifying_pace': 1.15,
                'development_rate': 1.15,
                'high_speed': 1.15,
                'low_speed': 1.12,
                'wet_performance': 1.12,
                'reliability': 1.15,
                'technical_tracks': 1.12,
                'street_circuits': 1.10,
                'strategy_execution': 1.15,
                'setup_optimization': 1.12,
                'race_start': 1.12
            },
            'Mercedes': {
                'aero_efficiency': 1.12,
                'power_unit': 1.15,
                'tire_management': 1.12,
                'race_pace': 1.12,
                'qualifying_pace': 1.12,
                'development_rate': 1.12,
                'high_speed': 1.12,
                'low_speed': 1.15,
                'wet_performance': 1.12,
                'reliability': 1.15,
                'technical_tracks': 1.15,
                'street_circuits': 1.12,
                'strategy_execution': 1.15,
                'setup_optimization': 1.15,
                'race_start': 1.12
            },
            'Aston Martin': {
                'aero_efficiency': 1.10,
                'power_unit': 1.12,
                'tire_management': 1.12,
                'race_pace': 1.10,
                'qualifying_pace': 1.08,
                'development_rate': 1.08,
                'high_speed': 1.08,
                'low_speed': 1.12,
                'wet_performance': 1.10,
                'reliability': 1.12,
                'technical_tracks': 1.12,
                'street_circuits': 1.10,
                'strategy_execution': 1.10,
                'setup_optimization': 1.10,
                'race_start': 1.08
            },
            'Visa Cash App Racing Bulls F1 Team': {
                'aero_efficiency': 1.08,
                'power_unit': 1.15,
                'tire_management': 1.08,
                'race_pace': 1.08,
                'qualifying_pace': 1.08,
                'development_rate': 1.08,
                'high_speed': 1.10,
                'low_speed': 1.08,
                'wet_performance': 1.08,
                'reliability': 1.10,
                'technical_tracks': 1.08,
                'street_circuits': 1.10,
                'strategy_execution': 1.08,
                'setup_optimization': 1.08,
                'race_start': 1.10
            },
            'Alpine': {
                'aero_efficiency': 1.05,
                'power_unit': 1.05,
                'tire_management': 1.08,
                'race_pace': 1.05,
                'qualifying_pace': 1.05,
                'development_rate': 1.05,
                'high_speed': 1.05,
                'low_speed': 1.08,
                'wet_performance': 1.08,
                'reliability': 1.05,
                'technical_tracks': 1.08,
                'street_circuits': 1.05,
                'strategy_execution': 1.08,
                'setup_optimization': 1.05,
                'race_start': 1.05
            },
            'Haas F1 Team': {
                'aero_efficiency': 1.05,
                'power_unit': 1.12,
                'tire_management': 1.05,
                'race_pace': 1.05,
                'qualifying_pace': 1.08,
                'development_rate': 1.05,
                'high_speed': 1.08,
                'low_speed': 1.05,
                'wet_performance': 1.05,
                'reliability': 1.08,
                'technical_tracks': 1.05,
                'street_circuits': 1.05,
                'strategy_execution': 1.05,
                'setup_optimization': 1.05,
                'race_start': 1.05
            },
            'Williams': {
                'aero_efficiency': 1.05,
                'power_unit': 1.12,
                'tire_management': 1.05,
                'race_pace': 1.05,
                'qualifying_pace': 1.05,
                'development_rate': 1.08,
                'high_speed': 1.08,
                'low_speed': 1.05,
                'wet_performance': 1.05,
                'reliability': 1.05,
                'technical_tracks': 1.05,
                'street_circuits': 1.05,
                'strategy_execution': 1.05,
                'setup_optimization': 1.05,
                'race_start': 1.05
            },
            'Kick Sauber': {
                'aero_efficiency': 1.02,
                'power_unit': 1.05,
                'tire_management': 1.05,
                'race_pace': 1.02,
                'qualifying_pace': 1.02,
                'development_rate': 1.02,
                'high_speed': 1.02,
                'low_speed': 1.05,
                'wet_performance': 1.05,
                'reliability': 1.05,
                'technical_tracks': 1.05,
                'street_circuits': 1.02,
                'strategy_execution': 1.05,
                'setup_optimization': 1.02,
                'race_start': 1.02
            }
        }
        
        # Default team characteristics for teams not specifically defined
        self.default_team_characteristics = {
            'aero_efficiency': 1.0,
            'power_unit': 1.0,
            'tire_management': 1.0,
            'race_pace': 1.0,
            'qualifying_pace': 1.0,
            'development_rate': 1.0,
            'high_speed': 1.0,
            'low_speed': 1.0,
            'wet_performance': 1.0,
            'reliability': 1.0,
            'technical_tracks': 1.0,
            'street_circuits': 1.0,
            'strategy_execution': 1.0,
            'setup_optimization': 1.0,
            'race_start': 1.0
        }
        
        # Enhanced track evolution characteristics
        self.track_evolution = {
            'high_grip': {
                'initial_grip': 0.85,
                'evolution_rate': 0.02,
                'rubber_buildup': 0.015,
                'temperature_sensitivity': 0.012,
                'rain_impact': -0.03
            },
            'medium_grip': {
                'initial_grip': 0.75,
                'evolution_rate': 0.025,
                'rubber_buildup': 0.02,
                'temperature_sensitivity': 0.015,
                'rain_impact': -0.04
            },
            'low_grip': {
                'initial_grip': 0.65,
                'evolution_rate': 0.03,
                'rubber_buildup': 0.025,
                'temperature_sensitivity': 0.02,
                'rain_impact': -0.05
            }
        }
        
        # Enhanced tire degradation modeling
        self.tire_characteristics = {
            'soft': {
                'initial_grip': 1.15,
                'deg_rate': 0.012,
                'optimal_window': {'min': 85, 'max': 105},
                'working_range': {'min': 75, 'max': 115},
                'graining_risk': 0.15,
                'wet_performance': 0.85
            },
            'medium': {
                'initial_grip': 1.08,
                'deg_rate': 0.008,
                'optimal_window': {'min': 90, 'max': 110},
                'working_range': {'min': 80, 'max': 120},
                'graining_risk': 0.10,
                'wet_performance': 0.90
            },
            'hard': {
                'initial_grip': 1.02,
                'deg_rate': 0.005,
                'optimal_window': {'min': 95, 'max': 115},
                'working_range': {'min': 85, 'max': 125},
                'graining_risk': 0.05,
                'wet_performance': 0.95
            }
        }

        # Enhanced Red Bull specific characteristics
        self.red_bull_characteristics = {
            'race_pace_boost': {
                'qualifying_to_race': 1.25,  # Significant improvement from qualifying to race
                'tire_management': 1.20,
                'strategy_execution': 1.18,
                'development_curve': 1.15
            },
            'verstappen_specific': {
                'race_craft_multiplier': 1.30,
                'tire_management_boost': 1.25,
                'qualifying_recovery': 1.28,
                'wet_condition_boost': 1.25,
                'track_evolution_adaptation': 1.20
            },
            'perez_specific': {
                'race_craft_multiplier': 1.15,
                'tire_management_boost': 1.18,
                'qualifying_recovery': 1.15,
                'wet_condition_boost': 1.12,
                'track_evolution_adaptation': 1.15
            }
        }
        
        # Enhanced track evolution modeling
        self.track_evolution_detailed = {
            'grip_progression': {
                'initial_phase': {
                    'duration': 0.2,  # First 20% of race
                    'grip_increase': 0.015,
                    'line_sensitivity': 1.2
                },
                'middle_phase': {
                    'duration': 0.6,  # Middle 60% of race
                    'grip_increase': 0.008,
                    'line_sensitivity': 1.1
                },
                'final_phase': {
                    'duration': 0.2,  # Final 20% of race
                    'grip_increase': 0.005,
                    'line_sensitivity': 1.05
                }
            },
            'temperature_impact': {
                'track_temp_optimal': 35,
                'air_temp_optimal': 25,
                'temp_sensitivity': 0.008,
                'temp_recovery_rate': 0.005
            },
            'rubber_evolution': {
                'buildup_rate': 0.012,
                'cleaning_effect': -0.02,
                'optimal_level': 0.85
            }
        }
        
        # Enhanced tire performance modeling
        self.tire_performance_detailed = {
            'compound_characteristics': {
                'soft': {
                    'peak_grip': 1.18,
                    'optimal_window': {'min': 85, 'max': 105},
                    'working_range': {'min': 75, 'max': 115},
                    'deg_phases': {
                        'initial': {'rate': 0.015, 'duration': 0.2},
                        'steady': {'rate': 0.010, 'duration': 0.6},
                        'cliff': {'rate': 0.020, 'duration': 0.2}
                    }
                },
                'medium': {
                    'peak_grip': 1.12,
                    'optimal_window': {'min': 90, 'max': 110},
                    'working_range': {'min': 80, 'max': 120},
                    'deg_phases': {
                        'initial': {'rate': 0.010, 'duration': 0.2},
                        'steady': {'rate': 0.007, 'duration': 0.6},
                        'cliff': {'rate': 0.015, 'duration': 0.2}
                    }
                },
                'hard': {
                    'peak_grip': 1.08,
                    'optimal_window': {'min': 95, 'max': 115},
                    'working_range': {'min': 85, 'max': 125},
                    'deg_phases': {
                        'initial': {'rate': 0.007, 'duration': 0.2},
                        'steady': {'rate': 0.005, 'duration': 0.6},
                        'cliff': {'rate': 0.010, 'duration': 0.2}
                    }
                }
            },
            'track_condition_impact': {
                'green': {'grip_factor': 0.92, 'deg_multiplier': 1.15},
                'rubbered': {'grip_factor': 1.05, 'deg_multiplier': 0.95},
                'optimal': {'grip_factor': 1.08, 'deg_multiplier': 0.90}
            }
        }

    def calculate_team_performance(self, historical_data, current_data):
        """Calculate team performance metrics based on historical and current data."""
        try:
            if historical_data is None or historical_data.empty:
                logger.warning("No historical data available")
                return pd.DataFrame()

            team_metrics = []

            # Enhanced team-specific characteristics
            team_characteristics = {
                "McLaren": {
                    "quali_strength": 1.12,      # Strong qualifying pace
                    "race_consistency": 1.10,    # Very consistent in races
                    "development_rate": 1.15,    # Strong development trajectory
                    "tire_management": 1.08,     # Good tire management
                    "reliability": 0.98          # Slight reliability concerns
                },
                "Red Bull Racing": {
                    "quali_strength": 1.10,      # Very strong qualifying
                    "race_consistency": 1.15,    # Excellent race consistency
                    "development_rate": 1.12,    # Strong development
                    "tire_management": 1.12,     # Excellent tire management
                    "reliability": 1.05          # High reliability
                },
                "Ferrari": {
                    "quali_strength": 1.15,      # Exceptional qualifying pace
                    "race_consistency": 0.95,    # Some race consistency issues
                    "development_rate": 1.08,    # Good development rate
                    "tire_management": 0.95,     # Tire management challenges
                    "reliability": 0.92          # Reliability concerns
                },
                "Mercedes": {
                    "quali_strength": 1.05,      # Good qualifying pace
                    "race_consistency": 1.08,    # Strong race consistency
                    "development_rate": 1.10,    # Good development rate
                    "tire_management": 1.05,     # Good tire management
                    "reliability": 1.02          # Reliable
                }
            }

            team_results = historical_data.groupby("TeamName")

            for team_name, results in team_results:
                if current_data.get("data_processor"):
                    team_name = current_data["data_processor"].standardize_team_name(team_name)

                # Get team-specific characteristics
                team_chars = team_characteristics.get(team_name, {
                    "quali_strength": 1.0,
                    "race_consistency": 1.0,
                    "development_rate": 1.0,
                    "tire_management": 1.0,
                    "reliability": 1.0
                })

                # Enhanced performance metrics
                finished_races = results[results["Status"] == "Finished"]
                
                # Position-based metrics with team-specific adjustments
                avg_position = finished_races["Position"].astype(float).mean()
                position_consistency = finished_races["Position"].std() if len(finished_races) > 1 else 5.0
                
                # Adjust position metrics with team characteristics
                avg_position = avg_position / team_chars["race_consistency"]
                position_consistency = position_consistency / team_chars["race_consistency"]
                
                # Reliability metrics with team-specific adjustments
                total_races = len(results)
                finished_count = len(finished_races)
                base_reliability = finished_count / total_races if total_races > 0 else 0.85
                reliability = base_reliability * team_chars["reliability"]
                
                # DNF analysis with team-specific consideration
                dnf_races = results[results["Status"] != "Finished"]
                mechanical_dnfs = dnf_races[dnf_races["Status"].str.contains("Technical|Engine|Gearbox|Hydraulics", na=False)].shape[0]
                mechanical_reliability = (1 - (mechanical_dnfs / total_races if total_races > 0 else 0.1)) * team_chars["reliability"]
                
                # Development trend with team-specific rate
                if len(finished_races) >= 4:
                    recent_races = finished_races.iloc[-2:]["Position"].mean()
                    earlier_races = finished_races.iloc[:-2]["Position"].mean()
                    development_trend = ((earlier_races - recent_races) / 20) * team_chars["development_rate"]
                else:
                    development_trend = 0
                
                # Current season performance
                team_points = 0
                if "constructor_standings" in current_data and not current_data["constructor_standings"].empty:
                    team_row = current_data["constructor_standings"][current_data["constructor_standings"]["TeamName"] == team_name]
                    if not team_row.empty:
                        team_points = team_row.iloc[0]["Points"]

                # Calculate comprehensive team strength with characteristics
                base_strength = (
                    (1 - avg_position / 20) * 0.3 +
                    reliability * 0.2 +
                    (1 - position_consistency / 10) * 0.15 +
                    mechanical_reliability * 0.15 +
                    development_trend * 0.1 +
                    (team_points / 100) * 0.1  # Normalized points contribution
                )

                # Apply team-specific adjustments
                team_strength = base_strength * team_chars["quali_strength"] * team_chars["tire_management"]

                team_metrics.append({
                    "TeamName": team_name,
                    "AvgPosition": avg_position,
                    "PositionConsistency": position_consistency,
                    "Reliability": reliability,
                    "MechanicalReliability": mechanical_reliability,
                    "DevelopmentTrend": development_trend,
                    "CurrentPoints": team_points,
                    "TeamStrength": team_strength
                })

            return pd.DataFrame(team_metrics)

        except Exception as e:
            logger.error(f"Error calculating team performance: {e}")
            return pd.DataFrame()

    def calculate_team_performance(self, prediction_data):
        """Enhanced team performance calculation with more detailed metrics"""
        try:
            historical_races = prediction_data["historical"]["race"]
            if historical_races.empty:
                logger.warning("No historical race data available")
                return pd.DataFrame()

            constructor_standings = prediction_data["constructor_standings"]
            if constructor_standings is None or constructor_standings.empty:
                logger.warning("No constructor standings available")
                return pd.DataFrame()

            team_metrics = []

            team_results = historical_races.groupby("TeamName")

            for team_name, results in team_results:
                if prediction_data.get("data_processor"):
                    team_name = prediction_data["data_processor"].standardize_team_name(team_name)

                # Enhanced performance metrics
                finished_races = results[results["Status"] == "Finished"]
                
                # Position-based metrics
                avg_position = finished_races["Position"].astype(float).mean()
                position_consistency = finished_races["Position"].std() if len(finished_races) > 1 else 5.0
                
                # Reliability metrics
                total_races = len(results)
                finished_count = len(finished_races)
                reliability = finished_count / total_races if total_races > 0 else 0.85
                
                # DNF analysis
                dnf_races = results[results["Status"] != "Finished"]
                mechanical_dnfs = dnf_races[dnf_races["Status"].str.contains("Technical|Engine|Gearbox|Hydraulics", na=False)].shape[0]
                mechanical_reliability = 1 - (mechanical_dnfs / total_races if total_races > 0 else 0.1)
                
                # Pace metrics
                if "MedianLapTime" in results.columns:
                    avg_pace = results["MedianLapTime"].dropna().mean()
                    pace_consistency = results["MedianLapTime"].dropna().std() if len(results["MedianLapTime"].dropna()) > 1 else 1.0
                else:
                    avg_pace = np.nan
                    pace_consistency = np.nan
                
                # Development trend (comparing recent races to earlier ones)
                if len(finished_races) >= 4:
                    recent_races = finished_races.iloc[-2:]["Position"].mean()
                    earlier_races = finished_races.iloc[:-2]["Position"].mean()
                    development_trend = (earlier_races - recent_races) / 20  # Normalized to -1 to 1
                else:
                    development_trend = 0
                
                # Current season performance
                team_points = 0
                if not constructor_standings.empty:
                    team_row = constructor_standings[constructor_standings["TeamName"] == team_name]
                    if not team_row.empty:
                        team_points = team_row.iloc[0]["Points"]

                # Calculate comprehensive team strength
                team_metrics.append({
                    "TeamName": team_name,
                    "AvgPosition": avg_position,
                    "PositionConsistency": position_consistency,
                    "Reliability": reliability,
                    "MechanicalReliability": mechanical_reliability,
                    "AvgPace": avg_pace,
                    "PaceConsistency": pace_consistency,
                    "DevelopmentTrend": development_trend,
                    "CurrentPoints": team_points
                })

            team_df = pd.DataFrame(team_metrics)

            # Fill missing values
            team_df["AvgPosition"].fillna(10, inplace=True)
            team_df["PositionConsistency"].fillna(5, inplace=True)
            team_df["Reliability"].fillna(0.85, inplace=True)
            team_df["MechanicalReliability"].fillna(0.9, inplace=True)
            team_df["AvgPace"].fillna(team_df["AvgPace"].mean(), inplace=True)
            team_df["PaceConsistency"].fillna(1.0, inplace=True)
            team_df["DevelopmentTrend"].fillna(0, inplace=True)

            # Calculate normalized metrics
            if not team_df.empty:
                # Normalize pace (lower is better)
                team_df["NormalizedPace"] = (team_df["AvgPace"] - team_df["AvgPace"].min()) / (team_df["AvgPace"].max() - team_df["AvgPace"].min()) if team_df["AvgPace"].max() != team_df["AvgPace"].min() else 0.5
                
                # Normalize consistency (lower is better)
                team_df["NormalizedConsistency"] = 1 - (team_df["PositionConsistency"] / team_df["PositionConsistency"].max())
                
                # Calculate comprehensive team strength with weighted factors
                team_df["TeamStrength"] = (
                    (1 - team_df["NormalizedPace"]) * 0.35 +  # Pace
                    team_df["Reliability"] * 0.20 +            # Overall reliability
                    team_df["MechanicalReliability"] * 0.15 +  # Mechanical reliability
                    team_df["NormalizedConsistency"] * 0.15 +  # Consistency
                    team_df["DevelopmentTrend"] * 0.15         # Recent development
                )

            return team_df
        except Exception as e:
            logger.error(f"Error calculating team performance: {e}")
            return pd.DataFrame()

    def process_driver_statistics(self, driver_data, team_metrics=None, circuit_name=None):
        """Process driver statistics for feature engineering with enhanced metrics."""
        if driver_data.empty:
            logging.warning("No driver statistics available")
            return pd.DataFrame()

        # Get team strength if available
        team_name = driver_data['Team'].iloc[0] if 'Team' in driver_data.columns else None
        team_strength = 1.0
        if team_metrics and team_name and team_name in team_metrics:
            team_strength = team_metrics[team_name]['TeamStrength']

        # Calculate recent form (last 3 races)
        recent_races = driver_data.sort_values('RoundNumber', ascending=False).head(3)
        
        # Calculate weighted recent performance
        if not recent_races.empty:
            weights = np.exp(-0.5 * np.arange(len(recent_races)))  # Stronger exponential decay
            weighted_positions = np.average(
                recent_races['Position'],
                weights=weights,
                axis=0
            )
        else:
            weighted_positions = driver_data['Position'].mean()

        # Calculate DNF rate with exponential penalty
        total_races = len(driver_data)
        finished_races = len(driver_data[driver_data['Status'] == 'Finished'])
        dnf_rate = 1 - (finished_races / total_races)
        dnf_penalty = math.exp(dnf_rate) - 1  # Exponential penalty for high DNF rates

        # Calculate qualifying performance
        quali_positions = driver_data['GridPosition'].mean()
        recent_quali = recent_races['GridPosition'].mean() if not recent_races.empty else quali_positions
        
        # Calculate qualifying vs race performance
        quali_vs_race = driver_data.apply(
            lambda x: x['GridPosition'] - x['Position'], axis=1
        ).mean()

        # Calculate race craft score with emphasis on recent races
        recent_position_changes = recent_races.apply(
            lambda x: x['GridPosition'] - x['Position'], axis=1
        )
        overall_position_changes = driver_data.apply(
            lambda x: x['GridPosition'] - x['Position'], axis=1
        )
        
        recent_race_craft = recent_position_changes.mean() if not recent_position_changes.empty else 0
        overall_race_craft = overall_position_changes.mean()
        race_craft = 0.7 * recent_race_craft + 0.3 * overall_race_craft
        
        # Calculate consistency scores
        recent_consistency = 1 / (recent_races['Position'].std() + 1) if not recent_races.empty else 0
        overall_consistency = 1 / (driver_data['Position'].std() + 1)
        consistency_score = 0.7 * recent_consistency + 0.3 * overall_consistency

        # Calculate wet weather performance if available
        wet_races = driver_data[driver_data['WetRace'] == True] if 'WetRace' in driver_data.columns else pd.DataFrame()
        wet_performance = 0
        if not wet_races.empty:
            wet_positions = wet_races['Position'].mean()
            dry_positions = driver_data[driver_data['WetRace'] == False]['Position'].mean()
            wet_performance = dry_positions - wet_positions  # Positive means better in wet

        # Calculate track-specific performance
        track_performance = 0
        if circuit_name:
            track_races = driver_data[driver_data['Circuit'] == circuit_name]
            if not track_races.empty:
                track_performance = 1 - (track_races['Position'].mean() / 20)
            else:
                track_performance = 1 - (weighted_positions / 20)

        # Calculate current form metrics
        current_points = recent_races['Points'].sum() if not recent_races.empty else 0
        current_position = recent_races['Position'].iloc[0] if not recent_races.empty else 20
        points_trend = recent_races['Points'].mean() - driver_data['Points'].mean() if not recent_races.empty else 0

        # Combine all factors into driver form
        base_form = (
            0.25 * (1 - weighted_positions / 20) +  # Recent weighted performance
            0.20 * (1 - dnf_penalty) +              # Reliability
            0.15 * consistency_score +              # Consistency
            0.15 * (race_craft / 10 + 0.5) +       # Race craft (normalized)
            0.10 * (quali_vs_race / 20 + 0.5) +    # Qualifying vs Race performance
            0.10 * (track_performance + 0.5) +      # Track-specific performance
            0.05 * (wet_performance / 20 + 0.5)     # Wet weather performance
        )

        # Apply team context
        driver_form = base_form * (0.7 + 0.3 * team_strength)  # Blend individual and team performance

        # Create feature dictionary
        features = {
            'ReliabilityScore': 1 - dnf_penalty,
            'WeightedAvgFinish': weighted_positions,
            'RecentQualiPerformance': recent_quali,
            'QualiRaceGap': quali_vs_race,
            'RaceCraftScore': race_craft,
            'ConsistencyScore': consistency_score,
            'WetPerformance': wet_performance,
            'TrackPerformance': track_performance,
            'CurrentPoints': current_points,
            'PointsTrend': points_trend,
            'CurrentPosition': current_position,
            'TeamStrength': team_strength,
            'DriverForm': driver_form
        }

        return pd.DataFrame([features])

    def get_track_characteristics(self, circuit_name, historical_data):
        """Enhanced track characteristics calculation with more detailed features"""
        try:
            # Get base track type and characteristics
            track_type = self._determine_track_type(circuit_name)
            track_length = self._get_track_length(circuit_name)
            track_elevation = self._get_track_elevation(circuit_name)
            
            # Get dynamic characteristics based on historical data
            overtaking_difficulty = self._calculate_overtaking_difficulty(circuit_name, historical_data)
            tire_degradation = self._calculate_tire_degradation(circuit_name, historical_data)
            drs_effectiveness = self._calculate_drs_effectiveness(circuit_name, historical_data)
            
            # Get weather-related characteristics
            weather_conditions = self._get_weather_conditions(circuit_name)
            typical_temperature = self._get_typical_temperature(circuit_name)
            typical_humidity = self._get_typical_humidity(circuit_name)
            typical_wind = self._get_typical_wind_speed(circuit_name)
            wet_conditions = self._get_typical_wet_conditions(circuit_name)
            
            # Calculate sector characteristics
            sector_types = self._get_sector_characteristics(circuit_name)
            
            # Calculate track-specific performance factors
            track_factors = {}
            
            # Apply track type weights
            if track_type in self.track_weights:
                weights = self.track_weights[track_type]
                track_factors.update(weights)
            else:
                # Use balanced weights as default
                track_factors.update({
                    'straight_speed': 0.5,
                    'corner_speed': 0.5,
                    'tire_management': 0.4,
                    'brake_management': 0.4
                })
            
            # Adjust weights based on track characteristics
            if track_length > 5.5:  # Long track
                track_factors['straight_speed'] *= 1.2
                track_factors['tire_management'] *= 1.1
            elif track_length < 4.5:  # Short track
                track_factors['corner_speed'] *= 1.2
                track_factors['brake_management'] *= 1.1
                
            # Adjust for elevation changes
            if track_elevation > 50:  # Significant elevation changes
                track_factors['engine_performance'] = 0.6
                track_factors['brake_management'] *= 1.2
            
            # Weather impact adjustments
            weather_factors = {}
            if typical_temperature > 30:  # Hot conditions
                weather_factors.update({
                    'tire_degradation': tire_degradation * 1.3,
                    'engine_stress': 0.7,
                    'physical_demand': 0.8
                })
            elif typical_temperature < 15:  # Cold conditions
                weather_factors.update({
                    'tire_warmup': 0.7,
                    'grip_level': 0.8
                })
                
            if typical_humidity > 70:  # High humidity
                weather_factors['engine_performance'] = weather_factors.get('engine_performance', 1.0) * 0.9
            
            # Combine all characteristics
            track_characteristics = {
                'circuit_name': circuit_name,
                'track_type': track_type,
                'track_length': track_length,
                'track_elevation': track_elevation,
                'overtaking_difficulty': overtaking_difficulty,
                'tire_degradation': tire_degradation,
                'drs_effectiveness': drs_effectiveness,
                'typical_temperature': typical_temperature,
                'typical_humidity': typical_humidity,
                'typical_wind': typical_wind,
                'wet_probability': wet_conditions,
                'sector_characteristics': sector_types,
                'performance_factors': track_factors,
                'weather_impact': weather_factors
            }
            
            # Calculate overall difficulty score
            difficulty_score = (
                overtaking_difficulty * 0.3 +
                tire_degradation * 0.25 +
                (1 - drs_effectiveness) * 0.15 +
                (track_elevation / 100) * 0.15 +
                (wet_conditions) * 0.15
            )
            track_characteristics['difficulty_score'] = min(1.0, difficulty_score)
            
            return track_characteristics
            
        except Exception as e:
            logger.error(f"Error calculating track characteristics: {e}")
            return self._get_default_track_characteristics()
            
    def _get_sector_characteristics(self, circuit_name):
        """Calculate sector-specific characteristics"""
        # Define sector characteristics for known circuits
        sector_data = {
            'Bahrain': {
                'S1': {'type': 'technical', 'corners': 4, 'key_feature': 'heavy_braking'},
                'S2': {'type': 'high_speed', 'corners': 6, 'key_feature': 'flowing_corners'},
                'S3': {'type': 'mixed', 'corners': 5, 'key_feature': 'traction_zones'}
            },
            'Saudi Arabia': {
                'S1': {'type': 'high_speed', 'corners': 7, 'key_feature': 'walls'},
                'S2': {'type': 'technical', 'corners': 8, 'key_feature': 'precision'},
                'S3': {'type': 'high_speed', 'corners': 5, 'key_feature': 'slipstream'}
            },
            'Australia': {
                'S1': {'type': 'mixed', 'corners': 5, 'key_feature': 'chicanes'},
                'S2': {'type': 'high_speed', 'corners': 4, 'key_feature': 'flowing'},
                'S3': {'type': 'technical', 'corners': 7, 'key_feature': 'stop_start'}
            }
        }
        
        # Return default characteristics if circuit not known
        if circuit_name not in sector_data:
            return {
                'S1': {'type': 'mixed', 'corners': 5, 'key_feature': 'standard'},
                'S2': {'type': 'mixed', 'corners': 5, 'key_feature': 'standard'},
                'S3': {'type': 'mixed', 'corners': 5, 'key_feature': 'standard'}
            }
            
        return sector_data[circuit_name]

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
            "high_speed": ["Monza", "Silverstone", "Spa", "Baku", "Jeddah"],
            "technical": ["Monaco", "Singapore", "Hungary", "Zandvoort"],
            "street": ["Monaco", "Singapore", "Baku", "Miami", "Las Vegas"],
            "desert": ["Bahrain", "Abu Dhabi", "Saudi Arabia"],
            "balanced": ["Melbourne", "Barcelona", "Austin", "Suzuka"],
        }

        for track_type, circuits in track_types.items():
            if circuit_name in circuits:
                return track_type

        return "balanced"  # Default to balanced if unknown

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
            "Monza": 5.793,
            "Silverstone": 5.891,
            "Spa": 7.004,
            "Baku": 6.003,
            "Jeddah": 6.174,
            "Monaco": 3.337,
            "Singapore": 5.063,
            "Hungary": 4.381,
            "Zandvoort": 4.259,
            "Bahrain": 5.412,
            "Abu Dhabi": 5.554,
            "Saudi Arabia": 6.174,
            "Melbourne": 5.278,
            "Barcelona": 4.655,
            "Austin": 5.513,
            "Suzuka": 5.807,
            "Miami": 5.412,
            "Las Vegas": 6.201,
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
            "Spa": 102.5,  # Highest point at Eau Rouge
            "Austin": 133.0,  # Highest point at Turn 1
            "Silverstone": 153.0,  # Highest point at Copse
            "Monza": 35.0,  # Highest point at Parabolica
            "Baku": 28.0,  # Highest point at Turn 1
            "Jeddah": 12.0,  # Highest point at Turn 1
            "Monaco": 100.0,  # Highest point at Casino Square
            "Singapore": 30.0,  # Highest point at Turn 1
            "Hungary": 250.0,  # Highest point at Turn 4
            "Zandvoort": 19.0,  # Highest point at Tarzan
            "Bahrain": 63.0,  # Highest point at Turn 1
            "Abu Dhabi": 8.0,  # Highest point at Turn 1
            "Melbourne": 30.0,  # Highest point at Turn 1
            "Barcelona": 33.0,  # Highest point at Turn 1
            "Suzuka": 40.0,  # Highest point at Turn 1
            "Miami": 5.0,  # Highest point at Turn 1
            "Las Vegas": 0.0,  # Flat circuit
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
            circuit_data = historical_data[historical_data["CircuitName"] == circuit_name]
            if circuit_data.empty:
                return self._get_default_overtaking_difficulty(circuit_name)

            # Calculate average position changes
            position_changes = circuit_data["GridPosition"] - circuit_data["FinishPosition"]
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
            circuit_data = historical_data[historical_data["CircuitName"] == circuit_name]
            if circuit_data.empty:
                return self._get_default_tire_degradation(circuit_name)

            # Calculate average pit stops
            avg_pit_stops = circuit_data["PitStops"].mean()

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
            circuit_data = historical_data[historical_data["CircuitName"] == circuit_name]
            if circuit_data.empty:
                return self._get_default_drs_effectiveness(circuit_name)

            # Calculate average DRS overtakes
            avg_drs_overtakes = circuit_data["DRSOvertakes"].mean()

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
            "Temperature": self._get_typical_temperature(circuit_name),
            "Humidity": self._get_typical_humidity(circuit_name),
            "WindSpeed": self._get_typical_wind_speed(circuit_name),
            "IsWet": self._get_typical_wet_conditions(circuit_name),
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
            "Bahrain": 30.0,
            "Abu Dhabi": 28.0,
            "Saudi Arabia": 32.0,
            "Melbourne": 22.0,
            "Baku": 25.0,
            "Miami": 28.0,
            "Monaco": 20.0,
            "Barcelona": 24.0,
            "Monza": 25.0,
            "Singapore": 30.0,
            "Austin": 26.0,
            "Mexico City": 22.0,
            "Sao Paulo": 25.0,
            "Las Vegas": 20.0,
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
            "Bahrain": 45.0,
            "Abu Dhabi": 60.0,
            "Saudi Arabia": 40.0,
            "Melbourne": 65.0,
            "Baku": 55.0,
            "Miami": 75.0,
            "Monaco": 70.0,
            "Barcelona": 65.0,
            "Monza": 60.0,
            "Singapore": 85.0,
            "Austin": 70.0,
            "Mexico City": 65.0,
            "Sao Paulo": 75.0,
            "Las Vegas": 30.0,
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
            "Bahrain": 15.0,
            "Abu Dhabi": 12.0,
            "Saudi Arabia": 10.0,
            "Melbourne": 20.0,
            "Baku": 15.0,
            "Miami": 18.0,
            "Monaco": 12.0,
            "Barcelona": 15.0,
            "Monza": 18.0,
            "Singapore": 10.0,
            "Austin": 15.0,
            "Mexico City": 12.0,
            "Sao Paulo": 15.0,
            "Las Vegas": 10.0,
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
        wet_circuits = ["Spa", "Silverstone", "Interlagos", "Monaco"]
        return circuit_name in wet_circuits

    def _get_default_track_characteristics(self):
        """
        Get default track characteristics when real data is unavailable

        Returns:
            dict: Default track characteristics
        """
        return {
            "CircuitName": "Unknown",
            "TrackType": "balanced",
            "TrackLength": 5.0,
            "TrackElevation": 0.0,
            "OvertakingDifficulty": 0.5,
            "TireDegradation": 0.5,
            "DRSEffectiveness": 0.5,
            "WeatherConditions": {
                "Temperature": 25.0,
                "Humidity": 60.0,
                "WindSpeed": 15.0,
                "IsWet": False,
            },
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
            "high_speed": 0.3,
            "technical": 0.8,
            "street": 0.9,
            "desert": 0.4,
            "balanced": 0.5,
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
            "high_speed": 0.7,
            "technical": 0.8,
            "street": 0.9,
            "desert": 0.6,
            "balanced": 0.5,
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
            "high_speed": 0.8,
            "technical": 0.4,
            "street": 0.3,
            "desert": 0.6,
            "balanced": 0.5,
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
                pole_time = qualifying_processed.loc[
                    qualifying_processed["Position"] == 1, "BestTime"
                ].values[0]
                qualifying_processed["GapToPole"] = qualifying_processed["BestTime"] - pole_time

                # Calculate normalized qualifying performance (0-1 where 1 is best)
                max_gap = qualifying_processed["GapToPole"].max()
                if max_gap > 0:
                    qualifying_processed["QualifyingPerformance"] = 1 - (
                        qualifying_processed["GapToPole"] / max_gap
                    )
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
                    team_strength = (
                        team_row["TeamStrength"].values[0] if not team_row.empty else 0.5
                    )

                    # Get qualifying performance
                    quali_performance = (
                        quali_row["QualifyingPerformance"]
                        if "QualifyingPerformance" in quali_row
                        else 0.5
                    )

                    # Find teammate's historical performance if available
                    teammate_performance = 0.5  # Default
                    if not historical_data["race"].empty:
                        teammates = historical_data["race"][
                            historical_data["race"]["TeamName"] == team_name
                        ]["DriverId"].unique()
                        if len(teammates) > 0:
                            teammate_races = historical_data["race"][
                                historical_data["race"]["DriverId"].isin(teammates)
                            ]
                            if not teammate_races.empty:
                                # Use average finish position of teammates
                                avg_position = teammate_races["Position"].astype(float).mean()
                                teammate_performance = 1 - (avg_position / 20)  # Normalize to 0-1

                    # Calculate estimated driver performance
                    # We give more weight to qualifying performance for rookie estimation
                    estimated_performance = (
                        quali_performance * 0.6 + team_strength * 0.3 + teammate_performance * 0.1
                    )

                    # Rookie penalty - rookies tend to make more mistakes
                    rookie_factor = 0.9  # 10% penalty for being a rookie

                    new_driver_features.append(
                        {
                            "DriverId": driver_id,
                            "FullName": quali_row["FullName"],
                            "TeamName": team_name,
                            "QualifyingPosition": quali_row["Position"],
                            "QualifyingPerformance": quali_performance,
                            "TeamStrength": team_strength,
                            "EstimatedPerformance": estimated_performance * rookie_factor,
                            "IsRookie": True,
                        }
                    )

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
            qualifying_performance = driver_data.get("QualifyingPerformance", 0.5)

            # Get real recent form (last 3 races)
            recent_form = driver_data.get("RecentForm", 0.5)

            # Get real team strength
            team_strength = driver_data.get("TeamStrength", 0.5)

            # Get real circuit characteristics
            circuit_type = circuit_data.get("CircuitType", "balanced")
            circuit_length = circuit_data.get("CircuitLength", 5.0)  # km
            circuit_elevation = circuit_data.get("CircuitElevation", 0)  # meters

            # Get real weather conditions
            temperature = weather_data.get("Temperature", 25)  # Celsius
            humidity = weather_data.get("Humidity", 50)  # percentage
            wind_speed = weather_data.get("WindSpeed", 0)  # km/h
            is_wet = weather_data.get("IsWet", False)

            # Calculate base race pace score using real data
            race_pace_score = (
                0.30 * qualifying_performance  # Qualifying performance (30%)
                + 0.25 * recent_form  # Recent form (25%)
                + 0.20 * team_strength  # Team strength (20%)
                + 0.15
                * self._get_circuit_adaptation(
                    circuit_type, circuit_length, circuit_elevation
                )  # Circuit adaptation (15%)
                + 0.10
                * self._get_weather_adaptation(
                    temperature, humidity, wind_speed, is_wet
                )  # Weather adaptation (10%)
            )

            # Apply circuit-specific adjustments based on real data
            if circuit_type == "high_speed":
                race_pace_score *= 1.05  # 5% boost for high-speed circuits
            elif circuit_type == "technical":
                race_pace_score *= 0.95  # 5% penalty for technical circuits
            elif circuit_type == "street":
                race_pace_score *= 0.90  # 10% penalty for street circuits
            elif circuit_type == "desert":
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
            if circuit_type == "high_speed":
                adaptation *= 1.05
            elif circuit_type == "technical":
                adaptation *= 0.95
            elif circuit_type == "street":
                adaptation *= 0.90
            elif circuit_type == "desert":
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
            "Driver": "DriverId",
            "Abbreviation": "DriverId",
            "Team": "TeamName",
            "GridPos": "Position",
            "BestLapTime": "BestTime",
        }

        for old_col, new_col in column_mapping.items():
            if old_col in features.columns and new_col not in features.columns:
                features[new_col] = features[old_col]

        # Ensure required columns exist
        if "Position" not in features.columns:
            features["Position"] = range(1, len(features) + 1)

        if "DriverId" not in features.columns:
            features["DriverId"] = [f"D{i}" for i in range(1, len(features) + 1)]

        if "TeamName" not in features.columns:
            features["TeamName"] = "Unknown"

        if "FullName" not in features.columns:
            features["FullName"] = features["DriverId"]

            # Calculate qualifying performance (0-1 scale)
        if "BestTime" in features.columns:
            min_time = features["BestTime"].min()
            features["QualifyingPerformance"] = 1 - ((features["BestTime"] - min_time) / min_time)
        else:
            features["QualifyingPerformance"] = 1.0 - (features["Position"] - 1) / len(features)

        # Get historical race data from dictionary
        historical_race_data = (
            historical_data.get("race", pd.DataFrame())
            if isinstance(historical_data, dict)
            else pd.DataFrame()
        )

        # Calculate DNF probability based on historical reliability
        features["DNFProbability"] = features.apply(
            lambda x: self._calculate_dnf_probability(x["TeamName"], historical_race_data), axis=1
        )

        # Calculate race pace score
        features["RacePaceScore"] = features.apply(
            lambda x: self._calculate_race_pace(x, x["Position"], track_info, self._get_weather_conditions(track_info.get('circuit_name', ''))),
            axis=1,
        )

        # Calculate projected position
        features["ProjectedPosition"] = features["RacePaceScore"].rank(ascending=False)

        # Add grid position for visualization
        features["GridPosition"] = features["Position"]

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
                "Red Bull Racing": 0.9998,  # Extremely reliable
                "Mercedes": 0.9997,  # Extremely reliable
                "Ferrari": 0.9995,  # Very reliable
                "McLaren": 0.9993,  # Very reliable
                "Aston Martin": 0.9992,  # Very reliable
                "Alpine F1 Team": 0.9990,  # Reliable
                "Williams": 0.9988,  # Reliable
                "Visa Cash App Racing Bulls F1 Team": 0.9987,  # Reliable
                "Kick Sauber": 0.9985,  # Reliable
                "Haas F1 Team": 0.9983,  # Less reliable
            }

            # Get team reliability factor
            team_factor = team_reliability.get(
                team_name, 0.9990
            )  # Default to 0.9990 for unknown teams

            # Calculate historical DNF rate if data is available
            historical_dnf_rate = 0.0
            if historical_data is not None and not historical_data.empty:
                team_dnfs = historical_data[
                    (historical_data["TeamName"] == team_name)
                    & (historical_data["Status"].str.contains("DNF", na=False))
                ].shape[0]
                team_races = historical_data[historical_data["TeamName"] == team_name].shape[0]

                if team_races > 0:
                    historical_dnf_rate = team_dnfs / team_races
                    # Cap historical DNF rate at 1% to prevent outliers
                    historical_dnf_rate = min(0.01, historical_dnf_rate)

            # Calculate recent reliability trend (last 5 races)
            recent_reliability = 0.5  # Default neutral value
            if historical_data is not None and not historical_data.empty:
                recent_races = (
                    historical_data[historical_data["TeamName"] == team_name]
                    .sort_values("Date", ascending=False)
                    .head(5)
                )

                if not recent_races.empty:
                    recent_dnfs = recent_races[
                        recent_races["Status"].str.contains("DNF", na=False)
                    ].shape[0]
                    recent_reliability = 1 - (recent_dnfs / len(recent_races))

            # Calculate circuit-specific reliability factors (reduced impact)
            circuit_factors = {
                "Monaco": 1.1,  # 10% higher chance of incidents
                "Singapore": 1.08,  # 8% higher chance of incidents
                "Baku": 1.05,  # 5% higher chance of incidents
                "Spa": 1.03,  # 3% higher chance of incidents
                "Bahrain": 0.98,  # 2% lower chance of incidents
                "Abu Dhabi": 0.98,  # 2% lower chance of incidents
                "Saudi Arabia": 0.98,  # 2% lower chance of incidents
            }

            # Apply circuit factor if available
            circuit_factor = 1.0
            if "CircuitName" in historical_data.columns and not historical_data.empty:
                circuit_name = historical_data["CircuitName"].iloc[0]
                circuit_factor = circuit_factors.get(circuit_name, 1.0)

            # Calculate weather impact on reliability (reduced impact)
            weather_factor = 1.0
            if "WeatherConditions" in historical_data.columns and not historical_data.empty:
                weather = historical_data["WeatherConditions"].iloc[0]
                if weather.get("IsWet", False):
                    weather_factor = 1.08  # 8% higher chance in wet conditions
                if weather.get("Temperature", 25) > 30:
                    weather_factor *= 1.03  # 3% higher chance in high temperatures

            # Weight the factors:
            # - 50% base probability
            # - 30% team reliability
            # - 15% historical performance
            # - 5% recent reliability trend and circuit/weather factors
            final_dnf_prob = (
                base_dnf_prob * 0.50
                + (1 - team_factor) * 0.30
                + historical_dnf_rate * 0.15
                + ((1 - recent_reliability) * 0.025 + (circuit_factor * weather_factor - 1) * 0.025)
            )

            # Cap the probability between 0.02% and 0.5%
            final_dnf_prob = max(0.0002, min(0.005, final_dnf_prob))

            # Add team-specific reliability adjustments (reduced impact)
            if team_name == "Haas F1 Team":
                final_dnf_prob *= 1.05  # 5% higher chance of DNF for Haas
            elif team_name in ["Williams", "Kick Sauber"]:
                final_dnf_prob *= 1.03  # 3% higher chance of DNF for lower midfield teams

            return final_dnf_prob

        except Exception as e:
            logger.error(f"Error calculating DNF probability: {e}")
            return 0.0005  # Return base DNF probability in case of error

    def _calculate_race_pace(self, driver_data, qualifying_position, track_info, weather_conditions):
        """Enhanced race pace calculation with improved modeling for all teams and advanced track characteristics"""
        try:
            # Extract driver and team information
            driver_name = driver_data.get('Abbreviation', '') or driver_data.get('BroadcastName', '')
            if isinstance(driver_name, pd.Series):
                driver_name = driver_name.iloc[0] if not driver_name.empty else ''
            
            team = driver_data.get('TeamName', '')
            if isinstance(team, pd.Series):
                team = team.iloc[0] if not team.empty else ''
            
            # Get driver-specific characteristics or defaults
            driver_chars = self.driver_characteristics.get(driver_name, {
                'wet_performance': 1.0,
                'tire_management': 1.0,
                'race_craft': 1.0,
                'qualifying_pace': 1.0,
                'consistency': 1.0,
                'adaptability': 1.0,
                'defensive_skill': 1.0,
                'aggressive_style': 1.0,
                'recovery_ability': 1.0,
                'track_knowledge': 1.0
            })
            
            # Calculate recent form with enhanced factors
            recent_form = self._calculate_recent_form(driver_data)
            
            # Get circuit name and characteristics
            circuit_name = track_info.get('circuit_name', '')
            circuit_chars = self.circuit_characteristics.get(circuit_name, self.default_circuit_characteristics)
            
            # Get team characteristics
            team_chars = self.team_characteristics.get(team, self.default_team_characteristics)
            
            # Define base weights for factors
            weights = {
                'qualifying_position': 0.25,
                'recent_form': 0.20,
                'track_specific': 0.20,
                'weather_impact': 0.15,
                'team_performance': 0.20
            }
            
            # Special adjustments for top drivers
            if driver_name == 'Verstappen':
                # Verstappen's ability to outperform his qualifying position
                weights['qualifying_position'] = 0.20
                weights['recent_form'] = 0.25
                weights['team_performance'] = 0.25
            elif driver_name in ['Hamilton', 'Leclerc', 'Norris']:
                # Slightly modified weights for other top drivers
                weights['qualifying_position'] = 0.22
                weights['recent_form'] = 0.23
                weights['team_performance'] = 0.22
            
            # Calculate qualifying performance factor - adjusted for better prediction
            # Lower qualifying positions have less impact on race performance
            if qualifying_position <= 3:
                quali_factor = 1.0 - (qualifying_position - 1) * 0.05
            elif qualifying_position <= 10:
                quali_factor = 0.90 - (qualifying_position - 3) * 0.03
            else:
                quali_factor = 0.70 - (qualifying_position - 10) * 0.02
                
            # Special case for Red Bull and Verstappen's recovery ability
            if team == 'Red Bull Racing':
                if driver_name == 'Verstappen' and qualifying_position > 3:
                    # Verstappen's exceptional ability to recover from worse grid positions
                    recovery_boost = driver_chars['recovery_ability'] * 0.25
                    quali_factor += recovery_boost
                elif driver_name == 'Perez' and qualifying_position > 5:
                    # Perez's ability to make up positions
                    recovery_boost = 0.15
                    quali_factor += recovery_boost
            
            # Special case for Ferrari recovery
            if team == 'Ferrari' and qualifying_position > 3:
                recovery_boost = 0.12
                quali_factor += recovery_boost
                    
            # Weather impact factor
            weather_factor = 1.0
            if weather_conditions.get('rain_chance', 0) > 0.3:
                rain_intensity = min(1.0, weather_conditions.get('rain_chance', 0))
                
                # Driver wet performance impact
                driver_wet_factor = driver_chars['wet_performance']
                
                # Team wet performance impact
                team_wet_factor = team_chars['wet_performance']
                
                # Combined wet performance with higher weight on driver skill
                weather_factor = 1.0 + (driver_wet_factor * 0.7 + team_wet_factor * 0.3 - 1.0) * rain_intensity
            
            # Temperature effects
            if 'air_temp' in weather_conditions:
                temp = weather_conditions['air_temp']
                # Team-specific temperature adjustments
                if team == 'Ferrari' and temp > 28:
                    weather_factor *= 1.05  # Ferrari performs better in higher temperatures
                elif team == 'Mercedes' and temp < 20:
                    weather_factor *= 0.95  # Mercedes struggles in lower temperatures
                elif team == 'Red Bull Racing':
                    # Red Bull is consistent across temperature ranges
                    weather_factor *= 1.02
            
            # Determine circuit type based on characteristics
            circuit_type = 'mixed'  # Default
            if circuit_chars.get('straight_speed_importance', 0) > 0.6:
                circuit_type = 'high_speed'
            elif circuit_chars.get('corner_speed_importance', 0) > 0.7:
                circuit_type = 'technical'
            elif circuit_name in ['Monaco', 'Singapore', 'Baku', 'Jeddah', 'Las Vegas']:
                circuit_type = 'street'
            
            # Calculate driver adaptation to this circuit type
            driver_circuit_adaptation = 1.0
            if circuit_type == 'high_speed':
                driver_circuit_adaptation = driver_chars['qualifying_pace'] * 0.6 + driver_chars['consistency'] * 0.4
            elif circuit_type == 'technical':
                driver_circuit_adaptation = driver_chars['race_craft'] * 0.6 + driver_chars['adaptability'] * 0.4
            elif circuit_type == 'street':
                driver_circuit_adaptation = driver_chars['defensive_skill'] * 0.5 + driver_chars['consistency'] * 0.5
            else:  # mixed
                driver_circuit_adaptation = (driver_chars['qualifying_pace'] + driver_chars['race_craft'] + 
                                          driver_chars['adaptability'] + driver_chars['consistency']) / 4
            
            # Calculate team adaptation to this circuit type
            team_circuit_adaptation = 1.0
            if circuit_type == 'high_speed':
                team_circuit_adaptation = team_chars.get('high_speed', 1.0)
            elif circuit_type == 'technical':
                team_circuit_adaptation = team_chars.get('technical_tracks', 1.0)
            elif circuit_type == 'street':
                team_circuit_adaptation = team_chars.get('street_circuits', 1.0)
            
            # Combine driver and team adaptations (60/40 split favoring team car capability)
            track_specific_factor = team_circuit_adaptation * 0.6 + driver_circuit_adaptation * 0.4
            
            # Enhanced team-specific adjustments
            team_performance = team_chars.get('race_pace', 1.0)
            
            # Special cases for certain teams
            if team == 'Red Bull Racing':
                # Red Bull's exceptional race pace regardless of qualifying
                team_boost = 1.10
                if driver_name == 'Verstappen':
                    team_boost *= 1.12  # Extra Verstappen factor
                team_performance *= team_boost
            
            elif team == 'Ferrari':
                # Ferrari's race pace improvements for 2024
                team_performance *= 1.06
            
            elif team == 'McLaren':
                # McLaren's improved race pace for 2024
                team_performance *= 1.08
                
            elif team == 'Mercedes':
                # Mercedes 2024 slight struggles
                team_performance *= 0.96
            
            # Calculate grid position advantage
            if qualifying_position <= 3:
                # Front row advantage
                grid_advantage = 1.05
            elif qualifying_position <= 6:
                # Clean side of grid in early rows
                grid_advantage = 1.02
            elif qualifying_position <= 10:
                # Midfield
                grid_advantage = 1.0
            else:
                # Back of grid
                grid_advantage = 0.98
            
            # Adjust grid advantage based on track overtaking difficulty
            overtaking_difficulty = circuit_chars.get('overtaking_difficulty', 0.5)
            if overtaking_difficulty > 0.7:  # Hard to overtake
                grid_advantage = grid_advantage ** 1.5  # Increase the importance of grid position
            elif overtaking_difficulty < 0.3:  # Easy to overtake
                grid_advantage = grid_advantage ** 0.7  # Decrease the importance of grid position
            
            # Calculate race pace value combining all factors
            race_pace = (
                quali_factor * weights['qualifying_position'] +
                recent_form * weights['recent_form'] +
                track_specific_factor * weights['track_specific'] +
                weather_factor * weights['weather_impact'] +
                team_performance * weights['team_performance']
            ) * grid_advantage
            
            # Ensure race pace is within reasonable range
            race_pace = max(0.5, min(1.5, race_pace))
            
            return race_pace
            
        except Exception as e:
            logger.error(f"Error calculating race pace: {e}")
            return 1.0  # Default in case of error

    def _calculate_recent_form(self, driver_data):
        """Enhanced recent form calculation with more sophisticated historical weighting"""
        try:
            recent_races = driver_data.get('RecentRaces', [])
            if not recent_races:
                return 1.0
            
            # Enhanced weighting for recent results - more emphasis on very recent races
            weights = [0.50, 0.25, 0.15, 0.07, 0.03]  # Exponentially decreasing weights
            weighted_positions = 0
            total_weight = 0
            
            # Track special performances for additional bonuses/penalties
            podium_count = 0
            win_count = 0
            dnf_count = 0
            comeback_performances = 0  # Count significant position improvements
            
            # Process up to 5 most recent races with detailed analysis
            for i, race in enumerate(recent_races[:5]):
                if not race:
                    continue
                    
                position = race.get('Position')
                if position is None or not isinstance(position, (int, float)):
                    continue
                    
                # Convert position to float if needed
                position = float(position)
                weight = weights[i] if i < len(weights) else 0.01
                
                # Position weighting
                weighted_positions += position * weight
                total_weight += weight
                
                # Special performance tracking
                if position == 1:
                    win_count += 1
                    if i == 0:  # Most recent race was a win
                        weighted_positions -= 0.5 * weight  # Bigger bonus for recent win
                    else:
                        weighted_positions -= 0.3 * weight
                        
                elif position <= 3:
                    podium_count += 1
                    if i == 0:  # Most recent race was a podium
                        weighted_positions -= 0.3 * weight
                    else:
                        weighted_positions -= 0.2 * weight
                        
                # Penalty for DNFs and issues
                status = race.get('Status', '')
                if status != 'Finished':
                    dnf_count += 1
                    if 'Accident' in status or 'Collision' in status:
                        weighted_positions += 0.5 * weight  # Bigger penalty for crashes
                    elif 'Technical' in status or 'Engine' in status or 'Mechanical' in status:
                        weighted_positions += 0.3 * weight  # Smaller penalty for technical issues
                    else:
                        weighted_positions += 0.4 * weight  # Medium penalty for other DNFs
                
                # Check for comeback performances (starting position vs finishing position)
                grid_pos = race.get('GridPosition')
                if grid_pos and isinstance(grid_pos, (int, float)):
                    grid_pos = float(grid_pos)
                    position_gain = grid_pos - position
                    
                    if position_gain > 5:  # Gained more than 5 positions
                        comeback_performances += 1
                        weighted_positions -= 0.2 * weight  # Bonus for strong comeback
                    elif position_gain < -3:  # Lost more than 3 positions
                        weighted_positions += 0.15 * weight  # Penalty for losing positions
            
            # Calculate base form score
            if total_weight > 0:
                avg_position = weighted_positions / total_weight
                base_form_score = 1 - (avg_position / 20)  # Normalize to 0-1 scale
            else:
                base_form_score = 0.5  # Default if no valid races
            
            # Apply special performance modifiers
            form_modifiers = 0
            
            # Consecutive podium/win streak bonus
            consecutive_strong_results = True
            for i, race in enumerate(recent_races[:3]):  # Check last 3 races
                if not race or race.get('Position', 20) > 5:
                    consecutive_strong_results = False
                    break
            
            if consecutive_strong_results:
                form_modifiers += 0.1  # Bonus for consistency at the front
            
            # Multiple wins/podiums bonus
            if win_count >= 2:
                form_modifiers += 0.15
            elif podium_count >= 2:
                form_modifiers += 0.1
            
            # DNF pattern penalty
            if dnf_count >= 2:
                form_modifiers -= 0.15
            
            # Comeback driver bonus
            if comeback_performances >= 2:
                form_modifiers += 0.08
            
            # Calculate final form score with modifiers
            final_form_score = base_form_score + form_modifiers
            
            # Ensure form score stays within reasonable bounds
            return max(0.5, min(1.5, final_form_score))
            
        except Exception as e:
            logger.error(f"Error calculating recent form: {e}")
            return 1.0

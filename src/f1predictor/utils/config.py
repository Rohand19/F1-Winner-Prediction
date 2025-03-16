"""Configuration management for the F1 prediction system."""

import os
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class Config:
    """Configuration settings for the F1 prediction system."""
    
    # Data paths
    data_dir: str = "data"
    cache_dir: str = "data/cache"
    raw_data_dir: str = "data/raw"
    models_dir: str = "models"
    results_dir: str = "results"
    
    # Model settings
    default_model_type: str = "gradient_boosting"
    tune_hyperparams: bool = False
    compare_models: bool = False
    
    # Data collection settings
    historical_races: int = 5
    include_practice: bool = False
    
    # Feature engineering settings
    feature_columns: list = None
    target_column: str = "Position"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for directory in [self.data_dir, self.cache_dir, self.raw_data_dir,
                         self.models_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
        
        if self.feature_columns is None:
            self.feature_columns = [
                "GridPosition", "Q1Time", "Q2Time", "Q3Time",
                "RacePaceScore", "AvgPosition", "DNFProbability",
                "DNFRate"
            ]

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Config':
        """Create a Config instance from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            Config instance with values from the dictionary
        """
        return cls(**{
            k: v for k, v in config_dict.items()
            if k in cls.__dataclass_fields__
        })

    def to_dict(self) -> Dict:
        """Convert the config to a dictionary.
        
        Returns:
            Dictionary representation of the config
        """
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        } 
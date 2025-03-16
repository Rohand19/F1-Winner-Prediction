"""Tests for the feature engineering module."""

import pytest
import pandas as pd
import numpy as np
from f1predictor.features.feature_engineering import F1FeatureEngineer

@pytest.fixture
def feature_engineer():
    return F1FeatureEngineer()

@pytest.fixture
def sample_qualifying_data():
    return pd.DataFrame({
        'DriverId': ['VER', 'HAM', 'LEC'],
        'FullName': ['Max Verstappen', 'Lewis Hamilton', 'Charles Leclerc'],
        'TeamName': ['Red Bull', 'Mercedes', 'Ferrari'],
        'Position': [1, 2, 3],
        'BestTime': [80.0, 80.2, 80.4]
    })

@pytest.fixture
def sample_historical_data():
    race_data = pd.DataFrame({
        'DriverId': ['VER', 'HAM', 'LEC'] * 2,
        'Position': [1, 3, 2, 2, 1, 3],
        'DNF': [False, False, True, False, False, False]
    })
    
    qualifying_data = pd.DataFrame({
        'DriverId': ['VER', 'HAM', 'LEC'] * 2,
        'Position': [1, 2, 3, 2, 1, 3]
    })
    
    return {
        'race': race_data,
        'qualifying': qualifying_data
    }

def test_feature_engineer_initialization(feature_engineer):
    assert isinstance(feature_engineer, F1FeatureEngineer)

def test_engineer_features_with_valid_data(
    feature_engineer,
    sample_qualifying_data,
    sample_historical_data
):
    features = feature_engineer.engineer_features(
        qualifying_results=sample_qualifying_data,
        historical_data=sample_historical_data,
        track_info={'Length': 5.0, 'Laps': 50}
    )
    
    assert isinstance(features, pd.DataFrame)
    assert len(features) == len(sample_qualifying_data)
    assert 'DNFProbability' in features.columns
    assert 'RacePaceScore' in features.columns

def test_engineer_features_with_missing_data(feature_engineer):
    features = feature_engineer.engineer_features(
        qualifying_results=pd.DataFrame(),
        historical_data={'race': pd.DataFrame(), 'qualifying': pd.DataFrame()},
        track_info=None
    )
    
    assert features is None or isinstance(features, pd.DataFrame)

def test_feature_ranges(
    feature_engineer,
    sample_qualifying_data,
    sample_historical_data
):
    features = feature_engineer.engineer_features(
        qualifying_results=sample_qualifying_data,
        historical_data=sample_historical_data,
        track_info={'Length': 5.0, 'Laps': 50}
    )
    
    assert all(0 <= p <= 1 for p in features['DNFProbability'])
    assert all(features['GridPosition'] > 0) 
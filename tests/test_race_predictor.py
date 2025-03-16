import pytest
import pandas as pd
import numpy as np
from src.f1predictor.models.race_predictor import RacePredictor


@pytest.fixture
def race_predictor():
    return RacePredictor(model=None, feature_columns=['GridPosition', 'RacePaceScore'])


@pytest.fixture
def sample_race_features():
    return pd.DataFrame({
        'DriverId': ['VER', 'HAM', 'PER'],
        'FullName': ['Max Verstappen', 'Lewis Hamilton', 'Sergio Perez'],
        'TeamName': ['Red Bull', 'Mercedes', 'Red Bull'],
        'GridPosition': [1, 2, 3],
        'RacePaceScore': [0.95, 0.93, 0.92],
        'ProjectedPosition': [1, 2, 3],
        'DNFProbability': [0.05, 0.05, 0.05]
    })


def test_predictor_initialization(race_predictor):
    assert isinstance(race_predictor, RacePredictor)
    assert race_predictor.feature_columns == ['GridPosition', 'RacePaceScore']
    assert race_predictor.total_laps == 58


def test_predict_finishing_positions(race_predictor, sample_race_features):
    results = race_predictor.predict_finishing_positions(sample_race_features)
    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    assert 'Position' in results.columns
    assert 'FinishStatus' in results.columns


def test_format_race_results(race_predictor):
    race_results = pd.DataFrame({
        'DriverId': ['VER', 'HAM'],
        'FullName': ['Max Verstappen', 'Lewis Hamilton'],
        'TeamName': ['Red Bull', 'Mercedes'],
        'Position': [1, 2],
        'GridPosition': [1, 2],
        'FinishTime': [5400.0, 5405.0],
        'DNF': [False, False]
    })
    formatted = race_predictor.format_race_results(race_results)
    assert isinstance(formatted, pd.DataFrame)
    assert 'FormattedGap' in formatted.columns
    assert 'FormattedRaceTime' in formatted.columns


def test_empty_race_features(race_predictor):
    results = race_predictor.predict_finishing_positions(pd.DataFrame())
    assert isinstance(results, pd.DataFrame)
    assert results.empty


def test_missing_required_columns(race_predictor):
    invalid_features = pd.DataFrame({
        'Driver': ['VER', 'HAM'],
        'Team': ['Red Bull', 'Mercedes']
    })
    results = race_predictor.predict_finishing_positions(invalid_features)
    assert isinstance(results, pd.DataFrame)
    assert results.empty 
import pytest
import pandas as pd
from src.data.data_processor import F1DataProcessor


@pytest.fixture
def data_processor():
    return F1DataProcessor(current_year=2025)


def test_data_processor_initialization(data_processor):
    assert isinstance(data_processor, F1DataProcessor)
    assert data_processor.current_year == 2025


def test_get_track_info(data_processor):
    track_info = data_processor.get_track_info(2025, 1)
    assert isinstance(track_info, dict)
    assert "Name" in track_info
    assert "Length" in track_info
    assert "Laps" in track_info


def test_load_qualifying_data_invalid_date(data_processor):
    qualifying_data = data_processor.load_qualifying_data(2025, 1)
    assert qualifying_data is None or isinstance(qualifying_data, pd.DataFrame)


def test_collect_historical_race_data(data_processor):
    historical_data = data_processor.collect_historical_race_data(
        current_year=2025, current_round=1, num_races=2
    )
    assert isinstance(historical_data, dict)
    assert "race" in historical_data
    assert "qualifying" in historical_data
    assert isinstance(historical_data["race"], pd.DataFrame)
    assert isinstance(historical_data["qualifying"], pd.DataFrame)

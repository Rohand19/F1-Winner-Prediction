import os
import logging
import pytest
from src.f1predictor.utils.config import Config
from src.f1predictor.utils.logging import setup_logging


def test_config_initialization():
    config = Config()
    assert isinstance(config, Config)
    assert config.data_dir == "data"
    assert config.default_model_type == "gradient_boosting"
    assert config.target_column == "Position"


def test_config_custom_values(tmp_path):
    data_dir = str(tmp_path / "data")
    config = Config(data_dir=data_dir, default_model_type="xgboost", historical_races=10)
    assert config.data_dir == data_dir
    assert config.default_model_type == "xgboost"
    assert config.historical_races == 10


def test_config_feature_columns():
    config = Config()
    assert isinstance(config.feature_columns, list)
    assert "GridPosition" in config.feature_columns
    assert "RacePaceScore" in config.feature_columns


def test_logging_setup(tmp_path):
    log_file = str(tmp_path / "test.log")
    setup_logging(level=logging.INFO, log_file=log_file)
    logger = logging.getLogger("test")
    logger.info("Test message")

    with open(log_file, "r") as f:
        log_content = f.read()
        assert "Test message" in log_content


def test_logging_console_only():
    setup_logging(level=logging.INFO, console=True, log_file=None)
    logger = logging.getLogger("test")
    logger.info("Test message")  # Should not raise any errors

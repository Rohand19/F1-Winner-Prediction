import pytest
import pandas as pd
import numpy as np
from src.f1predictor.models.model_trainer import F1ModelTrainer


@pytest.fixture
def model_trainer():
    return F1ModelTrainer(models_dir="test_models")


@pytest.fixture
def sample_training_data():
    return pd.DataFrame(
        {
            "DriverId": ["VER", "HAM", "PER"] * 10,
            "GridPosition": [1, 2, 3] * 10,
            "RacePaceScore": [0.95, 0.93, 0.92] * 10,
            "ProjectedPosition": [1, 2, 3] * 10,
            "DNF": [False, False, False] * 10,
            "Year": [2023] * 30,
            "RaceId": list(range(1, 11)) * 3,
        }
    )


def test_trainer_initialization(model_trainer):
    assert isinstance(model_trainer, F1ModelTrainer)
    assert model_trainer.models_dir == "test_models"
    assert hasattr(model_trainer, "feature_importance")


def test_prepare_data(model_trainer, sample_training_data):
    X, y, feature_names = model_trainer._prepare_data(sample_training_data)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, pd.Series)
    assert isinstance(feature_names, list)
    assert len(X) == len(y)
    assert "GridPosition" in feature_names
    assert "RacePaceScore" in feature_names


def test_train_position_model(model_trainer, sample_training_data):
    model = model_trainer.train_position_model(sample_training_data)
    assert model is not None


def test_evaluate_model(model_trainer, sample_training_data):
    model = model_trainer.train_position_model(sample_training_data)
    X, y, _ = model_trainer._prepare_data(sample_training_data)
    metrics = model_trainer.evaluate_model(model, sample_training_data)
    assert isinstance(metrics, dict)
    assert "mse" in metrics
    assert "r2" in metrics


def test_predict_with_model(model_trainer, sample_training_data):
    model = model_trainer.train_position_model(sample_training_data)
    predictions = model_trainer.predict_with_model(model, sample_training_data)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(sample_training_data)


def test_empty_training_data(model_trainer):
    empty_df = pd.DataFrame()
    X, y, feature_names = model_trainer._prepare_data(empty_df)
    assert X is None
    assert y is None
    assert feature_names is None

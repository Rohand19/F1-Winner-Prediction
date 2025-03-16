# Advanced F1 Race Prediction System

This project implements a comprehensive Formula 1 race prediction system that uses real qualifying data, historical race data, and machine learning to predict race outcomes with high accuracy.

## Overview

The F1 Race Prediction System analyzes various factors to predict race outcomes:

- Real qualifying data to establish starting grid positions
- Historical race pace data from the FastF1 API
- Driver and team performance metrics
- Track-specific characteristics
- Race strategy simulations
- Machine learning models to improve prediction accuracy

The system predicts:
- Final race positions for each driver
- Finish times and gaps between drivers
- Potential DNFs (Did Not Finish)
- Championship points earned

## Components

The prediction system consists of the following modules:

1. **Data Processor** (`data_processor.py`): Collects and processes F1 data from the FastF1 API, including qualifying results, historical race data, and track information.

2. **Feature Engineering** (`feature_engineering.py`): Extracts meaningful features from the raw data, including:
   - Team performance metrics
   - Driver form and historical performance
   - Track-specific characteristics
   - Qualifying performance relative to teammates
   - Race pace estimation

3. **Model Trainer** (`model_trainer.py`): Trains and evaluates machine learning models for race prediction:
   - Supports multiple model types (XGBoost, Gradient Boosting, Random Forest, etc.)
   - Includes hyperparameter tuning capabilities
   - Provides model evaluation metrics
   - Generates feature importance visualizations

4. **Race Predictor** (`race_predictor.py`): Simulates the race outcome based on engineered features:
   - Predicts finishing positions
   - Calculates race times with lap-by-lap simulation
   - Estimates potential DNFs based on reliability data
   - Calculates championship points
   - Creates visualizations of race results

5. **Main Predictor** (`main_predictor.py`): Orchestrates the end-to-end prediction process:
   - Handles command-line arguments
   - Manages the prediction pipeline
   - Generates visualizations and result outputs

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/f1-race-prediction.git
cd f1-race-prediction
```

2. Install required dependencies:
```
pip install -r requirements.txt
```

Required dependencies include:
- pandas
- numpy
- fastf1
- matplotlib
- seaborn
- scikit-learn
- xgboost
- argparse

## Usage

### Basic Usage

Predict the next upcoming race:

```
python main_predictor.py
```

Predict a specific race:

```
python main_predictor.py --year 2023 --race 3
```

Predict by event name:

```
python main_predictor.py --event "Australian"
```

### Advanced Options

Use different machine learning models:

```
python main_predictor.py --model-type random_forest
```

Compare multiple model types:

```
python main_predictor.py --compare-models
```

Tune model hyperparameters (slower but potentially more accurate):

```
python main_predictor.py --tune-hyperparams
```

Include additional visualizations:

```
python main_predictor.py --visualize
```

Use more historical race data for better predictions:

```
python main_predictor.py --historical-races 10
```

Include practice session data for predictions:

```
python main_predictor.py --include-practice
```

Force reload of data (ignore cache):

```
python main_predictor.py --reload-data
```

### Output

The system generates the following outputs in the `results` directory (or custom directory specified with `--output-dir`):

1. Race result predictions in CSV format
2. Visualizations of predicted race outcomes
3. Feature importance plots (when using machine learning models)
4. Team performance predictions
5. Starting grid vs. finishing position analysis

## How It Works

1. **Data Collection**: The system collects qualifying data for the race to be predicted, along with historical race data for feature engineering.

2. **Feature Engineering**: Raw data is transformed into meaningful features that capture driver skill, car performance, track characteristics, etc.

3. **Model Training**: Machine learning models are trained using historical data to predict race outcomes based on engineered features.

4. **Race Simulation**: A detailed race simulation takes place, accounting for factors like:
   - Qualifying performance
   - Historical race pace
   - Tire degradation
   - Starting position advantage/disadvantage
   - Reliability factors for DNF prediction

5. **Result Presentation**: The system generates formatted outputs and visualizations of the predicted race outcomes.

## Handling New Drivers

The system handles new drivers (those without historical data) by:
1. Using team performance as a baseline
2. Analyzing qualifying performance relative to teammates
3. Incorporating rookie factors into predictions

## Limitations

- Prediction accuracy depends on the quality and availability of historical data
- Cannot account for completely unexpected events (weather changes, red flags, accidents)
- New drivers with no historical data have higher prediction uncertainty

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The [FastF1](https://github.com/theOehrly/Fast-F1) project for providing access to F1 data
- Formula 1 for the racing data

---

Made with ❤️ for Formula 1 fans and data enthusiasts 
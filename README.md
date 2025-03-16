# Advanced F1 Race Prediction System

This project implements a comprehensive Formula 1 race prediction system that uses real qualifying data, historical race data, and machine learning to predict race outcomes with high accuracy.

## Overview

The F1 Race Prediction System analyzes various factors to predict race outcomes:

- Real qualifying data from FastF1 API
- Historical race data and performance metrics
- Driver and team performance analysis
- Track-specific characteristics
- Race strategy simulations
- Machine learning models with multiple algorithm options

The system predicts:
- Final race positions for each driver
- Finish times and gaps between drivers
- Potential DNFs (Did Not Finish)
- Championship points earned
- Race pace and performance metrics

## Key Features

- **Multiple ML Models**: Support for XGBoost, Gradient Boosting, Random Forest, Ridge, Lasso, and SVR
- **Comprehensive Feature Engineering**:
  - Driver performance metrics
  - Team strength analysis
  - Track-specific characteristics
  - Historical performance trends
  - Qualifying performance analysis
- **Rookie Driver Handling**: Special handling for drivers without historical data
- **Visualization Tools**: Generate plots for:
  - Grid vs. Finish position
  - Team performance analysis
  - Race pace predictions
  - DNF probability analysis
- **Data Caching**: Efficient data management with FastF1 cache system
- **Flexible Predictions**: Support for:
  - Next upcoming race
  - Specific race by round number
  - Specific race by event name
  - Historical race predictions

## Project Structure

```
f1-predictor/
├── src/                    # Source code
│   ├── data/              # Data processing and FastF1 integration
│   ├── features/          # Feature engineering and analysis
│   ├── models/            # ML models and predictions
│   ├── utils/             # Utility functions
│   └── f1predictor/       # Core prediction package
├── scripts/               # Entry point scripts
├── tests/                 # Test suite
├── data/                  # Data storage
├── models/               # Saved model states
├── results/              # Prediction outputs
├── cache/                # FastF1 cache
└── docs/                 # Documentation
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Rohand19/f1-race-prediction.git
cd f1-race-prediction
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Basic Race Prediction

To predict the next upcoming race:
```bash
python scripts/main_predictor.py
```

For a specific race:
```bash
python scripts/main_predictor.py --year 2024 --race 1
```

### Advanced Options

- Choose ML model:
```bash
python scripts/main_predictor.py --model-type xgboost
```

- Include practice session data:
```bash
python scripts/main_predictor.py --include-practice
```

- Generate visualizations:
```bash
python scripts/main_predictor.py --visualize
```

- Compare multiple models:
```bash
python scripts/main_predictor.py --compare-models
```

- Tune model hyperparameters:
```bash
python scripts/main_predictor.py --tune-hyperparams
```

### Configuration

The system can be configured through command-line arguments:

- `--year`: Race year (default: current year)
- `--race`: Race number
- `--event`: Event name (alternative to race number)
- `--historical-races`: Number of historical races to use (default: 5)
- `--model-type`: ML model to use
- `--output-dir`: Directory for results
- `--reload-data`: Force reload of cached data

## How It Works

1. **Data Collection**:
   - Fetches qualifying data from FastF1 API
   - Collects historical race data
   - Gathers track information

2. **Feature Engineering**:
   - Processes qualifying performance
   - Analyzes historical race pace
   - Calculates team performance metrics
   - Evaluates track characteristics
   - Handles new drivers and special cases

3. **Prediction Pipeline**:
   - Trains selected ML model
   - Simulates race conditions
   - Calculates finish times and gaps
   - Predicts DNF probabilities
   - Generates detailed race results

4. **Output Generation**:
   - Formatted race results
   - Performance visualizations
   - CSV exports for analysis
   - Detailed logging

## Limitations and Considerations

- Prediction accuracy depends on data quality and availability
- Cannot account for unexpected events (weather changes, accidents)
- New drivers and tracks have higher uncertainty
- Some features require recent historical data
- Practice session data may not always be available

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the test suite
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [FastF1](https://github.com/theOehrly/Fast-F1) for F1 data access
- Formula 1 for the racing data
- The F1 community for inspiration and feedback

---

Made with ❤️ for Formula 1 fans and data enthusiasts 
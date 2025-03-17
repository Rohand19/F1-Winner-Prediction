# 🏎️ Advanced F1 Race Prediction System

<div align="center">
  <img src="https://img.shields.io/badge/F1-Prediction-red?style=for-the-badge&logo=f1" alt="F1 Prediction"/>
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-AI-brightgreen?style=for-the-badge&logo=tensorflow" alt="Machine Learning"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License: MIT"/>
</div>

<br>

A sophisticated Formula 1 race prediction system that uses real-time qualifying data, historical performance metrics, and advanced machine learning to simulate and predict race outcomes with high accuracy. Experience the thrill of predicting F1 races with cutting-edge technology! 🚀

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Advanced Features](#-advanced-features)
- [How It Works](#-how-it-works)
- [Results and Visualization](#-results-and-visualization)
- [Limitations and Future Improvements](#-limitations-and-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🔍 Overview

The F1 Race Prediction System delivers comprehensive race simulations by analyzing a wide range of factors:

- 📊 Real qualifying data from FastF1 API
- 📜 Historical race data and performance metrics
- 👨‍✈️ Driver ability and consistency metrics
- 🏢 Team performance characteristics
- 🏁 Track-specific features and overtaking difficulty
- 🌧️ Dynamic weather conditions and their effects
- 🛞 Tire degradation and pit strategy simulation
- 🚦 Race incidents and safety car scenarios

The system predicts:
- 🏆 Final race positions and points for each driver
- ⏱️ Precise finish times and gaps between drivers
- 💥 Potential DNFs (Did Not Finish) with realistic probabilities
- 📈 Realistic race pace with varying conditions
- 🔄 Position changes and overtaking scenarios

## ✨ Key Features

### 🤖 Advanced Machine Learning
- **Multiple Algorithm Options**: Support for Gradient Boosting, Random Forest, and Neural Networks
- **Automated Hyperparameter Tuning**: Grid search optimization for best model performance
- **Model Evaluation**: Comprehensive metrics for model assessment

### 🔬 Sophisticated Feature Engineering
- **Driver Performance Analytics**:
  - 🏅 Historical finishing positions
  - 📊 Season points and form
  - 💧 Wet weather performance ratings
  - 🎯 Consistency metrics

- **Team Characteristics**:
  - 🛠️ Car performance metrics per track type
  - 🔌 Power unit reliability factors
  - 🛞 Tire management strategies
  - 📊 Recent development trajectory

- **Track-Specific Modeling**:
  - 🛣️ Over 20 unique track profiles
  - 🏎️ Overtaking difficulty ratings
  - 🔄 Track evolution simulation
  - 🔥 Temperature and surface modeling

### 🌦️ Advanced Race Condition Simulation
- **Dynamic Weather Effects**:
  - ☔ Rain intensity simulation
  - 💨 Changing conditions during race
  - 🌡️ Track temperature variations
  - 💦 Driver-specific wet weather skills

- **Realistic Race Incidents**:
  - 🚩 Safety car deployment simulation
  - 💥 DNF probability customized by driver/team
  - 🚧 Virtual safety car periods
  - ⚠️ Yellow flag scenarios

- **Sophisticated Pit Strategy**:
  - 🛞 Tire compound modeling
  - ⏱️ Dynamic pit stop timing
  - 🔄 Undercut/overcut simulation
  - 🛑 Team-specific pit stop performance

## 📂 Project Structure

```
f1-winner-prediction/
├── src/                    # Source code
│   ├── data/               # Data processing and FastF1 integration
│   ├── features/           # Feature engineering and analysis
│   ├── f1predictor/        # Core prediction package
│       ├── models/         # ML models and race simulation
│       └── utils/          # Utility functions
├── scripts/                # Entry point scripts
├── tests/                  # Test suite
├── data/                   # Data storage
├── models/                 # Saved model states
├── results/                # Prediction outputs
├── cache/                  # FastF1 cache
└── docs/                   # Documentation
```

## 🔧 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Rohand19/F1-Winner-Prediction.git
cd F1-Winner-Prediction
```

2. **Create and activate a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**:
```bash
pip install -r requirements.txt
```

4. **Install the FastF1 package** (required for data collection):
```bash
pip install fastf1
```

## 🚀 Usage

### Basic Race Prediction

To predict a specific race:
```bash
python scripts/main_predictor.py --year 2025 --race 1
```

For the upcoming race with default settings:
```bash
python scripts/main_predictor.py --next-race
```

### Command Line Options

```
usage: main_predictor.py [-h] --year YEAR --race RACE [--event EVENT]
                        [--historical-races HISTORICAL_RACES]
                        [--include-practice] [--reload-data]
                        [--model-type {gradient_boosting,random_forest,neural_network}]
                        [--tune-hyperparams] [--compare-models]
                        [--visualize] [--output-dir OUTPUT_DIR]
                        [--rain-chance RAIN_CHANCE] [--changing-conditions]
                        [--total-laps TOTAL_LAPS]
```

## 🔥 Advanced Features

### 🌧️ Weather Simulation

Predict how different weather conditions affect race outcomes:

```bash
# Simulate a 70% chance of rain
python scripts/main_predictor.py --year 2025 --race 1 --rain-chance 0.7

# Simulate changing weather conditions during the race
python scripts/main_predictor.py --year 2025 --race 1 --rain-chance 0.4 --changing-conditions
```

### 🎛️ Model Optimization

Fine-tune your predictions:

```bash
# Perform hyperparameter tuning
python scripts/main_predictor.py --year 2025 --race 1 --tune-hyperparams

# Generate detailed visualizations
python scripts/main_predictor.py --year 2025 --race 1 --visualize
```

### 📊 Multiple Race Analysis

Analyze an entire season:

```bash
# Run a season simulation script (requires custom script)
python scripts/season_simulator.py --year 2025 --races all
```

## ⚙️ How It Works

### 1. 📥 Data Collection Pipeline

The system begins by gathering comprehensive data:

- **Qualifying Data**: Collects detailed session timing from the FastF1 API
- **Historical Performance**: Analyzes past races to identify trends
- **Track Information**: Loads track-specific details (length, corners, surface type)
- **Weather Data**: Incorporates forecast or historical weather patterns

### 2. 🧪 Feature Engineering

Raw data is transformed into predictive features:

- **Driver Features**:
  - 📉 Recent form calculation
  - 🎯 Track-specific performance history
  - 🏎️ Qualifying improvement throughout sessions
  - 💧 Wet weather capability ratings

- **Team Features**:
  - 🔧 Car performance characteristics
  - ⚙️ Power unit reliability
  - 🏁 Track type specialization (street vs. permanent)
  - 📈 Development trajectory

- **Race Dynamics**:
  - 🏎️ Overtaking difficulty by track
  - 🛣️ Track position importance
  - 🔄 DRS effectiveness modeling
  - 🛞 Tire degradation by surface and temperature

### 3. 🧠 Prediction Model

The core system uses sophisticated machine learning:

1. **Training**: Model learns patterns from historical data
2. **Validation**: Cross-validation ensures accuracy
3. **Prediction**: Applies learned patterns to current data
4. **Simulation**: Executes a lap-by-lap race simulation

### 4. 🔄 Race Simulation

The race simulation models real-world dynamics:

- **Lap-by-Lap Timing**: Simulates realistic lap times with variation
- **Tire Degradation**: Progressive performance loss based on compound
- **Pit Strategies**: Optimal timing and compound selection
- **Race Incidents**: Realistic probabilities of DNFs and safety cars
- **Driver Interactions**: Overtaking based on relative pace and track position

## 📊 Results and Visualization

The system generates comprehensive outputs:

### 📋 Race Results Table

```
Position | Driver           | Team            | Laps | Time/Retired | Gap       | Grid | Points
---------|------------------|-----------------|------|--------------|-----------|------|-------
1        | Max Verstappen   | Red Bull Racing | 57   | 1:32:45.567  | Leader    | 1    | 25
2        | Charles Leclerc  | Ferrari         | 57   | 1:32:50.123  | +4.556s   | 2    | 18
3        | Lando Norris     | McLaren         | 57   | 1:32:51.875  | +6.308s   | 4    | 15
...
19       | Logan Sargeant   | Williams        | 56   | 1:33:12.450  | +1 Lap    | 19   | 0
DNF      | Yuki Tsunoda     | RB              | 23   | DNF          | DNF       | 11   | 0
```

### 📈 Visualizations

- **Grid vs. Finish Position**: How qualifying translates to race results
- **Team Performance**: Points distribution across teams
- **Position Changes**: Drivers who gained/lost positions
- **Pit Strategy Analysis**: Timing and effectiveness of pit stops
- **Race Pace Evolution**: Lap time trends throughout the race
- **Weather Impact Analysis**: Performance changes with weather conditions

## ⚠️ Limitations and Future Improvements

### Current Limitations

- **Unexpected Events**: Cannot predict random incidents or crashes
- **Driver Decisions**: Cannot model all human decision factors
- **Team Orders**: No modeling of strategic team decisions
- **Data Availability**: Relies on quality of available data

### 🔮 Future Improvements

- **Real-time Updates**: In-race predictions with live data
- **Video Analysis**: Computer vision for car damage and racing line analysis
- **Team Radio Integration**: NLP analysis of team communications
- **Extended Historical Analysis**: Deeper historical pattern recognition
- **User Customization**: Interface for user-defined scenario testing

## 👥 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the test suite (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastF1](https://github.com/theOehrly/Fast-F1) for the excellent F1 data API
- Formula 1 for the fascinating sport and data
- The F1 community for inspiration and feedback
- All contributors to the project

---

<div align="center">
  <b>Made with ❤️ for Formula 1 fans and data scientists everywhere</b>
  <br><br>
  <img src="https://img.shields.io/badge/Prediction%20Accuracy-High-success?style=for-the-badge" alt="High Accuracy"/>
  <img src="https://img.shields.io/badge/Updates-Regular-information?style=for-the-badge" alt="Regular Updates"/>
</div> 
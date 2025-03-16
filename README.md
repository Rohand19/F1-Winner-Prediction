# F1 Race Winner Prediction

A machine learning model that predicts Formula 1 race winners and standings based on qualifying times and historical race pace data.

## Overview

This project uses the FastF1 API to gather Formula 1 race data and builds a machine learning model to predict race outcomes for future Grand Prix events. The current implementation focuses on predicting the 2025 Australian Grand Prix results based on:

- Historical race pace data from the 2024 Australian Grand Prix
- Fictional qualifying data for the 2025 Australian Grand Prix
- Machine learning to correlate qualifying performance with race pace

## Features

- Data collection and processing using FastF1 API
- Gradient Boosting Regression model for race pace prediction
- Visualization of predicted race results
- Calculation of time gaps between drivers
- Caching of F1 data for improved performance

## Requirements

- Python 3.8+
- Libraries: fastf1, pandas, numpy, scikit-learn, matplotlib, seaborn

## Installation

1. Clone this repository
```bash
git clone https://github.com/Rohand19/f1-winner-prediction.git
cd f1-winner-prediction
```

2. Install required dependencies
```bash
pip install -r requirements.txt
```

## Usage

Run the prediction script:
```bash
python prediction.py
```

The script will:
1. Load race data from the 2024 Australian Grand Prix
2. Process qualifying data for the 2025 Australian GP
3. Train a machine learning model to predict race pace
4. Generate predicted race standings
5. Create a visualization of the results saved as `predicted_race_results.png`

## Data

The project uses:
- Real race data from the FastF1 API
- Fictional qualifying data for future races
- A mapping of driver full names to official driver codes

## Model

The prediction is based on a Gradient Boosting Regressor that:
- Takes qualifying lap times as input
- Predicts median race pace
- Calculates total race time based on the number of laps

## Results

The output includes:
- Predicted lap times for each driver
- Total race time predictions
- Time gaps to the race winner
- A visualization of the top 10 finishers

## Acknowledgments

- [FastF1](https://github.com/theOehrly/Fast-F1) for providing access to Formula 1 data
- Formula 1 for the underlying data 
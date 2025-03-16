# F1 Race Winner Prediction

A machine learning model that predicts Formula 1 race winners and standings based on qualifying times and historical race pace data.

## Overview

This project uses the FastF1 API to gather Formula 1 race data and builds a machine learning model to predict race outcomes for Grand Prix events. The current implementation focuses on predicting the Australian Grand Prix results based on:

- Historical race pace data from previous races
- Qualifying data processed from raw timing information
- Machine learning to correlate qualifying performance with race pace

The model only makes predictions for drivers who have historical race data available, ensuring reliable predictions by excluding rookies or drivers without prior data.

## Features

- Data collection and processing using FastF1 API
- Qualifying time processing from raw session data
- Gradient Boosting Regression model for race pace prediction
- Visualization of predicted race results
- Calculation of time gaps between drivers
- Caching of F1 data for improved performance
- Filtering of drivers without historical race data

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

1. First, process the qualifying data:
```bash
python qual_data.py
```
This will generate the `qualifying_times.csv` file with properly processed qualifying times.

2. Then, run the prediction script:
```bash
python prediction.py
```

The prediction script will:
1. Load historical race data from the FastF1 API
2. Load processed qualifying data from qualifying_times.csv
3. Filter out drivers without historical race data
4. Train a machine learning model to predict race pace
5. Generate predicted race standings
6. Create a visualization of the results saved as `predicted_race_results.png`

## Data

The project uses:
- Real race data from the FastF1 API
- Processed qualifying data from Q1, Q2, and Q3 sessions
- A mapping of driver names to official driver codes

## Data Processing

The data processing follows these steps:
1. The `qual_data.py` file processes raw qualifying times and handles different qualifying sessions (Q1, Q2, Q3)
2. For each driver, the best time from their highest achieved session is used
3. The processed data is saved to `qualifying_times.csv`
4. The prediction script loads this data and filters it based on available historical race data

## Model

The prediction is based on a Gradient Boosting Regressor that:
- Takes qualifying lap times as input
- Predicts median race pace
- Calculates total race time based on the number of laps
- Only makes predictions for drivers with historical data

## Results

The output includes:
- Predicted lap times for each driver with historical data
- Total race time predictions
- Time gaps to the race winner
- A visualization of the top 10 finishers
- A list of excluded drivers without historical data
- A list of drivers excluded due to missing qualifying times

## Acknowledgments

- [FastF1](https://github.com/theOehrly/Fast-F1) for providing access to Formula 1 data
- Formula 1 for the underlying data 
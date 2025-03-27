# 🏎️ F1 Race Prediction System - Web Interface

A user-friendly web interface for the Advanced F1 Race Prediction System, built with Streamlit.

![F1 Prediction Web Interface](https://placeholder-for-your-screenshot.png)

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Running the Interface](#running-the-interface)
- [Features](#features)
- [Using the Interface](#using-the-interface)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## Overview

This web interface provides a user-friendly way to interact with the F1 Race Prediction System. It allows users to configure prediction parameters, run predictions, visualize results, and analyze historical data without the need to use command-line tools or understand the underlying code.

The interface is built with [Streamlit](https://streamlit.io/), a powerful Python library for creating web applications for data science and machine learning projects.

## Installation

### Prerequisites

- Python 3.8 or higher
- The F1 Race Prediction System repository

### Dependencies

The web interface requires the following Python packages:

- streamlit
- pandas
- numpy
- matplotlib
- plotly
- (+ all dependencies required by the F1 Race Prediction System)

Install the required packages with:

```bash
pip install streamlit pandas plotly matplotlib
```

## Running the Interface

To start the web interface:

1. Navigate to the project root directory:
```bash
cd F1-Winner-Prediction
```

2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. The interface will automatically open in your default web browser, typically at http://localhost:8501

## Features

### 1. Prediction Configuration

- **Year and Race Selection**: Choose the Formula 1 season and race number to predict
- **Model Selection**: Select from different machine learning models:
  - Gradient Boosting (default)
  - Random Forest
  - Neural Network
- **Hyperparameter Tuning**: Toggle advanced model optimization
- **Weather Conditions**: Set race day weather:
  - Dry
  - Light Rain
  - Heavy Rain
  - Changing Conditions

### 2. Results Display

- **Race Results Table**: Detailed finishing order with position, points, and time gaps
- **Interactive Visualizations**: Multiple visualizations of race results:
  - Final positions and points
  - Team performance comparison
  - Grid vs. finish position analysis
  - Gap to winner visualization
  - Prediction accuracy (for races with known results)

### 3. Performance Metrics

- **Accuracy Metrics**: Various metrics to evaluate prediction performance:
  - Mean Absolute Error
  - Top 3 Accuracy
  - Top 5 Accuracy
  - Top 10 Accuracy
- **Detailed Metrics**: Expandable section with all available metrics

### 4. Historical Analysis

- **Previous Predictions**: Browse and analyze past prediction runs
- **Comparative Analysis**: Compare predictions across different races or model configurations

## Using the Interface

### Making a Prediction

1. **Configure Settings**:
   - In the sidebar, select the year and race number
   - Choose your preferred model type
   - Toggle hyperparameter tuning if desired
   - Select weather conditions

2. **Run Prediction**:
   - Click the "Run Prediction" button
   - A progress bar will appear while the prediction is running
   - Results will appear once processing is complete

3. **Analyze Results**:
   - Navigate through the tabs to view:
     - Race Results: The full predicted finishing order
     - Visualization: Interactive charts of the race outcomes
     - Metrics: Performance metrics (if comparing to actual results)
     - Log Output: Detailed output from the prediction process

### Exploring Historical Results

1. **Select Historical Result**:
   - Scroll down to the "Historical Analysis" section
   - Use the dropdown to select a past prediction run
   
2. **Analyze Historical Data**:
   - Navigate through the tabs to view the historical prediction data
   - Compare different races or model configurations

## Customization

### Appearance

The web interface uses custom CSS for styling. You can modify the appearance by editing the CSS in the `app.py` file:

```python
# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ff1801;
        ...
    }
    ...
</style>
""", unsafe_allow_html=True)
```

### Team Colors

The team colors for visualizations can be customized by editing the `TEAM_COLORS` dictionary in the `app.py` file:

```python
TEAM_COLORS = {
    "Red Bull Racing": "#0600EF",
    "Ferrari": "#DC0000",
    ...
}
```

## Troubleshooting

### Common Issues

1. **Interface Not Starting**:
   - Ensure Streamlit is installed: `pip install streamlit`
   - Make sure you're running the command from the project root directory

2. **Prediction Process Fails**:
   - Check the error output in the "Log Output" tab
   - Ensure that the F1 Race Prediction System is properly installed
   - Verify that the required data files are available

3. **Visualizations Not Appearing**:
   - Ensure plotly is installed: `pip install plotly`
   - Check that the results data contains the expected columns

4. **Slow Performance**:
   - Avoid using hyperparameter tuning for quick predictions
   - Consider reducing the number of historical races used for training

## Development

### Adding New Features

The web interface can be extended with new features by modifying the `app.py` file:

1. **New Visualizations**:
   - Add new visualization functions similar to `visualize_race_results()`
   - Create new tabs within the existing tab structure

2. **Additional Configuration Options**:
   - Add new input widgets to the sidebar
   - Update the `run_prediction()` function to pass new parameters

3. **Enhanced Data Analysis**:
   - Implement new analysis functions
   - Add new sections to the interface using Streamlit components

### Best Practices

- Keep the UI clean and intuitive
- Provide clear instructions and tooltips for complex options
- Ensure all visualizations have proper titles, labels, and legends
- Test new features with various prediction scenarios

---

For issues, suggestions, or contributions, please visit the [main project repository](https://github.com/Rohand19/F1-Winner-Prediction). 
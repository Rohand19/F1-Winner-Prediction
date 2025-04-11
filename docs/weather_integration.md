# Live Weather Integration for F1 Race Prediction

This document explains how to use the live weather integration feature in the F1 Race Prediction system.

## Overview

The F1 Race Prediction system now supports fetching real-time weather data for race locations, which can significantly enhance the realism and accuracy of race simulations. Weather conditions such as temperature, humidity, rain probability, and wind speed are crucial factors affecting race performance in Formula 1.

## Setup

To use live weather data, you'll need an API key from a weather service provider. The system currently supports OpenWeatherMap, but it can be extended to support other providers.

### Getting an API Key

1. Go to [OpenWeatherMap](https://openweathermap.org/) and create a free account
2. Navigate to your API keys section and copy your API key
3. You can either:
   - Set it as an environment variable: `export WEATHER_API_KEY=your_api_key_here`
   - Pass it directly when running the prediction: `--weather-api-key your_api_key_here`

## Using Live Weather

To enable live weather for race predictions, use the `--live-weather` flag when running the predictor:

```bash
python scripts/main_predictor.py --year 2025 --race 2 --live-weather
```

This will:
1. Fetch current weather data for the race location
2. Use the retrieved conditions in the race simulation
3. Save the weather data to a JSON file in the results directory

### Combining with Other Weather Options

You can combine live weather with other options:

- **With changing conditions**: `--live-weather --changing-conditions`
  This will use live weather data as a starting point but simulate dynamic changes during the race.

- **With rain chance override**: `--live-weather --rain-chance 0.8`
  This will use live weather data for temperature, humidity, etc., but override the rain probability with your specified value.

## Command-line Options

| Flag | Description |
|------|-------------|
| `--live-weather` | Enable fetching and using live weather data |
| `--weather-api-key KEY` | Provide your weather API key directly |
| `--rain-chance VALUE` | Override the rain probability (0.0-1.0) |
| `--changing-conditions` | Enable dynamic weather changes during race |

## Example

```bash
# Full example with all weather-related options
python scripts/main_predictor.py \
  --year 2025 \
  --race 5 \
  --tune-hyperparams \
  --compare-models \
  --visualize \
  --live-weather \
  --changing-conditions
```

## Weather Data Format

The weather data is stored in a JSON file with the following structure:

```json
{
  "track_temp": 30.5,
  "air_temp": 25.3,
  "humidity": 65.0,
  "wind_speed": 12.6,
  "rain_chance": 0.35,
  "changing_conditions": true,
  "weather_description": "scattered clouds",
  "timestamp": "2023-06-22T14:30:45.123456",
  "source": "OpenWeatherMap API"
}
```

## Fallback Mechanism

If live weather data cannot be retrieved (due to API issues, rate limiting, or network problems), the system will automatically fall back to default weather data for the specific circuit.

## Future Enhancements

Planned enhancements to the weather integration include:
- Support for additional weather providers
- Weather forecasting for upcoming races
- Historical weather pattern analysis for better predictions
- Circuit-specific microclimate modeling 
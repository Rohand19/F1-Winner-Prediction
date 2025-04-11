#!/usr/bin/env python3
"""
Example script demonstrating how to use live weather integration
for F1 race predictions.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Ensure the src directory is in the path for importing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.f1predictor.utils import WeatherService
from src.f1predictor.data_processor import F1DataProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="F1 Race Prediction with Live Weather")
    
    parser.add_argument("--year", type=int, required=True, help="Year of the race")
    parser.add_argument("--race", type=int, required=True, help="Race number")
    parser.add_argument("--api-key", type=str, help="Weather API key (or set WEATHER_API_KEY env var)")
    parser.add_argument("--save-only", action="store_true", help="Only save weather data without running prediction")
    
    return parser.parse_args()

def get_track_name(year, race):
    """Get track name from F1DataProcessor."""
    data_processor = F1DataProcessor(year)
    race_info = data_processor.get_race_info(race)
    if race_info is None:
        raise ValueError(f"Could not find race info for {year}, race {race}")
    
    # Extract track name or location
    track_name = race_info.get('EventName', '')
    if 'Grand Prix' in track_name:
        # Extract country/city name
        track_name = track_name.replace('Grand Prix', '').strip()
    
    return track_name

def main():
    args = parse_args()
    
    try:
        # Get track name for the selected race
        track_name = get_track_name(args.year, args.race)
        logger.info(f"Found track: {track_name}")
        
        # Initialize the weather service
        weather_service = WeatherService(api_key=args.api_key)
        
        # Fetch weather data
        logger.info(f"Fetching weather data for {track_name}...")
        weather_data = weather_service.get_current_weather(track_name)
        
        # Display weather information
        logger.info(f"Weather data retrieved:")
        logger.info(f"  Description: {weather_data.get('weather_description', 'N/A')}")
        logger.info(f"  Air Temperature: {weather_data.get('air_temp', 'N/A')}°C")
        logger.info(f"  Track Temperature: {weather_data.get('track_temp', 'N/A')}°C")
        logger.info(f"  Humidity: {weather_data.get('humidity', 'N/A')}%")
        logger.info(f"  Wind Speed: {weather_data.get('wind_speed', 'N/A')} km/h")
        logger.info(f"  Rain Chance: {weather_data.get('rain_chance', 0) * 100:.1f}%")
        logger.info(f"  Changing Conditions: {weather_data.get('changing_conditions', False)}")
        
        # Save weather data to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"weather_{track_name.replace(' ', '_')}_{timestamp}.json"
        
        import json
        with open(filename, 'w') as f:
            json.dump(weather_data, f, indent=2)
        
        logger.info(f"Weather data saved to {filename}")
        
        # Run prediction if not save-only mode
        if not args.save_only:
            # Prepare command to run prediction with live weather
            cmd = [
                "python", "../scripts/main_predictor.py",
                f"--year={args.year}",
                f"--race={args.race}",
                "--tune-hyperparams",
                "--compare-models",
                "--visualize",
                "--live-weather"
            ]
            
            if args.api_key:
                cmd.append(f"--weather-api-key={args.api_key}")
                
            # Print command for reference
            logger.info(f"Running prediction with command: {' '.join(cmd)}")
            
            # Execute the command
            import subprocess
            result = subprocess.run(cmd, check=True)
            
            if result.returncode == 0:
                logger.info("Prediction completed successfully")
            else:
                logger.error(f"Prediction failed with exit code {result.returncode}")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
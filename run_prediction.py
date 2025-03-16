#!/usr/bin/env python3
import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Import custom modules
from data_processor import F1DataProcessor
from feature_engineering import F1FeatureEngineer
from race_predictor import RacePredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("F1Predictor.Run")

def main():
    """
    Simplified prediction pipeline with robust error handling
    """
    # Setup parameters
    year = 2025  # Use a year with known data
    race_round = 1
    
    # Create output directory
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Initialize data processor
        logger.info(f"Initializing data processor for {year} Round {race_round}")
        data_processor = F1DataProcessor(current_year=year)
        
        # Debug: Print available events for the year
        schedule = data_processor.load_event_schedule(year)
        if schedule is not None:
            logger.info(f"Available events for {year}:")
            for i, event in schedule.iterrows():
                round_num = event.get('RoundNumber', i+1)
                event_name = event.get('EventName', f"Round {round_num}")
                logger.info(f"  Round {round_num}: {event_name}")
        
        # Load qualifying data
        logger.info(f"Loading qualifying data for {year} Round {race_round}")
        qualifying_data = data_processor.load_qualifying_data(year, race_round)
        
        if qualifying_data is None or qualifying_data.empty:
            logger.warning("No qualifying data available. Using mock data.")
            # Create mock qualifying data
            qualifying_data = pd.DataFrame({
                'DriverId': ['VER', 'HAM', 'LEC', 'PER', 'SAI', 'RUS', 'ALO', 'NOR', 'STR', 'OCO'],
                'FullName': ['Max Verstappen', 'Lewis Hamilton', 'Charles Leclerc', 'Sergio Perez', 'Carlos Sainz',
                            'George Russell', 'Fernando Alonso', 'Lando Norris', 'Lance Stroll', 'Esteban Ocon'],
                'TeamName': ['Red Bull', 'Mercedes', 'Ferrari', 'Red Bull', 'Ferrari',
                            'Mercedes', 'Aston Martin', 'McLaren', 'Aston Martin', 'Alpine'],
                'Position': [1, 3, 2, 4, 5, 6, 8, 7, 10, 9],
                'BestTime': [85.0, 85.8, 85.5, 86.0, 86.2, 86.5, 87.0, 86.8, 87.5, 87.3]
            })
        else:
            logger.info(f"Loaded qualifying data for {len(qualifying_data)} drivers")
        
        # Get track info
        logger.info(f"Getting track info for {year} Round {race_round}")
        track_info = data_processor.get_track_info(year, race_round)
        logger.info(f"Track info: {track_info}")
        
        # Collect historical data
        logger.info(f"Collecting historical data for {year} Round {race_round}")
        historical_data = data_processor.collect_historical_race_data(
            current_year=year,
            current_round=race_round,
            num_races=5,
            include_practice=False
        )
        
        if historical_data is None:
            logger.warning("Failed to collect historical data. Using empty data.")
            historical_data = {'race': pd.DataFrame(), 'qualifying': pd.DataFrame(), 'practice': pd.DataFrame()}
        
        # Initialize feature engineer
        logger.info("Initializing feature engineer")
        feature_engineer = F1FeatureEngineer()
        
        # Generate features
        logger.info("Generating race features")
        race_features = feature_engineer.engineer_features(
            qualifying_results=qualifying_data,
            historical_data=historical_data,
            track_info=track_info
        )
        
        if race_features is None or race_features.empty:
            logger.error("Failed to generate features. Exiting.")
            sys.exit(1)
            
        logger.info(f"Generated features for {len(race_features)} drivers")
        logger.info(f"Feature columns: {race_features.columns.tolist()}")
        
        # Initialize race predictor
        total_laps = track_info.get('Laps', 58) if track_info else 58
        logger.info(f"Initializing race predictor with {total_laps} laps")
        race_predictor = RacePredictor(total_laps=total_laps)
        
        # Predict race results
        logger.info("Predicting race results")
        race_results = race_predictor.predict_finishing_positions(race_features)
        
        # Print race results
        if race_results is not None and not race_results.empty:
            logger.info("Race prediction results:")
            
            # Check which columns are available for sorting
            sort_column = 'Position'
            if sort_column not in race_results.columns:
                if 'ProjectedPosition' in race_results.columns:
                    sort_column = 'ProjectedPosition'
                elif 'FinishTime' in race_results.columns:
                    sort_column = 'FinishTime'
                else:
                    # If no position column is available, don't sort
                    sort_column = None
            
            # Display results
            if sort_column:
                sorted_results = race_results.sort_values(sort_column)
            else:
                sorted_results = race_results
                
            for i, (_, driver) in enumerate(sorted_results.iterrows(), 1):
                status = "Finished" if not driver.get('DNF', False) else "DNF"
                team = driver.get('TeamName', 'Unknown Team')
                name = driver.get('FullName', driver.get('DriverId', f'Driver {i}'))
                logger.info(f"{i}. {name} ({team}): {status}")
            
            # Save results to CSV
            results_file = os.path.join(output_dir, f"race_results_{year}_round{race_round}.csv")
            race_results.to_csv(results_file, index=False)
            logger.info(f"Saved race results to {results_file}")
        else:
            logger.error("Failed to generate race predictions")
            
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        sys.exit(1)
        
    logger.info("F1 Race Prediction completed successfully")

if __name__ == "__main__":
    main()
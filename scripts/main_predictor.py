#!/usr/bin/env python3
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
from src.data.data_processor import F1DataProcessor
from src.features.feature_engineering import F1FeatureEngineer
from src.models.race_predictor import RacePredictor, print_race_results
from src.models.model_trainer import F1ModelTrainer

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"f1_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("F1Predictor.Main")

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="F1 Race Prediction System")
    
    # Race information
    parser.add_argument('--year', type=int, default=datetime.now().year,
                      help='Year of the race to predict')
    parser.add_argument('--race', type=int, default=None,
                      help='Race number to predict (default: next upcoming race)')
    parser.add_argument('--event', type=str, default=None,
                      help='Event name (optional, used instead of race number)')
    
    # Data options
    parser.add_argument('--historical-races', type=int, default=5,
                      help='Number of historical races to use for training data')
    parser.add_argument('--include-practice', action='store_true',
                      help='Include practice session data for predictions')
    parser.add_argument('--reload-data', action='store_true',
                      help='Force reload of all data (ignore cache)')
    
    # Training options
    parser.add_argument('--model-type', type=str, default='xgboost',
                      choices=['xgboost', 'gradient_boosting', 'random_forest', 'ridge', 'lasso', 'svr'],
                      help='Model type to use for predictions')
    parser.add_argument('--tune-hyperparams', action='store_true',
                      help='Tune model hyperparameters (slower)')
    parser.add_argument('--compare-models', action='store_true',
                      help='Compare multiple model types')
    parser.add_argument('--visualize', action='store_true',
                      help='Generate visualizations')
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Directory to save results and visualizations')
    
    return parser.parse_args()


def setup_directories(args):
    """
    Create required directories
    """
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logger.info(f"Created output directory: {args.output_dir}")
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join('cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        logger.info(f"Created cache directory: {cache_dir}")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join('models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"Created models directory: {models_dir}")

def get_current_race_info(args, data_processor):
    """
    Get information about the race to predict
    
    Args:
        args: Command line arguments
        data_processor: F1DataProcessor object
        
    Returns:
        Tuple of (year, race_round, event_name)
    """
    try:
        year = args.year
        
        # If race round is not specified, get the next upcoming race
        if args.race is None and args.event is None:
            upcoming_race = data_processor.get_upcoming_race()
            if upcoming_race is None:
                logger.warning("Could not determine upcoming race. Using latest completed race.")
                # Use the latest completed race
                schedule = data_processor.get_season_schedule(year)
                latest_race = schedule[schedule['EventDate'] < datetime.now()].iloc[-1]
                race_round = latest_race['RoundNumber']
                event_name = latest_race['EventName']
                logger.info(f"Using latest completed race: {event_name} (Round {race_round})")
            else:
                race_round = upcoming_race['RoundNumber']
                event_name = upcoming_race['EventName']
                logger.info(f"Predicting upcoming race: {event_name} (Round {race_round})")
        elif args.event is not None:
            # Find race by event name
            schedule = data_processor.get_season_schedule(year)
            matching_events = schedule[schedule['EventName'].str.contains(args.event, case=False)]
            
            if matching_events.empty:
                logger.error(f"Could not find event matching '{args.event}' in {year} season")
                sys.exit(1)
                
            race_round = matching_events.iloc[0]['RoundNumber']
            event_name = matching_events.iloc[0]['EventName']
            logger.info(f"Predicting race: {event_name} (Round {race_round})")
        else:
            # Use specified race round
            race_round = args.race
            schedule = data_processor.get_season_schedule(year)
            matching_round = schedule[schedule['RoundNumber'] == race_round]
            
            if matching_round.empty:
                logger.error(f"Could not find Round {race_round} in {year} season")
                sys.exit(1)
                
            event_name = matching_round.iloc[0]['EventName']
            logger.info(f"Predicting race: {event_name} (Round {race_round})")
            
        return year, race_round, event_name
    except Exception as e:
        logger.error(f"Error getting race information: {e}")
        sys.exit(1)

def main():
    """
    Main prediction pipeline
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup required directories
    setup_directories(args)
    
    logger.info("Starting F1 Race Prediction System")
    logger.info(f"Arguments: {args}")
    
    try:
        # Initialize data processor
        data_processor = F1DataProcessor(
            current_year=args.year
        )
        
        # Get race information
        year, race_round, event_name = get_current_race_info(args, data_processor)
        
        # Load qualifying data for current race
        qualifying_data = data_processor.load_qualifying_data(year, race_round)
        
        if qualifying_data is None or qualifying_data.empty:
            logger.error(f"No qualifying data available for {event_name} {year}")
            logger.info("Checking if qualifying session has been completed...")
            
            # Check if qualifying has been completed
            event_schedule = data_processor.get_event_schedule(year, race_round)
            if event_schedule is not None:
                qualifying_session = event_schedule[event_schedule['Session'].str.contains('Qualifying', case=False)]
                
                if not qualifying_session.empty:
                    qual_time = qualifying_session.iloc[0]['SessionStart']
                    if qual_time > datetime.now():
                        logger.info(f"Qualifying session has not started yet. Scheduled for: {qual_time}")
                    else:
                        logger.info(f"Qualifying session should have completed at: {qual_time}")
                        logger.info("Data may not be available yet or there was an error loading it.")
            
            # Exit if no qualifying data (can't make predictions without it)
            logger.error("Cannot make predictions without qualifying data")
            sys.exit(1)
            
        logger.info(f"Loaded qualifying data for {len(qualifying_data)} drivers")
        
        # Collect historical race data for feature engineering
        historical_data = data_processor.collect_historical_race_data(
            current_year=year,
            current_round=race_round,
            num_races=args.historical_races,
            include_practice=args.include_practice
        )
        
        if historical_data is None or len(historical_data) == 0:
            logger.error("Failed to collect historical race data")
            sys.exit(1)
            
        logger.info(f"Collected historical data from {len(historical_data)} races")
        
        # Get track information
        track_info = data_processor.get_track_info(year, race_round)
        
        if track_info is None:
            logger.warning("Could not retrieve track information")
        else:
            logger.info(f"Track: {track_info.get('Name', 'Unknown')} ({track_info.get('Length', 'Unknown')}km)")
            
        # Initialize feature engineer
        feature_engineer = F1FeatureEngineer()
        
        # Generate features for prediction
        race_features = feature_engineer.engineer_features(
            qualifying_results=qualifying_data,
            historical_data=historical_data,
            track_info=track_info
        )
        
        if race_features is None or race_features.empty:
            logger.error("Failed to engineer features for prediction")
            sys.exit(1)
            
        logger.info(f"Generated features for {len(race_features)} drivers")
        
        # Initialize race predictor (get number of laps from track info)
        total_laps = track_info.get('Laps', 58) if track_info else 58
        race_predictor = RacePredictor(total_laps=total_laps)
        
        if args.compare_models:
            # Compare different model types
            logger.info("Comparing different model types...")
            
            # Initialize model trainer
            model_trainer = F1ModelTrainer(models_dir='models')
            
            # Compare models
            comparison_results = model_trainer.compare_models(
                race_features,
                tune_hyperparams=args.tune_hyperparams
            )
            
            if comparison_results is None:
                logger.error("Model comparison failed")
                # Fallback to direct prediction
                logger.info("Falling back to direct prediction without ML model")
                race_results = race_predictor.predict_and_visualize(
                    race_features,
                    title=f"{event_name} {year} Prediction",
                    output_file=os.path.join(args.output_dir, f"prediction_{year}_round{race_round}.png")
                )
            else:
                logger.info(f"Best model: {comparison_results['best_model']}")
                
                # Use the best model for prediction
                best_model_type = comparison_results['best_model']
                best_model = comparison_results['models'][best_model_type]
                
                # Plot feature importance
                model_trainer.plot_feature_importance(
                    output_file=os.path.join(args.output_dir, f"feature_importance_{year}_round{race_round}.png")
                )
                
                # Make predictions
                positions = model_trainer.predict_with_model(best_model, race_features)
                
                if positions is None:
                    logger.error("Failed to make predictions with best model")
                    # Fallback to direct prediction
                    logger.info("Falling back to direct prediction without ML model")
                    race_results = race_predictor.predict_and_visualize(
                        race_features,
                        title=f"{event_name} {year} Prediction",
                        output_file=os.path.join(args.output_dir, f"prediction_{year}_round{race_round}.png")
                    )
                else:
                    # Update the projected positions
                    for i, (idx, _) in enumerate(race_features.iterrows()):
                        race_features.loc[idx, "ProjectedPosition"] = positions[i]
                    
                    # Get final race prediction
                    race_results = race_predictor.predict_and_visualize(
                        race_features,
                        title=f"{event_name} {year} Prediction (Model: {best_model_type})",
                        output_file=os.path.join(args.output_dir, f"prediction_{year}_round{race_round}.png")
                    )
        else:
            # Use selected model type
            if args.model_type != 'xgboost':
                logger.info(f"Training {args.model_type} model...")
                
                # Initialize model trainer
                model_trainer = F1ModelTrainer(models_dir='models')
                
                # Train model
                model = model_trainer.train_position_model(
                    race_features,
                    model_type=args.model_type,
                    tune_hyperparams=args.tune_hyperparams
                )
                
                if model is not None:
                    # Plot feature importance
                    model_trainer.plot_feature_importance(
                        output_file=os.path.join(args.output_dir, f"feature_importance_{year}_round{race_round}.png")
                    )
                    
                    # Make predictions
                    positions = model_trainer.predict_with_model(model, race_features)
                    
                    # Update the projected positions
                    for i, (idx, _) in enumerate(race_features.iterrows()):
                        race_features.loc[idx, "ProjectedPosition"] = positions[i]
                
            # Get final race prediction
            race_results = race_predictor.predict_and_visualize(
                race_features,
                title=f"{event_name} {year} Prediction",
                output_file=os.path.join(args.output_dir, f"prediction_{year}_round{race_round}.png")
            )
        
        # Print race results
        if race_results is not None and not race_results.empty:
            print_race_results(race_results)
            
            # Save results to CSV
            results_file = os.path.join(args.output_dir, f"race_results_{year}_round{race_round}.csv")
            race_results.to_csv(results_file, index=False)
            logger.info(f"Saved race results to {results_file}")
            
            if args.visualize:
                # Create additional visualizations
                
                # 1. Starting grid vs. finishing positions
                plt.figure(figsize=(12, 8))
                grid = race_results[['FullName', 'StartPosition', 'FinishPosition']].copy()
                grid = grid[grid['FinishPosition'].notna()]  # Filter out DNFs
                
                # Reshape for heatmap
                grid_vs_finish = pd.DataFrame(
                    0, 
                    index=range(1, grid['StartPosition'].max() + 1),
                    columns=range(1, grid['FinishPosition'].max() + 1)
                )
                
                for _, row in grid.iterrows():
                    start = int(row['StartPosition'])
                    finish = int(row['FinishPosition'])
                    grid_vs_finish.loc[start, finish] += 1
                
                # Plot heatmap
                sns.heatmap(grid_vs_finish, annot=True, cmap='YlGnBu', fmt='d', cbar_kws={'label': 'Number of Drivers'})
                plt.title('Starting Grid vs. Finishing Positions', fontsize=16)
                plt.xlabel('Finishing Position', fontsize=12)
                plt.ylabel('Grid Position', fontsize=12)
                plt.tight_layout()
                
                grid_plot_file = os.path.join(args.output_dir, f"grid_vs_finish_{year}_round{race_round}.png")
                plt.savefig(grid_plot_file, dpi=300, bbox_inches='tight')
                logger.info(f"Saved grid vs. finish plot to {grid_plot_file}")
                
                # 2. Team performance
                plt.figure(figsize=(12, 8))
                team_results = race_results.groupby('TeamName').agg({
                    'Points': 'sum',
                    'DriverId': 'count'
                }).reset_index()
                team_results = team_results.rename(columns={'DriverId': 'DriversFinished'})
                team_results = team_results.sort_values('Points', ascending=False)
                
                sns.barplot(x='Points', y='TeamName', data=team_results)
                plt.title('Predicted Team Performance', fontsize=16)
                plt.xlabel('Points', fontsize=12)
                plt.ylabel('Team', fontsize=12)
                
                # Add points and drivers finishing
                for i, row in enumerate(team_results.itertuples()):
                    plt.text(
                        row.Points + 0.5, i, 
                        f"{row.Points} pts ({row.DriversFinished} driver{'s' if row.DriversFinished > 1 else ''})",
                        va='center'
                    )
                
                plt.tight_layout()
                
                team_plot_file = os.path.join(args.output_dir, f"team_performance_{year}_round{race_round}.png")
                plt.savefig(team_plot_file, dpi=300, bbox_inches='tight')
                logger.info(f"Saved team performance plot to {team_plot_file}")
                
        else:
            logger.error("Failed to generate race predictions")
            
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        sys.exit(1)
        
    logger.info("F1 Race Prediction completed successfully")

if __name__ == "__main__":
    main() 
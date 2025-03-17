#!/usr/bin/env python3
import sys
import os
import pandas as pd
import subprocess
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("F1Predictor.Season")


def run_race_prediction(year, race):
    """Run prediction for a specific race"""
    try:
        cmd = f"python3.11 scripts/main_predictor.py --year {year} --race {race} --tune-hyperparams"
        subprocess.run(cmd, shell=True, check=True)

        # Find the most recent results directory
        results_dirs = [d for d in os.listdir("results") if d.startswith("run_")]
        if not results_dirs:
            logger.error(f"No results directory found for race {race}")
            return None

        latest_dir = max(results_dirs, key=lambda x: os.path.getctime(os.path.join("results", x)))
        results_file = f"results/{latest_dir}/race_results_{latest_dir}.csv"

        if os.path.exists(results_file):
            return pd.read_csv(results_file)
        else:
            logger.error(f"No results file found for race {race}")
            return None
    except Exception as e:
        logger.error(f"Error predicting race {race}: {e}")
        return None


def calculate_championship_standings(all_results):
    """Calculate driver and constructor championship standings"""

    # Initialize standings DataFrames
    driver_standings = pd.DataFrame()
    constructor_standings = pd.DataFrame()

    # Process each race's results
    for race_num, results in all_results.items():
        if results is None:
            continue

        # Add race points to driver standings
        race_points = results[["DriverId", "Driver", "Team", "Points"]].copy()

        if driver_standings.empty:
            driver_standings = race_points.rename(columns={"Points": f"Race{race_num}"})
        else:
            # Merge with existing standings
            driver_standings = driver_standings.merge(
                race_points[["DriverId", "Points"]].rename(columns={"Points": f"Race{race_num}"}),
                on="DriverId",
                how="outer",
            )

            # Update driver and team info if missing
            missing_info = driver_standings["Driver"].isna()
            if missing_info.any():
                driver_info = race_points[["DriverId", "Driver", "Team"]]
                driver_standings.loc[missing_info, ["Driver", "Team"]] = driver_standings[
                    missing_info
                ].merge(driver_info, on="DriverId", how="left")[["Driver", "Team"]]

        # Calculate constructor points
        team_points = results.groupby("Team")["Points"].sum().reset_index()

        if constructor_standings.empty:
            constructor_standings = team_points.rename(columns={"Points": f"Race{race_num}"})
        else:
            constructor_standings = constructor_standings.merge(
                team_points.rename(columns={"Points": f"Race{race_num}"}), on="Team", how="outer"
            )

    # Fill NaN values with 0
    driver_standings = driver_standings.fillna(0)
    constructor_standings = constructor_standings.fillna(0)

    # Calculate total points
    race_columns = [col for col in driver_standings.columns if col.startswith("Race")]
    driver_standings["TotalPoints"] = driver_standings[race_columns].sum(axis=1)

    race_columns = [col for col in constructor_standings.columns if col.startswith("Race")]
    constructor_standings["TotalPoints"] = constructor_standings[race_columns].sum(axis=1)

    # Sort by total points
    driver_standings = driver_standings.sort_values("TotalPoints", ascending=False)
    constructor_standings = constructor_standings.sort_values("TotalPoints", ascending=False)

    return driver_standings, constructor_standings


def format_standings(standings, title):
    """Format standings for display"""
    print(f"\n=== {title} ===\n")

    if "Driver" in standings.columns:
        # Driver standings
        print(f"{'POS':<4}{'DRIVER':<20}{'TEAM':<25}{'POINTS':<8}")
    else:
        # Constructor standings
        print(f"{'POS':<4}{'CONSTRUCTOR':<25}{'POINTS':<8}")

    print("-" * 45)

    for i, (_, row) in enumerate(standings.iterrows(), 1):
        if "Driver" in standings.columns:
            print(f"{i:<4}{row['Driver']:<20}{row['Team']:<25}{int(row['TotalPoints']):<8}")
        else:
            print(f"{i:<4}{row['Team']:<25}{int(row['TotalPoints']):<8}")


def main():
    """Main function to predict entire season and calculate championships"""
    year = 2024
    total_races = 24  # Total races in 2024 season

    logger.info(f"Starting predictions for {year} season")

    # Store results for each race
    all_results = {}

    # Predict each race
    for race in range(1, total_races + 1):
        logger.info(f"Predicting race {race}")
        results = run_race_prediction(year, race)
        if results is not None:
            all_results[race] = results

    # Calculate championship standings
    driver_standings, constructor_standings = calculate_championship_standings(all_results)

    # Display standings
    format_standings(driver_standings, "DRIVER CHAMPIONSHIP STANDINGS")
    format_standings(constructor_standings, "CONSTRUCTOR CHAMPIONSHIP STANDINGS")

    # Save standings to CSV
    driver_standings.to_csv(f"results/driver_standings_{year}.csv", index=False)
    constructor_standings.to_csv(f"results/constructor_standings_{year}.csv", index=False)

    logger.info("Season predictions completed")


if __name__ == "__main__":
    main()

import fastf1
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("f1_prediction.log"), logging.StreamHandler()],
)
logger = logging.getLogger("F1Predictor")

# Create cache directory if it doesn't exist
cache_dir = "cache"
if not os.path.exists(cache_dir):
    logger.info(f"Creating cache directory: {cache_dir}")
    os.makedirs(cache_dir)

# Enable caching for faster data loading
fastf1.Cache.enable_cache(cache_dir)


class F1DataProcessor:
    def __init__(self, current_year=None, historical_years=None):
        """
        Initialize the data processor with configuration for data collection

        Args:
            current_year: The year for prediction (default: current calendar year)
            historical_years: List of years to collect historical data from (default: 3 previous years)
        """
        # Set current year to current calendar year if not specified
        self.current_year = current_year or datetime.now().year

        # Set historical years to last 3 years if not specified
        if historical_years is None:
            self.historical_years = list(range(self.current_year - 3, self.current_year))
        else:
            self.historical_years = historical_years

        logger.info(
            f"Initialized F1DataProcessor for {self.current_year} with historical data from {self.historical_years}"
        )

        # Team name mapping for standardization
        self.team_name_mapping = {
            "Racing Bulls": "Visa Cash App Racing Bulls F1 Team",
            "Visa Cash App RB": "Visa Cash App Racing Bulls F1 Team",
            "RB": "Visa Cash App Racing Bulls F1 Team",
            "AlphaTauri": "Visa Cash App Racing Bulls F1 Team",
            "Toro Rosso": "Visa Cash App Racing Bulls F1 Team",
            "Alfa Romeo": "Kick Sauber",
            "Sauber": "Kick Sauber",
            "Force India": "Aston Martin",
            "Racing Point": "Aston Martin",
            "Renault": "Alpine",
        }

        # Enhanced track similarity mapping to weight historical data
        self.track_similarity = {
            "Bahrain": ["Bahrain", "Saudi Arabia", "Abu Dhabi"],
            "Saudi Arabia": ["Saudi Arabia", "Baku", "Jeddah", "Miami"],
            "Australia": ["Australia", "Hungary", "Spain"],
            "Japan": ["Japan", "UK", "Belgium", "Spain"],
            "China": ["China", "Bahrain", "Zandvoort", "Brazil"],
            "Miami": ["Miami", "Jeddah", "Monaco", "Singapore"],
            "Emilia Romagna": ["Emilia Romagna", "Monaco", "Canada", "Austria"],
            "Monaco": ["Monaco", "Singapore", "Hungary", "Canada"],
            "Canada": ["Canada", "Monaco", "Australia", "France"],
            "Spain": ["Spain", "France", "Hungary", "UK"],
            "Austria": ["Austria", "Mexico", "Brazil", "Belgium"],
            "UK": ["UK", "Belgium", "Japan", "Spain"],
            "Hungary": ["Hungary", "Monaco", "Spain", "Australia"],
            "Belgium": ["Belgium", "UK", "Italy", "France"],
            "Netherlands": ["Netherlands", "Zandvoort", "Austria", "Belgium"],
            "Italy": ["Italy", "Belgium", "France", "Spain"],
            "Baku": ["Baku", "Saudi Arabia", "Russia", "Vietnam"],
            "Singapore": ["Singapore", "Monaco", "Abu Dhabi", "Saudi Arabia"],
            "USA": ["USA", "Mexico", "Brazil", "Qatar"],
            "Mexico": ["Mexico", "Brazil", "Austria", "USA"],
            "Brazil": ["Brazil", "Mexico", "USA", "Austria"],
            "Las Vegas": ["Las Vegas", "Baku", "Saudi Arabia", "Italy"],
            "Qatar": ["Qatar", "USA", "Abu Dhabi", "Bahrain"],
            "Abu Dhabi": ["Abu Dhabi", "Bahrain", "Qatar", "Singapore"]
        }

        # Enhanced recency weighting for historical data
        self.recency_weights = {
            0: 1.0,  # Current year
            1: 0.7,  # Previous year
            2: 0.5,  # Two years ago
            3: 0.3,  # Three years ago
            4: 0.2,  # Four years ago
            5: 0.1,  # Five years ago
        }

        # Track characteristic evolution factors (how much a track has changed)
        self.track_evolution_factors = {
            "Bahrain": 1.0,       # Consistent track
            "Saudi Arabia": 0.9,  # Relatively new with changes
            "Australia": 0.85,    # Recent layout changes
            "Miami": 0.85,        # New track with changes
            "China": 0.95,        # Resurfaced recently
            "Monaco": 0.98,       # Very consistent
            "Spain": 0.9,         # Layout changes
            "Canada": 0.95,       # Relatively consistent
            "UK": 0.95,           # Consistent
            "Austria": 0.98,      # Consistent recently
            "Hungary": 0.95,      # Consistent
            "Belgium": 0.9,       # Recent changes
            "Netherlands": 0.9,   # Recently rejoined calendar
            "Singapore": 0.9,     # Layout modifications
            "Japan": 0.98,        # Very consistent
            "Qatar": 0.85,        # New to calendar
            "USA": 0.95,          # Consistent recently
            "Mexico": 0.98,       # Consistent
            "Brazil": 0.95,       # Consistent layout
            "Las Vegas": 0.8,     # New track
            "Abu Dhabi": 0.9,     # Minor changes
        }

        # Store collected data
        self.race_data = {}
        self.qualifying_data = {}
        self.practice_data = {}
        self.driver_info = {}
        self.team_info = {}
        self.track_info = {}

    def load_event_schedule(self, year=None):
        """
        Load the F1 schedule for a given year

        Args:
            year: The year to load schedule for (default: current_year)

        Returns:
            DataFrame with race schedule
        """
        year = year or self.current_year
        try:
            schedule = fastf1.get_event_schedule(year)

            # Ensure consistent column names
            if schedule is not None:
                # Check if RoundNumber exists, if not create it from RoundNumber or round
                if "RoundNumber" not in schedule.columns:
                    if "round" in schedule.columns:
                        schedule["RoundNumber"] = schedule["round"]
                    elif "Round" in schedule.columns:
                        schedule["RoundNumber"] = schedule["Round"]
                    else:
                        # Create round numbers based on index
                        schedule["RoundNumber"] = range(1, len(schedule) + 1)

                # Ensure EventDate exists
                if "EventDate" not in schedule.columns:
                    if "date" in schedule.columns:
                        schedule["EventDate"] = pd.to_datetime(schedule["date"])
                    elif "Date" in schedule.columns:
                        schedule["EventDate"] = pd.to_datetime(schedule["Date"])
                    else:
                        # Use current date as fallback
                        schedule["EventDate"] = datetime.now()

                # Ensure EventName exists
                if "EventName" not in schedule.columns:
                    if "name" in schedule.columns:
                        schedule["EventName"] = schedule["name"]
                    elif "Name" in schedule.columns:
                        schedule["EventName"] = schedule["Name"]
                    elif "Location" in schedule.columns:
                        schedule["EventName"] = schedule["Location"] + " Grand Prix"
                    else:
                        schedule["EventName"] = [
                            f"Round {i} Grand Prix" for i in schedule["RoundNumber"]
                        ]

                # Log the column names for debugging
                logger.info(f"Schedule columns: {schedule.columns.tolist()}")

            return schedule
        except Exception as e:
            logger.error(f"Error loading {year} schedule: {e}")
            return None

    def load_session_data(self, year, gp_round, session_type):
        """
        Load session data from FastF1 API

        Args:
            year: F1 season year
            gp_round: Grand Prix round number or name
            session_type: Session type (e.g., "R" for race, "Q" for qualifying, "FP1", "FP2", "FP3")

        Returns:
            fastf1.core.Session object
        """
        try:
            session = fastf1.get_session(year, gp_round, session_type)
            session.load()
            logger.info(f"Successfully loaded {year} {gp_round} {session_type} data")
            return session
        except Exception as e:
            logger.error(f"Error loading session data: {e}")
            return None

    def process_lap_data(self, session):
        """
        Process lap data from a session

        Args:
            session: fastf1.core.Session object

        Returns:
            DataFrame with processed lap data
        """
        if not session:
            return None

        try:
            # Extract lap data
            laps = session.laps.copy()

            # Basic cleaning and transformation
            laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()

            # Add driver and team info
            drivers = {}
            for _, driver in session.drivers.items():
                try:
                    driver_info = session.get_driver(driver)
                    drivers[driver] = {
                        "FullName": driver_info["FullName"],
                        "Abbreviation": driver_info["Abbreviation"],
                        "TeamName": driver_info["TeamName"],
                    }
                except:
                    # Some older sessions might not have complete driver info
                    pass

            driver_df = pd.DataFrame.from_dict(drivers, orient="index")

            return {
                "laps": laps,
                "drivers": driver_df,
                "session_info": {
                    "EventName": session.event["EventName"],
                    "CircuitName": session.event["CircuitName"],
                    "Country": session.event["Country"],
                    "Year": session.event.year,
                    "Round": session.event.round,
                    "SessionType": session.name,
                    "SessionDate": session.date,
                },
            }
        except Exception as e:
            logger.error(f"Error processing lap data: {e}")
            return None

    def process_qualifying_data(self, session):
        """
        Process qualifying data from a session

        Args:
            session: fastf1.core.Session object

        Returns:
            DataFrame with qualifying results
        """
        if not session:
            return None

        try:
            # Get qualifying results
            quali_results = session.results

            # Extract Q1, Q2, Q3 times
            quali_data = []
            for _, result in quali_results.iterrows():
                driver_id = result["Abbreviation"]

                # Convert time strings to seconds
                q1_time = (
                    pd.to_timedelta(result["Q1"]).total_seconds()
                    if pd.notna(result["Q1"])
                    else None
                )
                q2_time = (
                    pd.to_timedelta(result["Q2"]).total_seconds()
                    if pd.notna(result["Q2"])
                    else None
                )
                q3_time = (
                    pd.to_timedelta(result["Q3"]).total_seconds()
                    if pd.notna(result["Q3"])
                    else None
                )

                # Use the best time (Q3 if available, otherwise Q2, otherwise Q1)
                best_time = (
                    q3_time if pd.notna(q3_time) else (q2_time if pd.notna(q2_time) else q1_time)
                )
                best_session = (
                    "Q3"
                    if pd.notna(q3_time)
                    else ("Q2" if pd.notna(q2_time) else ("Q1" if pd.notna(q1_time) else None))
                )

                quali_data.append(
                    {
                        "DriverId": driver_id,
                        "FullName": result["FullName"],
                        "TeamName": result["TeamName"],
                        "Position": result["Position"],
                        "Q1Time": q1_time,
                        "Q2Time": q2_time,
                        "Q3Time": q3_time,
                        "BestTime": best_time,
                        "BestSession": best_session,
                    }
                )

            return pd.DataFrame(quali_data)
        except Exception as e:
            logger.error(f"Error processing qualifying data: {e}")
            return None

    def _calculate_points(self, position):
        """
        Calculate points based on finishing position

        Args:
            position: Finishing position (1-20)

        Returns:
            int: Points awarded
        """
        points_system = {
            1: 25,  # 1st place
            2: 18,  # 2nd place
            3: 15,  # 3rd place
            4: 12,  # 4th place
            5: 10,  # 5th place
            6: 8,  # 6th place
            7: 6,  # 7th place
            8: 4,  # 8th place
            9: 2,  # 9th place
            10: 1,  # 10th place
        }
        return points_system.get(position, 0)  # 0 points for positions 11-20

    def process_race_data(self, session):
        """
        Process race data from a session

        Args:
            session: fastf1.core.Session object

        Returns:
            DataFrame with race results
        """
        if not session:
            return None

        try:
            # Get race results
            race_results = session.results

            # Get race laps data for pace analysis
            laps_data = session.laps

            # Calculate median lap time excluding first lap and pit laps
            median_lap_times = {}
            for driver in race_results["Abbreviation"]:
                driver_laps = laps_data[laps_data["Driver"] == driver]
                # Filter out first lap and pit in/out laps
                clean_laps = driver_laps[
                    (driver_laps["LapNumber"] > 1)
                    & (~driver_laps["PitInTime"].notna())
                    & (~driver_laps["PitOutTime"].notna())
                ]

                if len(clean_laps) > 0:
                    median_lap_time = clean_laps["LapTime"].dt.total_seconds().median()
                    median_lap_times[driver] = median_lap_time

            # Add median lap times to results
            race_data = []
            for _, result in race_results.iterrows():
                driver_id = result["Abbreviation"]
                position = result["Position"]

                # Calculate points based on position
                points = self._calculate_points(position)

                race_data.append(
                    {
                        "DriverId": driver_id,
                        "FullName": result["FullName"],
                        "TeamName": result["TeamName"],
                        "Position": position,
                        "Status": result["Status"],
                        "Points": points,
                        "GridPosition": result["GridPos"],
                        "MedianLapTime": median_lap_times.get(driver_id),
                        "Finished": result["Status"] == "Finished",
                        "Laps": result["Laps"],
                    }
                )

            return pd.DataFrame(race_data)
        except Exception as e:
            logger.error(f"Error processing race data: {e}")
            return None

    def collect_historical_race_data(
        self, current_year=None, current_round=None, num_races=5, include_practice=False
    ):
        """
        Collect historical race data for the specified circuit with enhanced weighting

        Args:
            current_year: Current year (default: self.current_year)
            current_round: Current race round (default: None)
            num_races: Number of historical races to collect (default: 5)
            include_practice: Whether to include practice session data (default: False)

        Returns:
            Dictionary with historical race data including race, qualifying, and optionally practice data
        """
        try:
            current_year = current_year or self.current_year
            if current_round is None:
                current_round = self.get_upcoming_race()["RoundNumber"]

            # Get circuit name for current race
            current_race = self.get_event_schedule(current_year, current_round)
            if current_race is None or (
                isinstance(current_race, pd.DataFrame) and current_race.empty
            ):
                logger.warning(f"No race found for year {current_year} and round {current_round}")
                return {"race": pd.DataFrame(), "qualifying": pd.DataFrame(), "practice": pd.DataFrame()}

            # Get circuit name from available columns with better logging
            circuit_name = None
            available_columns = (
                current_race.columns.tolist() if isinstance(current_race, pd.DataFrame) else []
            )
            logger.debug(f"Available columns for circuit name: {available_columns}")

            # Try different column names in order of preference
            column_priorities = ["CircuitName", "Location", "Circuit", "EventName"]
            for col in column_priorities:
                if col in available_columns:
                    circuit_name = current_race[col].iloc[0]
                    logger.info(f"Found circuit name '{circuit_name}' in column '{col}'")
                    break

            if circuit_name is None:
                logger.warning(
                    f"Could not find circuit name in available columns: {available_columns}"
                )
                circuit_name = f"Round {current_round} Circuit"
                logger.info(f"Using default circuit name: {circuit_name}")

            # Get similar tracks for enhanced data collection
            similar_tracks = self.track_similarity.get(circuit_name, [circuit_name])
            
            # Collect historical race data
            historical_data = {"race": pd.DataFrame(), "qualifying": pd.DataFrame(), "practice": pd.DataFrame()}
            
            # Extend historical years search range to get more data for new tracks
            search_years = sorted(self.historical_years + [current_year-4, current_year-5] 
                                if circuit_name in ["Miami", "Las Vegas", "Qatar", "Saudi Arabia"] 
                                else self.historical_years,
                               reverse=True)

            collected_races = 0
            race_weights = []
            
            # Enhanced data collection strategy - prioritize exact track matches first
            for year in search_years:
                years_ago = current_year - year
                
                # Apply recency weighting
                recency_weight = self.recency_weights.get(years_ago, 0.1)
                
                # First prioritize exact track matches
                try:
                    year_schedule = self.get_season_schedule(year)
                    if year_schedule is None or year_schedule.empty:
                        logger.debug(f"No schedule found for year {year}")
                        continue

                    # Try to match exact circuit name
                    circuit_race = None
                    for col in column_priorities:
                        if col in year_schedule.columns:
                            circuit_race = year_schedule[
                                year_schedule[col].str.contains(circuit_name, case=False, na=False)
                            ]
                            if not circuit_race.empty:
                                logger.debug(f"Found exact matching race in {year} using column {col}")
                                
                                # Apply track evolution factor (how much the track has changed over time)
                                track_evolution = self.track_evolution_factors.get(circuit_name, 0.9)
                                adjusted_weight = recency_weight * (track_evolution ** years_ago)
                                
                                # Process race data with weight
                                race_round = circuit_race["RoundNumber"].iloc[0]
                                self._process_historical_race(historical_data, year, race_round, 
                                                             include_practice, adjusted_weight, 1.0)
                                
                                collected_races += 1
                                race_weights.append(adjusted_weight)
                                break
                except Exception as e:
                    logger.warning(f"Error loading exact track data for {year} {circuit_name}: {e}")
                    continue
            
            # If we didn't get enough data, look for similar tracks
            if collected_races < num_races:
                for similar_track in similar_tracks[1:]:  # Skip the first one (same as circuit_name)
                    for year in search_years:
                        # Skip if we have enough data
                        if collected_races >= num_races:
                            break
                            
                        years_ago = current_year - year
                        recency_weight = self.recency_weights.get(years_ago, 0.1)
                        
                        try:
                            year_schedule = self.get_season_schedule(year)
                            if year_schedule is None or year_schedule.empty:
                                continue

                            # Try to match similar circuit name
                            circuit_race = None
                            for col in column_priorities:
                                if col in year_schedule.columns:
                                    circuit_race = year_schedule[
                                        year_schedule[col].str.contains(similar_track, case=False, na=False)
                                    ]
                                    if not circuit_race.empty:
                                        # Apply track evolution and similarity factor (reduce weight for similar tracks)
                                        similarity_index = similar_tracks.index(similar_track) if similar_track in similar_tracks else len(similar_tracks)
                                        similarity_factor = max(0.4, 1.0 - (similarity_index * 0.2))  # 0.8, 0.6, 0.4 for similar tracks
                                        
                                        track_evolution = self.track_evolution_factors.get(similar_track, 0.9)
                                        adjusted_weight = recency_weight * similarity_factor * (track_evolution ** years_ago)
                                        
                                        # Process race data with weight
                                        race_round = circuit_race["RoundNumber"].iloc[0]
                                        self._process_historical_race(historical_data, year, race_round, 
                                                                     include_practice, adjusted_weight, similarity_factor)
                                        
                                        collected_races += 1
                                        race_weights.append(adjusted_weight)
                                        break
                        except Exception as e:
                            logger.warning(f"Error loading similar track data for {year} {similar_track}: {e}")
                            continue
            
            # Normalize weights for final DataFrame
            if race_weights and len(race_weights) > 0:
                weight_sum = sum(race_weights)
                normalized_weights = [w/weight_sum for w in race_weights]
                
                # Store the weights in the DataFrames for later use
                if not historical_data["race"].empty:
                    historical_data["race"]["DataWeight"] = normalized_weights[:len(historical_data["race"])]
                
                if not historical_data["qualifying"].empty:
                    historical_data["qualifying"]["DataWeight"] = normalized_weights[:len(historical_data["qualifying"])]

            logger.info(
                f"Collected historical data: {len(historical_data['race'])} races, {len(historical_data['practice'])} practice sessions"
            )
            return historical_data

        except Exception as e:
            logger.error(f"Error collecting historical race data: {e}")
            return {"race": pd.DataFrame(), "qualifying": pd.DataFrame(), "practice": pd.DataFrame()}
            
    def _process_historical_race(self, historical_data, year, race_round, include_practice, weight_factor, similarity_factor):
        """Helper method to process historical race data with weights"""
        try:
            # Load race data
            race_data = self.load_session_data(year, race_round, "Race")
            if race_data is not None:
                race_results = self.process_race_data(race_data)
                if not race_results.empty:
                    race_results["Year"] = year
                    race_results["Round"] = race_round
                    race_results["WeightFactor"] = weight_factor
                    race_results["SimilarityFactor"] = similarity_factor
                    historical_data["race"] = pd.concat([historical_data["race"], race_results])
                    logger.debug(f"Added race data from {year} with weight {weight_factor:.2f}")
                    
            # Load qualifying data
            qualifying_data = self.load_session_data(year, race_round, "Qualifying")
            if qualifying_data is not None:
                try:
                    qualifying_results = self.process_qualifying_data(qualifying_data)
                    if qualifying_results is not None and not qualifying_results.empty:
                        qualifying_results["Year"] = year
                        qualifying_results["Round"] = race_round
                        qualifying_results["WeightFactor"] = weight_factor
                        qualifying_results["SimilarityFactor"] = similarity_factor
                        historical_data["qualifying"] = pd.concat(
                            [historical_data["qualifying"], qualifying_results]
                        )
                        logger.debug(f"Added qualifying data from {year} with weight {weight_factor:.2f}")
                except Exception as e:
                    logger.warning(f"Error processing qualifying data from {year}: {e}")
                    # Create a minimal qualifying DataFrame if processing fails
                    if race_results is not None and not race_results.empty:
                        min_qual = race_results[["DriverId", "FullName", "TeamName"]].copy()
                        min_qual["Position"] = range(1, len(min_qual) + 1)
                        min_qual["Year"] = year
                        min_qual["Round"] = race_round
                        min_qual["WeightFactor"] = weight_factor
                        min_qual["SimilarityFactor"] = similarity_factor
                        historical_data["qualifying"] = pd.concat(
                            [historical_data["qualifying"], min_qual]
                        )
                        logger.debug(f"Added minimal qualifying data from {year}")

            # Load practice data if requested
            if include_practice:
                for session in ["Practice 1", "Practice 2", "Practice 3"]:
                    practice_data = self.load_session_data(year, race_round, session)
                    if practice_data is not None:
                        practice_results = self.process_lap_data(practice_data)
                        if (
                            practice_results is not None
                            and "laps" in practice_results
                            and not practice_results["laps"].empty
                        ):
                            practice_results["laps"]["Year"] = year
                            practice_results["laps"]["Round"] = race_round
                            practice_results["laps"]["Session"] = session
                            practice_results["laps"]["WeightFactor"] = weight_factor
                            practice_results["laps"]["SimilarityFactor"] = similarity_factor
                            historical_data["practice"] = pd.concat(
                                [historical_data["practice"], practice_results["laps"]]
                            )
                            logger.debug(f"Added {session} data from {year}")
                            
            return True
        except Exception as e:
            logger.warning(f"Error in _process_historical_race for {year} round {race_round}: {e}")
            return False

    def collect_current_event_data(self, gp_name=None, gp_round=None):
        """
        Collect current event data for prediction

        Args:
            gp_name: Grand Prix name
            gp_round: Grand Prix round number

        Returns:
            Dictionary with collected data for the current event
        """
        # Either gp_name or gp_round must be provided
        if gp_name is None and gp_round is None:
            logger.error("Either gp_name or gp_round must be provided")
            return None

        try:
            # Get event schedule for the current year
            schedule = self.load_event_schedule()
            if schedule is None:
                return None

            # Find the event
            if gp_name:
                event = schedule[schedule["EventName"].str.contains(gp_name, case=False)]
            else:
                event = schedule[schedule["RoundNumber"] == gp_round]

            if len(event) == 0:
                logger.error(
                    f"Event not found for {self.current_year} with name/round: {gp_name or gp_round}"
                )
                return None

            event = event.iloc[0]
            gp_name = event["EventName"]
            gp_round = event["RoundNumber"]

            logger.info(f"Collecting data for {self.current_year} {gp_name} (Round {gp_round})...")

            # Collect data from all available sessions
            current_data = {}

            # Practice sessions
            for practice in ["FP1", "FP2", "FP3"]:
                session = self.load_session_data(self.current_year, gp_round, practice)
                if session:
                    current_data[practice] = self.process_lap_data(session)

            # Qualifying
            quali_session = self.load_session_data(self.current_year, gp_round, "Q")
            if quali_session:
                current_data["qualifying"] = self.process_qualifying_data(quali_session)

            # Store event information
            current_data["event_info"] = {
                "EventName": gp_name,
                "Round": gp_round,
                "CircuitName": event["CircuitName"],
                "Country": event["Country"],
                "Year": self.current_year,
            }

            return current_data

        except Exception as e:
            logger.error(f"Error collecting current event data: {e}")
            return None

    def get_driver_standings(self, year=None):
        """
        Get driver standings for a specific year

        Args:
            year: Year to get standings for (default: current year)

        Returns:
            DataFrame with driver standings
        """
        year = year or self.current_year
        try:
            standings = fastf1.get_driver_standings(year)
            return standings
        except Exception as e:
            logger.error(f"Error getting driver standings for {year}: {e}")
            return None

    def get_constructor_standings(self, year=None):
        """
        Get constructor standings for a specific year

        Args:
            year: Year to get standings for (default: current year)

        Returns:
            DataFrame with constructor standings
        """
        year = year or self.current_year
        try:
            standings = fastf1.get_constructor_standings(year)
            return standings
        except Exception as e:
            logger.error(f"Error getting constructor standings for {year}: {e}")
            return None

    def collect_driver_statistics(self):
        """
        Collect comprehensive statistics for all drivers

        Returns:
            DataFrame with driver statistics
        """
        all_stats = []

        for year in self.historical_years + [self.current_year]:
            logger.info(f"Collecting driver statistics for {year}...")

            # Get standings for the year
            standings = self.get_driver_standings(year)
            if standings is None:
                continue

            # Get schedule for the year
            schedule = self.load_event_schedule(year)
            if schedule is None:
                continue

            # Process each driver
            for _, driver in standings.iterrows():
                driver_id = driver["Abbreviation"]
                team = driver["TeamName"]

                # Collect race results for the driver
                driver_races = []
                for _, event in schedule.iterrows():
                    gp_round = event["RoundNumber"]

                    # Skip future events
                    event_date = event["EventDate"]
                    if event_date > datetime.now():
                        continue

                    # Get race session
                    race_session = self.load_session_data(year, gp_round, "R")
                    if race_session:
                        # Find driver's result
                        try:
                            result = race_session.results[
                                race_session.results["Abbreviation"] == driver_id
                            ]
                            if len(result) > 0:
                                driver_races.append(
                                    {
                                        "EventName": event["EventName"],
                                        "CircuitName": event["CircuitName"],
                                        "Position": result.iloc[0]["Position"],
                                        "Status": result.iloc[0]["Status"],
                                        "GridPosition": result.iloc[0]["GridPos"],
                                    }
                                )
                        except:
                            # Skip if driver didn't participate
                            pass

                # Calculate statistics
                if driver_races:
                    races_df = pd.DataFrame(driver_races)

                    # Count race finishes and DNFs
                    finished_races = races_df[races_df["Status"] == "Finished"].shape[0]
                    dnf_races = races_df[races_df["Status"] != "Finished"].shape[0]

                    # Calculate average finishing and grid positions
                    finished_positions = races_df[races_df["Status"] == "Finished"]["Position"]
                    average_finish = (
                        finished_positions.mean() if len(finished_positions) > 0 else None
                    )

                    grid_positions = races_df["GridPosition"].astype(float)
                    average_grid = grid_positions.mean() if len(grid_positions) > 0 else None

                    # Calculate position improvement from grid to finish
                    position_improvements = []
                    for _, race in races_df.iterrows():
                        if race["Status"] == "Finished" and not pd.isna(race["GridPosition"]):
                            improvement = float(race["GridPosition"]) - float(race["Position"])
                            position_improvements.append(improvement)

                    average_improvement = (
                        np.mean(position_improvements) if position_improvements else 0
                    )

                    # Add to statistics
                    all_stats.append(
                        {
                            "Year": year,
                            "DriverId": driver_id,
                            "FullName": driver["FullName"],
                            "TeamName": team,
                            "Points": driver["Points"],
                            "Position": driver["Position"],
                            "RacesCompleted": len(driver_races),
                            "FinishedRaces": finished_races,
                            "DNFs": dnf_races,
                            "DNFRate": (
                                dnf_races / len(driver_races) if len(driver_races) > 0 else 0
                            ),
                            "AverageFinish": average_finish,
                            "AverageGrid": average_grid,
                            "AverageImprovement": average_improvement,
                        }
                    )

        return pd.DataFrame(all_stats)

    def export_processed_data(self, output_dir="data"):
        """
        Export all processed data to CSV files

        Args:
            output_dir: Directory to save data to
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export race data
        if hasattr(self, "race_data") and self.race_data:
            for race_type, data in self.race_data.items():
                if isinstance(data, pd.DataFrame) and not data.empty:
                    filepath = os.path.join(output_dir, f"race_{race_type}_{timestamp}.csv")
                    data.to_csv(filepath, index=False)
                    logger.info(f"Exported race data to {filepath}")

        # Export qualifying data
        if hasattr(self, "qualifying_data") and self.qualifying_data:
            for quali_type, data in self.qualifying_data.items():
                if isinstance(data, pd.DataFrame) and not data.empty:
                    filepath = os.path.join(output_dir, f"quali_{quali_type}_{timestamp}.csv")
                    data.to_csv(filepath, index=False)
                    logger.info(f"Exported qualifying data to {filepath}")

        # Export driver info
        if (
            hasattr(self, "driver_info")
            and isinstance(self.driver_info, pd.DataFrame)
            and not self.driver_info.empty
        ):
            filepath = os.path.join(output_dir, f"drivers_{timestamp}.csv")
            self.driver_info.to_csv(filepath, index=False)
            logger.info(f"Exported driver info to {filepath}")

        # Export team info
        if (
            hasattr(self, "team_info")
            and isinstance(self.team_info, pd.DataFrame)
            and not self.team_info.empty
        ):
            filepath = os.path.join(output_dir, f"teams_{timestamp}.csv")
            self.team_info.to_csv(filepath, index=False)
            logger.info(f"Exported team info to {filepath}")

    def prepare_prediction_dataset(self, circuit_name, include_new_drivers=True):
        """
        Prepare a complete dataset for race prediction at a specific circuit

        Args:
            circuit_name: Name of the circuit for prediction
            include_new_drivers: Whether to include drivers without historical data

        Returns:
            Dictionary with processed data ready for prediction
        """
        # Collect historical data for the circuit
        historical_data = self.collect_historical_race_data()

        # Collect current event data
        current_data = self.collect_current_event_data(gp_name=circuit_name)

        # Collect driver statistics
        driver_stats = self.collect_driver_statistics()

        # Combine data
        prediction_data = {
            "historical": historical_data,
            "current": current_data,
            "driver_stats": driver_stats,
            "circuit_name": circuit_name,
        }

        # Get current driver and constructor standings
        prediction_data["driver_standings"] = self.get_driver_standings()
        prediction_data["constructor_standings"] = self.get_constructor_standings()

        return prediction_data

    def get_upcoming_race(self):
        """
        Get the next upcoming race

        Returns:
            Dictionary with upcoming race information
        """
        try:
            schedule = self.load_event_schedule()
            if schedule is None:
                return None

            # Find the next race after current date
            upcoming_races = schedule[schedule["EventDate"] > datetime.now()]
            if upcoming_races.empty:
                return None

            # Return the next race
            return upcoming_races.iloc[0].to_dict()
        except Exception as e:
            logger.error(f"Error getting upcoming race: {e}")
            return None

    def get_season_schedule(self, year=None):
        """
        Get the season schedule for a given year

        Args:
            year: Year to get schedule for (default: current_year)

        Returns:
            DataFrame with season schedule
        """
        year = year or self.current_year
        return self.load_event_schedule(year)

    def get_event_schedule(self, year=None, round=None):
        """
        Get event schedule for a specific round

        Args:
            year: Year to get schedule for
            round: Round number

        Returns:
            DataFrame with event schedule
        """
        try:
            schedule = self.load_event_schedule(year)
            if schedule is None:
                return None

            if round is not None:
                event = schedule[schedule["RoundNumber"] == round]
                if event.empty:
                    return None

                # Return event sessions
                event_name = event.iloc[0]["EventName"]
                logger.info(f"Getting session schedule for {year} {event_name}")

                # Here you would normally get detailed session info
                # For now, return a simplified schedule
                event_date = event.iloc[0]["EventDate"]
                sessions = pd.DataFrame(
                    [
                        {"Session": "Practice 1", "SessionStart": event_date - timedelta(days=2)},
                        {
                            "Session": "Practice 2",
                            "SessionStart": event_date - timedelta(days=2, hours=-5),
                        },
                        {
                            "Session": "Practice 3",
                            "SessionStart": event_date - timedelta(days=1, hours=4),
                        },
                        {"Session": "Qualifying", "SessionStart": event_date - timedelta(days=1)},
                        {"Session": "Race", "SessionStart": event_date},
                    ]
                )
                return sessions
        except Exception as e:
            logger.error(f"Error getting event schedule: {e}")
            return None

    def load_qualifying_data(self, year, round):
        """
        Load qualifying data for a specific race

        Args:
            year: Year
            round: Round number

        Returns:
            DataFrame with qualifying data
        """
        try:
            # Get qualifying session
            quali_session = self.load_session_data(year, round, "Q")
            if quali_session is None:
                return None

            # Process qualifying data
            return self.process_qualifying_data(quali_session)
        except Exception as e:
            logger.error(f"Error loading qualifying data: {e}")
            return None

    def get_track_info(self, year, round):
        """
        Get track information for a specific race

        Args:
            year: Year
            round: Round number

        Returns:
            Dictionary with track information
        """
        try:
            schedule = self.load_event_schedule(year)
            if schedule is None:
                logger.warning(f"No schedule found for year {year}")
                return self._create_default_track_info(round)

            event = schedule[schedule["RoundNumber"] == round]
            if event.empty:
                logger.warning(f"No event found for round {round} in year {year}")
                return self._create_default_track_info(round)

            # Determine the circuit name from available columns
            circuit_name = None
            if "CircuitName" in event.columns:
                circuit_name = event.iloc[0]["CircuitName"]
            elif "Circuit" in event.columns:
                circuit_name = event.iloc[0]["Circuit"]
            elif "Location" in event.columns:
                circuit_name = event.iloc[0]["Location"]
            elif "EventName" in event.columns:
                circuit_name = event.iloc[0]["EventName"]
            else:
                circuit_name = f"Round {round} Circuit"

            # Return basic track info
            track_info = {
                "Name": circuit_name,
                "Length": 5.0,  # Default value
                "Laps": 58,  # Default value
            }

            # Try to get more detailed info if available
            if "CircuitLength" in event.columns:
                track_info["Length"] = event.iloc[0]["CircuitLength"]

            if "Laps" in event.columns:
                track_info["Laps"] = event.iloc[0]["Laps"]

            return track_info
        except Exception as e:
            logger.error(f"Error getting track info: {e}")
            return self._create_default_track_info(round)

    def standardize_team_name(self, team_name):
        """Standardize team names using the mapping.

        Args:
            team_name: The team name to standardize

        Returns:
            The standardized team name
        """
        return self.team_name_mapping.get(team_name, team_name)

    def _create_default_track_info(self, round):
        """
        Create default track info when actual data is not available

        Args:
            round: Round number

        Returns:
            Dictionary with default track information
        """
        return {
            "Name": f"Round {round} Circuit",
            "Length": 5.0,  # Default value in km
            "Laps": 58,  # Default value
        }

    def process_actual_race_data(self, year: int, race: int) -> pd.DataFrame:
        """Process actual race data for a specific year and race."""
        # First, get the qualifying data to get grid positions
        quali_session = self.load_session_data(year, race, 'Q')
        if quali_session is None:
            print("Could not load qualifying data")
            return None

        # Get race results
        race_session = self.load_session_data(year, race, 'R')
        if race_session is None:
            print("Could not load race data")
            return None

        # Get the results DataFrames
        quali_data = quali_session.results
        race_data = race_session.results

        print("Race DataFrame columns:", race_data.columns.tolist())
        print("\nRace DataFrame head:")
        print(race_data.head())

        # Create a mapping of driver numbers to their qualifying positions
        quali_positions = {}
        if 'DriverNumber' in quali_data.columns and 'Position' in quali_data.columns:
            for _, row in quali_data.iterrows():
                quali_positions[str(row['DriverNumber'])] = row['Position']

        # Create a new DataFrame with the required columns
        result_df = pd.DataFrame()
        
        # Add required columns from race data
        columns_to_copy = ['DriverNumber', 'Position', 'Status', 'Points', 'FullName', 'TeamName']
        for col in columns_to_copy:
            if col in race_data.columns:
                result_df[col] = race_data[col]

        # Rename columns to match expected format
        result_df = result_df.rename(columns={'FullName': 'Driver', 'TeamName': 'Team'})

        # Add grid positions from qualifying data
        result_df['GridPosition'] = result_df['DriverNumber'].astype(str).map(quali_positions)

        # Convert numeric columns
        numeric_columns = ['Position', 'GridPosition', 'Points']
        for col in numeric_columns:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')

        return result_df

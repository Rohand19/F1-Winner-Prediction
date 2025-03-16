Data Processing
==============

The F1 Race Predictor uses historical Formula 1 race data to make predictions. This section describes the data sources, processing pipeline, and data structures used in the project.

Data Sources
-----------

The project uses the FastF1 API to fetch official Formula 1 data, including:

* Race results
* Qualifying results
* Practice session data
* Track information
* Weather conditions
* Lap times
* Car telemetry

Data Processing Pipeline
----------------------

.. code-block:: python

   from f1predictor.data_processor import F1DataProcessor

   # Initialize processor
   processor = F1DataProcessor(current_year=2024)

   # Get track information
   track_info = processor.get_track_info(year=2024, race=1)

   # Load qualifying data
   qualifying_data = processor.load_qualifying_data(year=2024, race=1)

   # Collect historical race data
   historical_data = processor.collect_historical_race_data(
       current_year=2024,
       current_round=1,
       num_races=5
   )

Data Structures
--------------

Track Information
~~~~~~~~~~~~~~~~

Track information is returned as a dictionary containing:

* Track name
* Circuit length
* Number of laps
* Track characteristics

Race Data
~~~~~~~~~

Race data is stored in pandas DataFrames with the following key columns:

* Driver
* Team
* Grid position
* Finish position
* Points
* Status
* Fastest lap
* Race time

Qualifying Data
~~~~~~~~~~~~~

Qualifying data includes:

* Q1, Q2, Q3 times
* Final grid position
* Sector times
* Compound used

Data Caching
-----------

To improve performance and reduce API calls, the data processor implements caching:

* Race results are cached locally
* Cache is invalidated after configurable time
* Cache location is configurable via environment variables

Example cache configuration:

.. code-block:: python

   import os
   os.environ['F1_CACHE_DIR'] = '/path/to/cache'
   
   processor = F1DataProcessor(
       cache_enabled=True,
       cache_expiry_days=7
   )

Data Validation
-------------

The data processor includes several validation steps:

* Check for missing values
* Validate date ranges
* Verify data consistency
* Handle edge cases (e.g., cancelled races)

Example validation:

.. code-block:: python

   # Validate race data
   processor.validate_race_data(race_data)

   # Check for missing qualifying data
   if processor.is_qualifying_data_complete(qualifying_data):
       # Process data
       pass 
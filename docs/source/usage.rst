Usage Guide
===========

This guide covers the main use cases and examples for the F1 Race Predictor.

Command Line Interface
-------------------

The F1 Race Predictor can be used directly from the command line:

.. code-block:: bash

   # Make a prediction for a specific race
   f1predictor --year 2025 --race 1 --model-type xgboost --visualize

   # List available models
   f1predictor --list-models

   # Get help
   f1predictor --help

Docker Usage
----------

Using Docker is the recommended way to run F1 Race Predictor:

.. code-block:: bash

   # Build the Docker image
   docker build -t f1predictor .

   # Run a prediction
   docker run -v "${PWD}/data:/app/data" \
             -v "${PWD}/models:/app/models" \
             -v "${PWD}/results:/app/results" \
             -v "${PWD}/cache:/app/cache" \
             f1predictor --year 2025 --race 1 --model-type xgboost --visualize

Python API
---------

You can also use F1 Race Predictor as a Python library:

.. code-block:: python

   from f1predictor import RacePredictor
   from f1predictor.data_processor import F1DataProcessor

   # Initialize the predictor
   predictor = RacePredictor(model_type='xgboost')

   # Load and process data
   data_processor = F1DataProcessor()
   data = data_processor.load_data()

   # Make predictions
   predictions = predictor.predict(year=2024, race=1)

   # Print results
   predictor.print_race_results(predictions)

Configuration
------------

The F1 Race Predictor can be configured using environment variables or a configuration file:

.. code-block:: bash

   # Environment variables
   export F1_DATA_DIR=/path/to/data
   export F1_MODEL_TYPE=xgboost
   export F1_CACHE_DIR=/path/to/cache

   # Configuration file (config.yaml)
   data_dir: /path/to/data
   model_type: xgboost
   cache_dir: /path/to/cache

Model Types
----------

The following model types are supported:

* XGBoost (default)
* Random Forest
* Neural Network
* Ensemble

Each model type has its own strengths and can be selected based on your needs:

.. code-block:: bash

   # Using XGBoost
   f1predictor --model-type xgboost

   # Using Random Forest
   f1predictor --model-type random_forest

   # Using Neural Network
   f1predictor --model-type neural_network

   # Using Ensemble
   f1predictor --model-type ensemble

Visualization
-----------

The predictor can generate various visualizations:

* Prediction probability distribution
* Feature importance plots
* Historical performance comparison
* Driver performance trends

Enable visualization with the ``--visualize`` flag:

.. code-block:: bash

   f1predictor --year 2024 --race 1 --model-type xgboost --visualize

The visualizations will be saved in the ``results`` directory. 
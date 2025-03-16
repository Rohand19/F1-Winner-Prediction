F1 Race Predictor Documentation
============================

Welcome to F1 Race Predictor's documentation! This project provides machine learning-based predictions for Formula 1 race outcomes.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   models
   data
   contributing
   changelog

Features
--------

* Predict Formula 1 race winners using machine learning models
* Support for multiple model types (XGBoost, etc.)
* Historical F1 race data processing
* Visualization of predictions and results
* Docker support for easy deployment
* Comprehensive test suite

Installation
-----------

To install F1 Race Predictor, run:

.. code-block:: bash

   git clone https://github.com/Rohand19/f1-race-predictor.git
   cd f1-race-predictor
   pip install -e .

For development installation with all extras:

.. code-block:: bash

   pip install -e ".[dev,docs,test]"

Quick Start
----------

To make a prediction for an upcoming race:

.. code-block:: bash

   docker run -v "${PWD}/data:/app/data" \
             -v "${PWD}/models:/app/models" \
             -v "${PWD}/results:/app/results" \
             -v "${PWD}/cache:/app/cache" \
             f1predictor --year 2025 --race 1 --model-type xgboost --visualize

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 
Contributing
============

We love your input! We want to make contributing to F1 Race Predictor as easy and transparent as possible.

Development Setup
---------------

1. Fork the repository
2. Clone your fork:

   .. code-block:: bash

      git clone https://github.com/Rohand19/f1-race-predictor.git
      cd f1-race-predictor

3. Install development dependencies:

   .. code-block:: bash

      pip install -e ".[dev,docs,test]"

4. Create a branch for your feature:

   .. code-block:: bash

      git checkout -b feature-name

Code Style
---------

We use several tools to maintain code quality:

* ``black`` for code formatting
* ``isort`` for import sorting
* ``flake8`` for style guide enforcement
* ``mypy`` for type checking

Run the following before committing:

.. code-block:: bash

   # Format code
   black src/f1predictor tests
   isort src/f1predictor tests

   # Check style
   flake8 src/f1predictor tests
   mypy src/f1predictor

Testing
-------

We use pytest for testing. Run the test suite:

.. code-block:: bash

   pytest tests/ --cov=f1predictor

All new features should include tests.

Documentation
------------

We use Sphinx for documentation. Build the docs locally:

.. code-block:: bash

   cd docs
   make html

View the docs at ``build/html/index.html``.

Pull Request Process
------------------

1. Update the documentation.
2. Update the changelog.
3. Ensure CI passes.
4. Get approval from maintainers.

Code of Conduct
-------------

Please note that this project is released with a Code of Conduct. By participating in this project you agree to abide by its terms.

* Be respectful and inclusive
* Accept constructive criticism
* Focus on what is best for the community
* Show empathy towards others 
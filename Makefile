.PHONY: install test lint format clean build docker-build docker-run

install:
	pip install -e ".[dev]"

test:
	pytest tests/ --cov=f1predictor --cov-report=term-missing

lint:
	flake8 src/f1predictor tests
	mypy src/f1predictor tests
	black --check src/f1predictor tests
	isort --check-only src/f1predictor tests

format:
	black src/f1predictor tests
	isort src/f1predictor tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

build:
	python setup.py sdist bdist_wheel

docker-build:
	docker build -t f1predictor .

docker-run:
	docker run -v $(PWD)/data:/app/data -v $(PWD)/models:/app/models -v $(PWD)/results:/app/results f1predictor 
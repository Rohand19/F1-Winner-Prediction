from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="f1predictor",
    version="1.0.0",
    author="Rohan Divakar",
    author_email="rohanb2000@gmail.com",
    description="A comprehensive Formula 1 race prediction system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rohand19/f1-race-prediction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.19.0",
        "fastf1>=3.0.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "xgboost>=1.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "myst-parser",
            "sphinx-autodoc-typehints",
        ],
    },
    entry_points={
        "console_scripts": [
            "f1predict=f1predictor.scripts.main_predictor:main",
            "f1predict-quick=f1predictor.scripts.run_prediction:main",
        ],
    },
)

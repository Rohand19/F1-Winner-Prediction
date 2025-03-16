"""Logging configuration for the F1 prediction system."""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logging(
    level: int = logging.INFO, log_file: Optional[str] = None, console: bool = True
) -> None:
    """Set up logging configuration.

    Args:
        level: The logging level (default: INFO)
        log_file: Optional path to log file
        console: Whether to log to console (default: True)
    """
    # Create formatters
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")

    # Create handlers
    handlers = []

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)

    if log_file is None:
        # Default log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"f1_predictor_{timestamp}.log"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(level=level, handlers=handlers, force=True)

    # Set FastF1 logger to WARNING to reduce verbosity
    logging.getLogger("fastf1").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

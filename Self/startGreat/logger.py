import logging
import os
from datetime import datetime
from typing import Optional

def get_logger(
    name: str = "apps_logger",
    log_dir: str = "logs",
    level: int = logging.DEBUG
) -> logging.Logger:
    """
    Creates a logger that outputs to both console and a timestamped file.

    Args:
        name: Base name for the logger and log file.
        log_dir: Directory where log files are saved.
        level: Root logging level (default: DEBUG).

    Returns:
        Configured logging.Logger instance.
    """
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Build file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.log"
    filepath = os.path.join(log_dir, filename)

    # Create or retrieve the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers to the same logger
    if not logger.handlers:
        # Formatter for both console and file
        fmt = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
        formatter = logging.Formatter(fmt)

        # File handler (all levels)
        file_handler = logging.FileHandler(filepath, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler (INFO+ by default)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


logger = get_logger()
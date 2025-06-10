import logging
import os
from datetime import datetime


def setup_logger(name='mnist_classifier', log_dir='logs/', level=logging.INFO):
    """
    Set up professional logging configuration.

    Args:
        name (str): Logger name
        log_dir (str): Directory to save log files
        level: Logging level

    Returns:
        logging.Logger: Configured logger
    """

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(f'{log_dir}training_{timestamp}.log')
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
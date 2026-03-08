"""Simple logging setup."""

import logging


def setup_logging(log_level: str = "INFO") -> None:
    """Setup simple logging to console.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

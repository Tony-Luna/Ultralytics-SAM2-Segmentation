# utils/logger_manager.py
# 01/02/2025
# Author: Tony-Luna
# -*- coding: utf-8 -*-
"""Logger Manager (V2, Google Style)

Configures a global logger for the entire application. Verbosity levels:
- 0 -> WARNING
- 1 -> INFO
- >=2 -> DEBUG
"""

import sys
import logging


class LoggerManager:
    """Manages logging setup for the application."""

    def __init__(self, verbosity: int) -> None:
        """Initializes the LoggerManager with a given verbosity.

        Args:
            verbosity (int): The desired logging verbosity level.
        """
        self.log_level: int = self._get_log_level(verbosity)
        self.logger: logging.Logger = self._setup_logging()

    def _get_log_level(self, verbosity: int) -> int:
        """Converts a verbosity integer into a Python logging level.

        Args:
            verbosity (int): 0=WARNING, 1=INFO, >=2=DEBUG

        Returns:
            int: The corresponding logging level constant.
        """
        if verbosity >= 2:
            return logging.DEBUG
        elif verbosity == 1:
            return logging.INFO
        return logging.WARNING

    def _setup_logging(self) -> logging.Logger:
        """Creates and configures the logger with a console handler.

        Returns:
            logging.Logger: The configured logger instance.
        """
        logger = logging.getLogger()
        logger.setLevel(self.log_level)

        # Remove any existing handlers to avoid duplication
        logger.handlers = []

        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        return logger

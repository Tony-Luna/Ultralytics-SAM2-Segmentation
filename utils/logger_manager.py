# utils/logger_manager.py
# -*- coding: utf-8 -*-
"""
Logger Manager Module.

This module provides the LoggerManager class to set up application logging.

Author: anlun
"""

import sys
import logging

class LoggerManager:
    """
    Manages the logging configuration for the application.
    """

    def __init__(self, verbosity: int):
        self.log_level = self.get_log_level(verbosity)
        self.logger = self.setup_logging()

    def get_log_level(self, verbosity: int) -> int:
        """
        Maps numerical verbosity to logging levels.
        """
        if verbosity >= 2:
            return logging.DEBUG
        elif verbosity == 1:
            return logging.INFO
        else:
            return logging.WARNING

    def setup_logging(self) -> logging.Logger:
        """
        Sets up the logging configuration.
        """
        logger = logging.getLogger()
        logger.setLevel(self.log_level)
        logger.handlers = []

        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        return logger

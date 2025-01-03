# config/loader.py
# 01/02/2025
# Author: Tony-Luna
# -*- coding: utf-8 -*-
"""Configuration Loader (V2) - Google Style

Provides a class to load and parse a YAML configuration file,
exposing the usage parameters for the segmentation application.
"""

import os
import yaml
from typing import Dict, Any


class ConfigLoader:
    """Reads and interprets the YAML configuration for the segmentation application."""

    def __init__(self, config_path: str) -> None:
        """Initializes ConfigLoader.

        Args:
            config_path (str): Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the config file is not found.
            yaml.YAMLError: If the file is not valid YAML.
        """
        self.config_path: str = config_path
        self.config: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Loads the YAML configuration file into a Python dictionary.

        Returns:
            Dict[str, Any]: Parsed configuration data.

        Raises:
            FileNotFoundError: If the file doesn't exist at config_path.
            yaml.YAMLError: If parsing fails.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        try:
            with open(self.config_path, 'r') as file:
                config_data = yaml.safe_load(file)
            return config_data
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}")

    def get_usage_parameters(self) -> Dict[str, Any]:
        """Retrieves the 'usage_parameters' subsection of the config.

        Returns:
            Dict[str, Any]: Dictionary of usage parameters if found, else an empty dict.
        """
        return self.config.get('usage_parameters', {})

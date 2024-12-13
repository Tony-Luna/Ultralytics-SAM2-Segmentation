# config/config_loader.py
# -*- coding: utf-8 -*-
"""
Configuration Loader Module.

This module provides the ConfigLoader class to load and parse configuration from a YAML file.

Author: anlun
"""

import os
import yaml

class ConfigLoader:
    """
    Loads and parses the configuration from a YAML file.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> dict:
        """
        Loads the YAML configuration file.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}")

    def get_usage_parameters(self) -> dict:
        return self.config.get('usage_parameters', {})

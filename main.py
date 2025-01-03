# main.py
# 01/02/2025
# Author: Tony-Luna
# -*- coding: utf-8 -*-
"""Main Entry Point (Enhanced with Google-Style Docstrings)

This script orchestrates the entire segmentation application:
1. Loads the YAML config (config.yaml) to retrieve paths and parameters.
2. Initializes the logger with the configured verbosity.
3. Instantiates the SAM-based segmenter.
4. Launches a Tkinter GUI for interactive image segmentation with positive and negative points.
5. Provides "Clear Last" and "Clear All" functionalities for removing points.
6. Saves cropped RGBA segmentation results upon user request.

Usage:
    python main.py
"""

import sys
import yaml
import os
import glob
from typing import List

from config.loader import ConfigLoader
from utils.logger_manager import LoggerManager
from segmentation.segmenter import Segmenter
from gui.segmentation_gui import SegmentationGUI


def get_image_files(directory: str) -> List[str]:
    """Retrieves a sorted list of image file paths from the specified directory.

    Searches for PNG, JPG, JPEG, BMP, TIFF, TIF.

    Args:
        directory (str): Directory path containing images.

    Returns:
        List[str]: Sorted list of image file paths matching supported extensions.
    """
    supported_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif')
    image_files: List[str] = []
    for ext in supported_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    image_files.sort()
    return image_files


def main() -> None:
    """Sets up and runs the segmentation application.

    Steps:
        1. Load configuration from 'config.yaml'.
        2. Set up a logger using the specified verbosity level.
        3. Initialize the SAM-based segmenter model.
        4. Check the input path (directory or single file) and collect image(s).
        5. Launch the Tkinter GUI for interactive point-based segmentation.
    """
    config_path = "config.yaml"
    try:
        config_loader = ConfigLoader(config_path)
        usage_params = config_loader.get_usage_parameters()
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    verbosity = usage_params.get('verbosity', 1)
    if not isinstance(verbosity, int):
        print("Invalid verbosity level in config. Must be an integer.")
        sys.exit(1)

    # Initialize logger
    logger_manager = LoggerManager(verbosity)
    logger = logger_manager.logger

    try:
        input_path: str = usage_params['input_image_path']
        output_dir: str = usage_params['output_image_path']
        model_path: str = usage_params['model_path']
        device: str = usage_params.get('device', 'cpu')
        visualization_size: int = usage_params.get('visualization_size', 800)

        logger.debug(f"Input Path: {input_path}")
        logger.debug(f"Output Directory: {output_dir}")
        logger.debug(f"Model Path: {model_path}")
        logger.debug(f"Device: {device}")
        logger.debug(f"Visualization Size: {visualization_size}")

        # Initialize segmenter
        segmenter = Segmenter(
            model_path=model_path,
            device=device,
            logger=logger
        )

        # Prepare output subdirectories
        os.makedirs(os.path.join(output_dir, "points"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

        if os.path.isdir(input_path):
            # Process all images in the directory
            logger.info(f"Input path is a directory. Processing images in: {input_path}")
            image_files = get_image_files(input_path)
            if not image_files:
                logger.warning(f"No supported image files found in: {input_path}")
                sys.exit(0)
            total_images = len(image_files)
            logger.info(f"Found {total_images} image(s).")

            gui = SegmentationGUI(
                image_files=image_files,
                segmenter=segmenter,
                output_dir=output_dir,
                logger=logger,
                visualization_size=visualization_size
            )
            gui.run()

        elif os.path.isfile(input_path):
            # Process a single image file
            logger.info(f"Input path is a single file: {input_path}")
            image_files = [input_path]
            gui = SegmentationGUI(
                image_files=image_files,
                segmenter=segmenter,
                output_dir=output_dir,
                logger=logger,
                visualization_size=visualization_size
            )
            gui.run()
        else:
            logger.error(f"Input path is neither a file nor a directory: {input_path}")
            sys.exit(1)

        logger.info("Processing completed.")
    except Exception as ex:
        logger.exception(f"An unexpected error occurred: {ex}")
        sys.exit(1)


if __name__ == "__main__":
    main()

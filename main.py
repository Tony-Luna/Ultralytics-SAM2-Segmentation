# main.py
# -*- coding: utf-8 -*-
"""
Object Segmentation Application with Tkinter-based GUI and visualization scaling.
"""

import sys
import yaml
import os
import glob
from config.config_loader import ConfigLoader
from detectors.object_segmenter import ObjectSegmenter
from utils.logger_manager import LoggerManager
from gui.segmentation_gui import SegmentationGUI

def get_image_files(directory: str) -> list:
    supported_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif')
    image_files = []
    for ext in supported_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    image_files.sort()
    return image_files

def main():
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
    logger_manager = LoggerManager(verbosity)
    logger = logger_manager.logger

    try:
        input_path = usage_params['input_image_path']
        output_dir = usage_params['output_image_path']
        model_path = usage_params['model_path']
        device = usage_params.get('device', 'cpu')
        visualization_size = usage_params.get('visualization_size', 800)

        logger.debug(f"Input Path: {input_path}")
        logger.debug(f"Output Directory: {output_dir}")
        logger.debug(f"Model Path: {model_path}")
        logger.debug(f"Device: {device}")
        logger.debug(f"Visualization Size: {visualization_size}")

        segmenter = ObjectSegmenter(model_path=model_path, device=device, logger=logger)

        os.makedirs(os.path.join(output_dir, "points"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

        if os.path.isdir(input_path):
            logger.info(f"Input path is a directory. Processing images in: {input_path}")
            image_files = get_image_files(input_path)
            if not image_files:
                logger.warning(f"No supported image files found in directory: {input_path}")
                sys.exit(0)
            total_images = len(image_files)
            logger.info(f"Found {total_images} image(s) to process.")

            gui = SegmentationGUI(
                image_files=image_files,
                segmenter=segmenter,
                output_dir=output_dir,
                logger=logger,
                visualization_size=visualization_size
            )
            gui.run()
        elif os.path.isfile(input_path):
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

        logger.info("All processing completed.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

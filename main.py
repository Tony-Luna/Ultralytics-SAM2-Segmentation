# # main.py
# # -*- coding: utf-8 -*-
# """
# Object Segmentation Application using SAM2.

# This application allows interactive segmentation via a GUI. The user can click to define points and 
# press 'y' to confirm and save the final segmented image (as PNG) in the given output directory 
# with alpha channel for transparency, or press 'ESC' to exit without saving.

# Author: anlun
# """

# import sys
# import yaml
# from config.config_loader import ConfigLoader
# from detectors.object_segmenter import ObjectSegmenter
# from utils.logger_manager import LoggerManager
# from gui.segmentation_gui import SegmentationGUI

# def main():
#     config_path = "config.yaml"
#     try:
#         config_loader = ConfigLoader(config_path)
#         usage_params = config_loader.get_usage_parameters()
#     except (FileNotFoundError, yaml.YAMLError) as e:
#         print(f"Error loading configuration: {e}")
#         sys.exit(1)

#     # Setup logging
#     verbosity = usage_params.get('verbosity', 1)
#     if not isinstance(verbosity, int):
#         print("Invalid verbosity level in config. Must be an integer.")
#         sys.exit(1)
#     logger_manager = LoggerManager(verbosity)
#     logger = logger_manager.logger

#     try:
#         input_image_path = usage_params['input_image_path']
#         output_dir = usage_params['output_image_path']  # Now this is treated as a directory
#         model_path = usage_params['model_path']
#         device = usage_params.get('device', 'cpu')

#         logger.debug(f"Input Image: {input_image_path}")
#         logger.debug(f"Output Directory: {output_dir}")
#         logger.debug(f"Model Path: {model_path}")
#         logger.debug(f"Device: {device}")

#         segmenter = ObjectSegmenter(model_path=model_path, device=device, logger=logger)
#         gui = SegmentationGUI(image_path=input_image_path, segmenter=segmenter, output_dir=output_dir, logger=logger)
#         gui.run()
#         logger.info("Process finished.")
#     except Exception as e:
#         logger.exception(f"An unexpected error occurred: {e}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()

# main.py
# -*- coding: utf-8 -*-
"""
Object Segmentation Application using SAM2.

This application allows interactive segmentation via a GUI. The user can click to define points and 
press 'y' to confirm and save the final segmented image (as PNG) in the given output directory 
with alpha channel for transparency, or press 'ESC' to exit without saving. 
If a directory is provided as input, the application processes all images sequentially.

Author: anlun
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
    """
    Retrieves a sorted list of image file paths from the given directory.
    Supports common image extensions.
    """
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

    # Setup logging
    verbosity = usage_params.get('verbosity', 1)
    if not isinstance(verbosity, int):
        print("Invalid verbosity level in config. Must be an integer.")
        sys.exit(1)
    logger_manager = LoggerManager(verbosity)
    logger = logger_manager.logger

    try:
        input_path = usage_params['input_image_path']
        output_dir = usage_params['output_image_path']  # Directory to save output images
        model_path = usage_params['model_path']
        device = usage_params.get('device', 'cpu')

        logger.debug(f"Input Path: {input_path}")
        logger.debug(f"Output Directory: {output_dir}")
        logger.debug(f"Model Path: {model_path}")
        logger.debug(f"Device: {device}")

        segmenter = ObjectSegmenter(model_path=model_path, device=device, logger=logger)

        # Check if input_path is a directory or a file
        if os.path.isdir(input_path):
            logger.info(f"Input path is a directory. Preparing to process all images in: {input_path}")
            image_files = get_image_files(input_path)
            if not image_files:
                logger.warning(f"No supported image files found in directory: {input_path}")
                sys.exit(0)
            total_images = len(image_files)
            logger.info(f"Found {total_images} image(s) to process.")

            for idx, image_path in enumerate(image_files, start=1):
                logger.info(f"Processing image {idx}/{total_images}: {image_path}")
                gui = SegmentationGUI(
                    image_path=image_path, 
                    segmenter=segmenter, 
                    output_dir=output_dir, 
                    logger=logger,
                    image_index=idx,
                    total_images=total_images
                )
                gui.run()
                logger.info(f"Finished processing image {idx}/{total_images}: {image_path}")
        elif os.path.isfile(input_path):
            logger.info(f"Input path is a single file. Processing image: {input_path}")
            gui = SegmentationGUI(
                image_path=input_path, 
                segmenter=segmenter, 
                output_dir=output_dir, 
                logger=logger
            )
            gui.run()
            logger.info(f"Finished processing image: {input_path}")
        else:
            logger.error(f"Input path is neither a file nor a directory: {input_path}")
            sys.exit(1)

        logger.info("All processing completed.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


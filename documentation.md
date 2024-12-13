```python
# main.py
# -*- coding: utf-8 -*-
"""
Object Segmentation Application using SAM2.

This application allows interactive segmentation via a GUI. The user can click to define points and 
press 'y' to confirm and save the final segmented image (as PNG) in the given output directory 
with alpha channel for transparency, or press 'ESC' to exit without saving.

Author: anlun
"""

import sys
import yaml
from config.config_loader import ConfigLoader
from detectors.object_segmenter import ObjectSegmenter
from utils.logger_manager import LoggerManager
from gui.segmentation_gui import SegmentationGUI

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
        input_image_path = usage_params['input_image_path']
        output_dir = usage_params['output_image_path']  # Now this is treated as a directory
        model_path = usage_params['model_path']
        device = usage_params.get('device', 'cpu')

        logger.debug(f"Input Image: {input_image_path}")
        logger.debug(f"Output Directory: {output_dir}")
        logger.debug(f"Model Path: {model_path}")
        logger.debug(f"Device: {device}")

        segmenter = ObjectSegmenter(model_path=model_path, device=device, logger=logger)
        gui = SegmentationGUI(image_path=input_image_path, segmenter=segmenter, output_dir=output_dir, logger=logger)
        gui.run()
        logger.info("Process finished.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

```

```yaml
# config.yaml

usage_parameters:
  input_image_path: "C:/Users/anlun/OneDrive/Pictures/Mary/Raw/1.jpg"         # Path to the input image
  output_image_path: "C:/Users/anlun/OneDrive/Pictures/Mary/Segmented"        # Directory to save the output image
  model_path: "./models/sam2.1_b.pt"             # Path to the SAM2 model weights
  verbosity: 2                                   # Logging verbosity: 0=WARNING, 1=INFO, 2=DEBUG
  device: "cuda:0"                               # Device: "cpu" or "cuda:0" for GPU

```

```python
# detectors/object_segmenter.py
# -*- coding: utf-8 -*-
"""
Object Segmenter Module.

Author: anlun
"""

import os
import logging
from typing import List

from ultralytics import SAM
from PIL import Image
import numpy as np

class ObjectSegmenter:
    """
    Performs object segmentation on images using a SAM2 model.
    """

    def __init__(self, model_path: str, device: str, logger: logging.Logger):
        self.logger = logger
        self.logger.debug(f"Loading SAM2 model from: {model_path}")
        self.model = SAM(model_path)
        self.device = device if device else 'cpu'
        self.model.to(self.device)
        self.logger.info(f"SAM2 model loaded on device: {self.device}")

    def segment(self, image_path: str, points: List[tuple], labels: List[int]):
        """
        Perform object segmentation on the given image using the provided points as prompts.
        Returns a single Results object or None.
        """
        if not os.path.exists(image_path):
            self.logger.error(f"Input image not found at: {image_path}")
            return None

        if len(points) == 0:
            self.logger.debug("No points provided, skipping segmentation.")
            return None

        self.logger.info(f"Performing segmentation on image: {image_path} with {len(points)} point(s).")

        try:
            points_array = np.array(points, dtype=np.float32)
            labels_array = np.array(labels, dtype=np.int32)
            results = self.model(image_path, points=points_array, labels=labels_array)
            # If the model returns a list, take the first element
            if isinstance(results, list) and len(results) > 0:
                results = results[0]
            self.logger.debug(f"Segmentation results: {results}")
            return results
        except Exception as e:
            self.logger.error(f"Error during segmentation: {e}")
            return None

    def save_cropped_alpha_result(self, results, output_path: str):
        """
        Saves the segmentation results as a cropped RGBA image with a transparent background.
        """
        if results is None or not hasattr(results, 'path'):
            self.logger.warning("No results or missing image path in results.")
            return

        image_path = results.path
        if not os.path.exists(image_path):
            self.logger.error("Original image file not found.")
            return

        original = Image.open(image_path).convert("RGBA")
        width, height = original.size

        combined_mask = np.zeros((height, width), dtype=np.uint8)
        if hasattr(results, 'masks') and results.masks is not None:
            mask_data = results.masks.data.cpu().numpy()  # (N, H, W)
            for m in mask_data:
                combined_mask = np.logical_or(combined_mask, m > 0).astype(np.uint8)

        if not np.any(combined_mask):
            self.logger.warning("No segmented region found; not saving image.")
            return

        rows = np.any(combined_mask, axis=1)
        cols = np.any(combined_mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        cropped_image = original.crop((xmin, ymin, xmax + 1, ymax + 1))
        cropped_mask = combined_mask[ymin:ymax+1, xmin:xmax+1]

        mask_pil = Image.fromarray((cropped_mask * 255).astype(np.uint8), mode='L')
        final_rgba = Image.new("RGBA", cropped_image.size, (0, 0, 0, 0))
        final_rgba.paste(cropped_image, (0, 0), mask_pil)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_rgba.save(output_path, "PNG")
        self.logger.info(f"Segmented image saved to: {output_path}")

```

```python
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

```

```python
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

```

```python
# gui/segmentation_gui.py
# -*- coding: utf-8 -*-
"""
Segmentation GUI Module.

Author: anlun
"""

import cv2
import numpy as np
import os
from typing import List, Tuple

class SegmentationGUI:
    """
    GUI for interactive segmentation.
    """

    def __init__(self, image_path: str, segmenter, output_dir: str, logger):
        self.image_path = image_path
        self.segmenter = segmenter
        self.output_dir = output_dir
        self.logger = logger

        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            self.logger.error(f"Failed to load image: {self.image_path}")
            raise FileNotFoundError(f"Image not found at: {self.image_path}")

        self.display_image = self.original_image.copy()
        self.points: List[Tuple[int, int]] = []
        self.labels: List[int] = []

        self.window_name = "Segmentation Tool - LMB: Add Point | RMB: Remove Point | ESC: Quit | y: Save"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.logger.info("Segmentation GUI initialized.")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self.labels.append(1)
            self.logger.debug(f"Added point: ({x}, {y})")
            self.update_segmentation()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                removed_point = self.points.pop()
                self.labels.pop()
                self.logger.debug(f"Removed point: {removed_point}")
                self.update_segmentation()

    def update_segmentation(self):
        self.logger.debug(f"Current points: {self.points}")

        if len(self.points) == 0:
            self.display_image = self.original_image.copy()
            return

        results = self.segmenter.segment(self.image_path, self.points, self.labels)
        self.display_image = self.original_image.copy()

        # Draw points
        for pt in self.points:
            cv2.circle(self.display_image, pt, radius=5, color=(0, 255, 0), thickness=-1)

        # Overlay masks if present
        if results and results.masks is not None:
            try:
                mask_data = results.masks.data.cpu().numpy()
                for m in mask_data:
                    mask_bool = m > 0
                    colored_mask = np.zeros_like(self.display_image, dtype=np.uint8)
                    colored_mask[:] = (0, 0, 255)  # Red
                    alpha = 0.5
                    self.display_image[mask_bool] = (
                        self.display_image[mask_bool]*(1 - alpha) +
                        colored_mask[mask_bool]*alpha
                    ).astype(np.uint8)
            except Exception as e:
                self.logger.error(f"Error during mask overlay: {e}")

    def run(self):
        self.logger.info("Starting GUI event loop.")
        try:
            while True:
                cv2.imshow(self.window_name, self.display_image)
                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC
                    self.logger.info("ESC pressed. Exiting without saving.")
                    break
                elif key == ord('y'):
                    self.logger.info("'y' pressed. Saving the segmented image.")
                    if len(self.points) > 0:
                        results = self.segmenter.segment(self.image_path, self.points, self.labels)
                        if results:
                            base_name = os.path.splitext(os.path.basename(self.image_path))[0] + ".png"
                            output_path = os.path.join(self.output_dir, base_name)
                            self.segmenter.save_cropped_alpha_result(results, output_path)
                    break
        except KeyboardInterrupt:
            self.logger.warning("KeyboardInterrupt received. Exiting gracefully.")
        finally:
            cv2.destroyAllWindows()
            self.logger.info("GUI closed, resources released.")

```

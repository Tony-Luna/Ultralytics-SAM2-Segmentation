```python
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

```

```yaml
# config.yaml

usage_parameters:
  input_image_path: "C:/Users/anlun/OneDrive/Pictures/bella/Raw"
  output_image_path: "C:/Users/anlun/OneDrive/Pictures/bella/Segmented"
  model_path: "./models/sam2.1_b.pt"
  verbosity: 2
  device: "cuda:0"
  visualization_size: 800

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
Tkinter-based Segmentation GUI Module with image scaling and improved point visualization.
"""

import os
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from typing import List, Tuple

class SegmentationGUI:
    """
    GUI for interactive segmentation using Tkinter.
    Scales images for visualization according to a given max dimension.
    Points are drawn as green "X"s on the original resolution image.
    """

    def __init__(self, image_files: List[str], segmenter, output_dir: str, logger, visualization_size: int = 800):
        self.image_files = image_files
        self.segmenter = segmenter
        self.output_dir = output_dir
        self.logger = logger
        self.visualization_size = visualization_size

        self.current_index = 0
        self.total_images = len(self.image_files)
        self.points: List[Tuple[int, int]] = []
        self.labels: List[int] = []

        self.image_path = None
        self.original_image = None
        self.display_image = None

        # Display scale factor (for visualization)
        self.display_scale = 1.0

        self.load_current_image()

        # Create Tkinter window
        self.root = tk.Tk()
        self.root.title("Segmentation Tool")

        # Control frame
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        prev_button = tk.Button(control_frame, text="<", command=self.on_prev_image)
        prev_button.pack(side=tk.LEFT, padx=5, pady=5)

        next_button = tk.Button(control_frame, text=">", command=self.on_next_image)
        next_button.pack(side=tk.LEFT, padx=5, pady=5)

        ok_button = tk.Button(control_frame, text="OK", command=self.on_save)
        ok_button.pack(side=tk.LEFT, padx=5, pady=5)

        remove_button = tk.Button(control_frame, text="RemoveLast", command=self.on_remove_last_point)
        remove_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.info_label = tk.Label(control_frame, text=f"{self.current_index+1}/{self.total_images}")
        self.info_label.pack(side=tk.LEFT, padx=10)

        # Image display frame
        image_frame = tk.Frame(self.root)
        image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Bind mouse events
        self.image_label.bind("<Button-1>", self.on_image_click_left)   # add point
        self.image_label.bind("<Button-3>", self.on_image_click_right)  # remove last point

        self.update_image_display()

    def load_points_if_exist(self, image_path):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        points_file = os.path.join(self.output_dir, "points", base_name + ".npz")
        if os.path.exists(points_file):
            data = np.load(points_file)
            self.points = [tuple(pt) for pt in data["points"]]
            self.labels = data["labels"].tolist()
            self.logger.info(f"Loaded existing points for image: {image_path}")
        else:
            self.points = []
            self.labels = []

    def save_points(self, image_path):
        if len(self.points) == 0:
            self.logger.debug("No points to save for this image.")
            return
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        points_file = os.path.join(self.output_dir, "points", base_name + ".npz")
        np.savez(points_file, points=np.array(self.points, dtype=np.int32), labels=np.array(self.labels, dtype=np.int32))
        self.logger.info(f"Points saved: {points_file}")

    def load_current_image(self):
        self.image_path = self.image_files[self.current_index]
        self.logger.info(f"Loading image {self.current_index+1}/{self.total_images}: {self.image_path}")
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            self.logger.error(f"Failed to load image: {self.image_path}")
            raise FileNotFoundError(f"Image not found at: {self.image_path}")
        self.load_points_if_exist(self.image_path)
        self.update_segmentation()

    def on_prev_image(self):
        self.save_current_work()
        if self.current_index > 0:
            self.current_index -= 1
            self.info_label.config(text=f"{self.current_index+1}/{self.total_images}")
            self.load_current_image()
            self.update_image_display()
        else:
            self.logger.info("Already at the first image.")

    def on_next_image(self):
        self.save_current_work()
        if self.current_index < self.total_images - 1:
            self.current_index += 1
            self.info_label.config(text=f"{self.current_index+1}/{self.total_images}")
            self.load_current_image()
            self.update_image_display()
        else:
            self.logger.info("Already at the last image.")

    def on_save(self):
        self.save_current_work(save_image=True)

    def on_remove_last_point(self):
        if self.points:
            removed_point = self.points.pop()
            self.labels.pop()
            self.logger.debug(f"Removed point via button: {removed_point}")
            self.update_segmentation()
            self.update_image_display()

    def save_current_work(self, save_image=False):
        self.save_points(self.image_path)
        if save_image and len(self.points) > 0:
            results = self.segmenter.segment(self.image_path, self.points, self.labels)
            if results:
                base_name = os.path.splitext(os.path.basename(self.image_path))[0] + ".png"
                output_path = os.path.join(self.output_dir, "images", base_name)
                self.segmenter.save_cropped_alpha_result(results, output_path)
                self.logger.info(f"Segmented image saved to: {output_path}")

    def on_image_click_left(self, event):
        # Convert from display coords to original coords
        disp_x, disp_y = event.x, event.y
        orig_x = int(disp_x / self.display_scale)
        orig_y = int(disp_y / self.display_scale)

        self.points.append((orig_x, orig_y))
        self.labels.append(1)
        self.logger.debug(f"Added point at original coords: ({orig_x}, {orig_y})")
        self.update_segmentation()
        self.update_image_display()

    def on_image_click_right(self, event):
        if self.points:
            removed_point = self.points.pop()
            self.labels.pop()
            self.logger.debug(f"Removed point by right-click: {removed_point}")
            self.update_segmentation()
            self.update_image_display()

    def update_segmentation(self):
        self.logger.debug(f"Current points: {self.points}")
        self.display_image = self.original_image.copy()

        if len(self.points) == 0:
            return

        results = self.segmenter.segment(self.image_path, self.points, self.labels)
        self.display_image = self.original_image.copy()

        # Draw points as a small green "X"
        # Each "X" is made of two crossing lines
        for pt in self.points:
            x, y = pt
            cv2.line(self.display_image, (x-5, y-5), (x+5, y+5), (0,255,0), 2)
            cv2.line(self.display_image, (x-5, y+5), (x+5, y-5), (0,255,0), 2)

        # Overlay mask if present
        if results and results.masks is not None:
            try:
                mask_data = results.masks.data.cpu().numpy()
                combined_mask = np.zeros(self.display_image.shape[:2], dtype=bool)
                for m in mask_data:
                    combined_mask |= (m > 0)

                colored_mask = np.zeros_like(self.display_image, dtype=np.uint8)
                colored_mask[:] = (0, 0, 255)  # Red
                alpha = 0.5
                self.display_image[combined_mask] = (
                    self.display_image[combined_mask]*(1 - alpha) +
                    colored_mask[combined_mask]*alpha
                ).astype(np.uint8)
            except Exception as e:
                self.logger.error(f"Error during mask overlay: {e}")

    def update_image_display(self):
        # Convert display_image (BGR) to RGB
        rgb_image = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Determine the display scale
        img_w, img_h = pil_image.size
        largest_dim = max(img_w, img_h)
        if largest_dim > self.visualization_size:
            self.display_scale = self.visualization_size / largest_dim
        else:
            self.display_scale = 1.0

        display_w = int(img_w * self.display_scale)
        display_h = int(img_h * self.display_scale)
        pil_image = pil_image.resize((display_w, display_h), Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=pil_image)
        self.image_label.imgtk = imgtk
        self.image_label.configure(image=imgtk)

    def run(self):
        self.logger.info("Starting Tkinter main event loop.")
        self.root.mainloop()
        self.logger.info("GUI closed, resources released.")

```

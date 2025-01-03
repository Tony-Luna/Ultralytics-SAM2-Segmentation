# segmentation/segmenter.py
# 01/02/2025
# Author: Tony-Luna
# -*- coding: utf-8 -*-
"""Segmenter (V2, Google Style)

Defines a Segmenter class using the Ultralytics SAM (Segment Anything Model) for instance segmentation.
Supports both positive (label=1) and negative (label=0) prompt points.
"""

import os
import logging
from typing import List, Optional, Any

from ultralytics import SAM
from PIL import Image
import numpy as np


class Segmenter:
    """A high-level interface for performing object segmentation with Ultralytics SAM."""

    def __init__(self, model_path: str, device: str, logger: logging.Logger) -> None:
        """Initializes the Segmenter with a specified SAM model and device.

        Args:
            model_path (str): Path to the SAM .pt file (e.g., 'sam2.1_b.pt').
            device (str): 'cpu', 'cuda:0', etc.
            logger (logging.Logger): Logger for status messages.
        """
        self.logger = logger
        self.logger.debug(f"Loading SAM model from: {model_path}")

        self.model = SAM(model_path)
        self.device = device if device else 'cpu'
        self.model.to(self.device)

        self.logger.info(f"SAM model loaded on device: {self.device}")

    def segment(
        self, 
        image_path: str, 
        points: List[tuple], 
        labels: List[int]
    ) -> Optional[Any]:
        """Executes segmentation on the given image using prompt points/labels.

        Args:
            image_path (str): Full path to the image file.
            points (List[tuple]): A list of (x, y) pixel coordinates.
            labels (List[int]): 1=positive, 0=negative, matching 'points' length.

        Returns:
            Optional[Any]: A results object from SAM containing masks, or None if an error occurs.

        Raises:
            FileNotFoundError: If the provided image_path does not exist.
        """
        if not os.path.exists(image_path):
            self.logger.error(f"Image not found: {image_path}")
            raise FileNotFoundError(f"Cannot locate image: {image_path}")

        if len(points) == 0:
            self.logger.debug("No prompt points provided. Skipping segmentation.")
            return None

        self.logger.info(f"Segmenting image with {len(points)} prompt point(s).")
        try:
            points_array = np.array(points, dtype=np.float32)
            labels_array = np.array(labels, dtype=np.int32)

            results = self.model(image_path, points=points_array, labels=labels_array)
            # If the model returns a list, use the first entry
            if isinstance(results, list) and len(results) > 0:
                results = results[0]

            self.logger.debug("Segmentation completed successfully.")
            return results
        except Exception as e:
            self.logger.error(f"Error during SAM segmentation: {e}")
            return None

    def save_cropped_alpha_result(
        self, 
        results: Any, 
        output_path: str
    ) -> None:
        """Saves a cropped RGBA image with transparency outside the combined segmentation mask.

        Args:
            results (Any): The result object from SAM, expected to have '.path' and '.masks'.
            output_path (str): File path for saving the PNG.

        Returns:
            None

        Raises:
            FileNotFoundError: If the original image path in 'results' does not exist.
        """
        if results is None or not hasattr(results, 'path'):
            self.logger.warning("No results or invalid result object. Nothing to save.")
            return

        image_path: str = results.path
        if not os.path.exists(image_path):
            self.logger.error(f"Original image not found: {image_path}")
            raise FileNotFoundError(f"Cannot locate image: {image_path}")

        original = Image.open(image_path).convert("RGBA")
        width, height = original.size

        combined_mask = np.zeros((height, width), dtype=np.uint8)

        if hasattr(results, 'masks') and results.masks is not None:
            mask_data = results.masks.data.cpu().numpy()  # shape: (N, H, W)
            for m in mask_data:
                combined_mask = np.logical_or(combined_mask, m > 0).astype(np.uint8)

        if not np.any(combined_mask):
            self.logger.warning("Combined mask is empty. No segmented region to save.")
            return

        # Calculate bounding box of the combined mask
        rows = np.any(combined_mask, axis=1)
        cols = np.any(combined_mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        # Crop the image and mask
        cropped_image = original.crop((xmin, ymin, xmax + 1, ymax + 1))
        cropped_mask = combined_mask[ymin:ymax+1, xmin:xmax+1]

        mask_pil = Image.fromarray((cropped_mask * 255).astype(np.uint8), mode='L')
        final_rgba = Image.new("RGBA", cropped_image.size, (0, 0, 0, 0))
        final_rgba.paste(cropped_image, (0, 0), mask_pil)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_rgba.save(output_path, "PNG")
        self.logger.info(f"Saved segmented RGBA image to: {output_path}")

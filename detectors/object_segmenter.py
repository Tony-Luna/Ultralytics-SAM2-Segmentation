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

# # gui/segmentation_gui.py
# # -*- coding: utf-8 -*-
# """
# Segmentation GUI Module.

# Author: anlun
# """

# import cv2
# import numpy as np
# import os
# from typing import List, Tuple

# class SegmentationGUI:
#     """
#     GUI for interactive segmentation.
#     """

#     def __init__(self, image_path: str, segmenter, output_dir: str, logger):
#         self.image_path = image_path
#         self.segmenter = segmenter
#         self.output_dir = output_dir
#         self.logger = logger

#         self.original_image = cv2.imread(self.image_path)
#         if self.original_image is None:
#             self.logger.error(f"Failed to load image: {self.image_path}")
#             raise FileNotFoundError(f"Image not found at: {self.image_path}")

#         self.display_image = self.original_image.copy()
#         self.points: List[Tuple[int, int]] = []
#         self.labels: List[int] = []

#         self.window_name = "Segmentation Tool - LMB: Add Point | RMB: Remove Point | ESC: Quit | y: Save"
#         cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
#         cv2.setMouseCallback(self.window_name, self.mouse_callback)

#         self.logger.info("Segmentation GUI initialized.")

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

    def __init__(self, image_path: str, segmenter, output_dir: str, logger, image_index: int = None, total_images: int = None):
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

        # Update window name to include progress if provided
        if image_index is not None and total_images is not None:
            self.window_name = f"Segmentation Tool - Image {image_index}/{total_images} - LMB: Add Point | RMB: Remove Point | ESC: Quit | y: Save"
        else:
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

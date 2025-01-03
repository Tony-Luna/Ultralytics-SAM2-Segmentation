# gui/segmentation_gui.py
# 01/02/2025
# Author: Tony-Luna
# -*- coding: utf-8 -*-
"""Segmentation GUI (V2, Google Style)

Tkinter-based GUI for interactive segmentation with the Ultralytics SAM model.
Allows:
- Left-click to add positive points (label=1).
- Right-click to add negative points (label=0).
- "Clear Last" to remove the last added point.
- "Clear All" to remove all points.
- "OK" to run segmentation and save a cropped RGBA result.

The display shows:
- Green crosses for label=1.
- Red crosses for label=0.
- Overlay masks in semi-transparent red if segmentation is available.

Usage:
    gui = SegmentationGUI(
        image_files=[...],
        segmenter=<Segmenter instance>,
        output_dir='...',
        logger=<Logger instance>,
        visualization_size=800
    )
    gui.run()
"""

import os
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from typing import List, Tuple, Optional

class SegmentationGUI:
    """Provides an interactive Tkinter window for point-based segmentation.

    Attributes:
        image_files (List[str]): List of file paths to images for segmentation.
        segmenter: An instance of Segmenter or compatible class with .segment() method.
        output_dir (str): Directory to save points and segmented images.
        logger: Logger for debug/info messages.
        visualization_size (int): Max dimension for resizing images in display.
        current_index (int): Index of the currently displayed image.
        total_images (int): Number of images to process.
        points (List[Tuple[int, int]]): List of user-added points (x, y).
        labels (List[int]): List of point labels (1=pos, 0=neg).
        display_scale (float): Scale factor for displayed image.
        image_path (Optional[str]): Current image path.
        original_image (Optional[np.ndarray]): The loaded original image (BGR).
        display_image (Optional[np.ndarray]): The image displayed after overlaying points/masks.
    """

    def __init__(
        self,
        image_files: List[str],
        segmenter,
        output_dir: str,
        logger,
        visualization_size: int = 800
    ) -> None:
        """Initializes the SegmentationGUI and sets up the Tkinter interface.

        Args:
            image_files (List[str]): Paths to images for segmentation.
            segmenter: A Segmenter instance for SAM-based segmentation.
            output_dir (str): Path where points and segmented images are saved.
            logger: Logger for status messages.
            visualization_size (int, optional): Max dimension for resizing images. Defaults to 800.
        """
        self.image_files: List[str] = image_files
        self.segmenter = segmenter
        self.output_dir: str = output_dir
        self.logger = logger
        self.visualization_size: int = visualization_size

        self.current_index: int = 0
        self.total_images: int = len(self.image_files)
        self.points: List[Tuple[int, int]] = []
        self.labels: List[int] = []

        self.image_path: Optional[str] = None
        self.original_image: Optional[np.ndarray] = None
        self.display_image: Optional[np.ndarray] = None
        self.display_scale: float = 1.0

        self._load_current_image()

        # Tkinter setup
        self.root = tk.Tk()
        self.root.title("Segmentation Tool v2 (Google-Style)")

        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Navigation Buttons
        prev_button = tk.Button(control_frame, text="<", command=self._on_prev_image)
        prev_button.pack(side=tk.LEFT, padx=5, pady=5)

        next_button = tk.Button(control_frame, text=">", command=self._on_next_image)
        next_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Segmentation Buttons
        ok_button = tk.Button(control_frame, text="OK", command=self._on_save)
        ok_button.pack(side=tk.LEFT, padx=5, pady=5)

        clear_last_button = tk.Button(control_frame, text="Clear Last", command=self._on_clear_last_point)
        clear_last_button.pack(side=tk.LEFT, padx=5, pady=5)

        clear_all_button = tk.Button(control_frame, text="Clear All", command=self._on_clear_all_points)
        clear_all_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.info_label = tk.Label(control_frame, text=f"{self.current_index+1}/{self.total_images}")
        self.info_label.pack(side=tk.LEFT, padx=10)

        # Image display area
        image_frame = tk.Frame(self.root)
        image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Mouse events: left-click => positive, right-click => negative
        self.image_label.bind("<Button-1>", self._on_image_click_left)
        self.image_label.bind("<Button-3>", self._on_image_click_right)

        self._update_image_display()

    def _load_points_if_exist(self, image_path: str) -> None:
        """Loads saved points and labels for a given image if they exist.

        Args:
            image_path (str): The path to the current image.
        """
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        points_file = os.path.join(self.output_dir, "points", base_name + ".npz")
        if os.path.exists(points_file):
            data = np.load(points_file)
            self.points = [tuple(pt) for pt in data["points"]]
            self.labels = list(data["labels"])
            self.logger.info(f"Loaded existing points from {points_file}")
        else:
            self.points.clear()
            self.labels.clear()

    def _save_points(self, image_path: str) -> None:
        """Saves the current points and labels for the given image.

        Args:
            image_path (str): The path to the current image.

        Returns:
            None
        """
        if len(self.points) == 0:
            self.logger.debug("No points to save for this image.")
            return
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        points_file = os.path.join(self.output_dir, "points", base_name + ".npz")
        np.savez(points_file, points=np.array(self.points, dtype=np.int32), labels=np.array(self.labels, dtype=np.int32))
        self.logger.info(f"Saved points to {points_file}")

    def _load_current_image(self) -> None:
        """Loads the current image from disk using OpenCV and applies any stored points."""
        self.image_path = self.image_files[self.current_index]
        self.logger.info(f"Loading image [{self.current_index+1}/{self.total_images}]: {self.image_path}")
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            self.logger.error(f"Failed to load image: {self.image_path}")
            raise FileNotFoundError(f"Cannot open image: {self.image_path}")
        self._load_points_if_exist(self.image_path)
        self._update_segmentation()

    def _on_prev_image(self) -> None:
        """Navigates to the previous image, saving current data first."""
        self._save_current_work()
        if self.current_index > 0:
            self.current_index -= 1
            self.info_label.config(text=f"{self.current_index+1}/{self.total_images}")
            self._load_current_image()
            self._update_image_display()
        else:
            self.logger.info("Already at the first image.")

    def _on_next_image(self) -> None:
        """Navigates to the next image, saving current data first."""
        self._save_current_work()
        if self.current_index < self.total_images - 1:
            self.current_index += 1
            self.info_label.config(text=f"{self.current_index+1}/{self.total_images}")
            self._load_current_image()
            self._update_image_display()
        else:
            self.logger.info("Already at the last image.")

    def _on_save(self) -> None:
        """Saves points and, if any exist, runs segmentation and saves the resulting image."""
        self._save_current_work(save_image=True)

    def _on_clear_last_point(self) -> None:
        """Removes the last added point (if any) and updates segmentation."""
        if self.points:
            removed_pt = self.points.pop()
            removed_label = self.labels.pop()
            self.logger.debug(f"Cleared last point: {removed_pt} (label={removed_label})")
        self._update_segmentation()
        self._update_image_display()

    def _on_clear_all_points(self) -> None:
        """Removes all points and updates segmentation."""
        if self.points:
            self.points.clear()
            self.labels.clear()
            self.logger.debug("Cleared all points.")
        self._update_segmentation()
        self._update_image_display()

    def _save_current_work(self, save_image: bool = False) -> None:
        """Saves current points and optionally runs segmentation to save the result.

        Args:
            save_image (bool, optional): Whether to run segmentation and save output image.
                Defaults to False.
        """
        self._save_points(self.image_path)
        if save_image and len(self.points) > 0:
            results = self.segmenter.segment(self.image_path, self.points, self.labels)
            if results:
                base_name = os.path.splitext(os.path.basename(self.image_path))[0] + ".png"
                output_path = os.path.join(self.output_dir, "images", base_name)
                self.segmenter.save_cropped_alpha_result(results, output_path)
                self.logger.info(f"Segmented image saved: {output_path}")

    def _on_image_click_left(self, event) -> None:
        """Handles left-click on the image, adding a positive point (label=1).

        Args:
            event: Tkinter event containing the x,y coordinates.
        """
        disp_x, disp_y = event.x, event.y
        orig_x = int(disp_x / self.display_scale)
        orig_y = int(disp_y / self.display_scale)

        self.points.append((orig_x, orig_y))
        self.labels.append(1)
        self.logger.debug(f"Added positive point: ({orig_x}, {orig_y})")

        self._update_segmentation()
        self._update_image_display()

    def _on_image_click_right(self, event) -> None:
        """Handles right-click on the image, adding a negative point (label=0).

        Args:
            event: Tkinter event containing the x,y coordinates.
        """
        disp_x, disp_y = event.x, event.y
        orig_x = int(disp_x / self.display_scale)
        orig_y = int(disp_y / self.display_scale)

        self.points.append((orig_x, orig_y))
        self.labels.append(0)
        self.logger.debug(f"Added negative point: ({orig_x}, {orig_y})")

        self._update_segmentation()
        self._update_image_display()

    def _update_segmentation(self) -> None:
        """Runs model segmentation if points exist, otherwise clears any existing overlay.

        Renders crosses for each point and overlays the mask if available.
        """
        self.logger.debug(f"Current points: {self.points}")
        # Start from the original image
        if self.original_image is None:
            return

        self.display_image = self.original_image.copy()

        if not self.points:
            # No points => no model call
            return

        results = self.segmenter.segment(self.image_path, self.points, self.labels)
        self.display_image = self.original_image.copy()

        # Draw crosses
        for (x, y), label in zip(self.points, self.labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.line(self.display_image, (x-5, y-5), (x+5, y+5), color, 2)
            cv2.line(self.display_image, (x-5, y+5), (x+5, y-5), color, 2)

        if results and hasattr(results, "masks") and results.masks is not None:
            try:
                mask_data = results.masks.data.cpu().numpy()
                combined_mask = np.zeros(self.display_image.shape[:2], dtype=bool)
                for m in mask_data:
                    combined_mask |= (m > 0)

                colored_mask = np.zeros_like(self.display_image, dtype=np.uint8)
                colored_mask[:] = (0, 0, 255)  # Red overlay
                alpha = 0.5
                self.display_image[combined_mask] = (
                    self.display_image[combined_mask] * (1 - alpha)
                    + colored_mask[combined_mask] * alpha
                ).astype(np.uint8)
            except Exception as e:
                self.logger.error(f"Mask overlay error: {e}")

    def _update_image_display(self) -> None:
        """Converts the display image to a Tkinter-compatible PhotoImage and scales it."""
        if self.display_image is None:
            return
        rgb_image = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Scale to visualization_size if needed
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

    def run(self) -> None:
        """Starts the Tkinter main loop. Blocks until the GUI is closed."""
        self.logger.info("Starting Tkinter main loop (V2, Google-Style).")
        self.root.mainloop()
        self.logger.info("GUI closed.")

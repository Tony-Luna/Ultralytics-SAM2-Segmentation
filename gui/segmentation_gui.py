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

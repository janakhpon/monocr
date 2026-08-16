import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple

class LineSegmenter:
    """
    Robust line segmenter using Horizontal Projection Profiles with Smoothing.
    Handles noisy documents and touching lines by finding valleys in the projection.

    Kept deliberately in step with ``monocr_onnx.segmenter.LineSegmenter``: the
    same thresholds, the same minimum line height, and the same relative
    padding, so the two packages cut a page the same way. Changing a number here
    without changing it there is what makes two bindings disagree on one input.
    """
    def __init__(self, min_line_h: int = 10, smooth_kernel: int = 5, threshold_ratio: float = 0.02):
        self.min_line_h = min_line_h
        self.smooth_kernel = smooth_kernel
        self.threshold_ratio = threshold_ratio

    def segment(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        Segment a document image into text lines.
        Returns list of (cropped_image, (x, y, w, h)) sorted top-to-bottom.
        """
        # 1. Convert to Grayscale & Numpy
        img_np = np.array(image.convert("L"))
        h_img, w_img = img_np.shape

        # 2. Binarize (Adaptive Thresholding)
        # Invert so text is white, background black
        binary = cv2.adaptiveThreshold(
            img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 10
        )

        # 3. Horizontal Projection Profile
        hist = np.sum(binary, axis=1).astype(np.float32)  # Shape (H,)

        # 4. Smooth Histogram
        if self.smooth_kernel > 1:
            kernel = np.ones(self.smooth_kernel) / self.smooth_kernel
            hist = np.convolve(hist, kernel, mode='same')

        # 5. Find Text Regions vs Spaces
        #
        # A blank page needs no special case: its maximum is 0, so the threshold
        # is 0 and the strict `>` excludes every row. An explicit early return
        # here was dead code — mutation testing on 2026-08-16 removed it and no
        # test changed.
        max_val = np.max(hist)
        threshold = max_val * self.threshold_ratio

        is_text_row = hist > threshold

        lines: List[Tuple[Image.Image, Tuple[int, int, int, int]]] = []
        start_y = None

        for y, is_text in enumerate(is_text_row):
            if is_text and start_y is None:
                start_y = y # Start of line
            elif not is_text and start_y is not None:
                # End of line (found a gap). Anything shorter than min_line_h is
                # a speckle, not a line.
                if (y - start_y) >= self.min_line_h:
                    self._extract_line(binary, img_np, start_y, y, image, lines)
                start_y = None

        # Handle last block
        if start_y is not None and (h_img - start_y) >= self.min_line_h:
            self._extract_line(binary, img_np, start_y, h_img, image, lines)

        return lines

    def _extract_line(self, binary, gray, r_start, r_end, source_image, lines_list):
        """Crop one detected line region and append it as (crop, bbox).

        Trims left and right whitespace and pads relative to the line's own
        height. A full-width crop with a fixed 4-pixel pad — what this class did
        until 2.3.0, with this method sitting unused beside it — hands the model
        a strip whose aspect ratio is the page's, not the line's, and the resize
        then squeezes the text horizontally to fit the window.
        """
        # Find horizontal boundaries (cropping left/right whitespace)
        line_slice = binary[r_start:r_end, :]
        col_sums = np.sum(line_slice, axis=0)
        col_indices = np.where(col_sums > 0)[0]

        if len(col_indices) == 0:
            return

        x_start, x_end = col_indices[0], col_indices[-1]

        # Relative padding based on line height
        h_raw = r_end - r_start
        pad_y = int(h_raw * 0.20)
        pad_x = int(h_raw * 0.15)

        y1 = max(0, r_start - pad_y)
        y2 = min(gray.shape[0], r_end + pad_y)
        x1 = max(0, x_start - pad_x)
        x2 = min(gray.shape[1], x_end + pad_x)

        crop = source_image.crop((x1, y1, x2, y2))
        lines_list.append((crop, (int(x1), int(y1), int(x2 - x1), int(y2 - y1))))


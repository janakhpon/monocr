import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple

class LineSegmenter:
    """
    Robust line segmenter using Horizontal Projection Profiles with Smoothing.
    Handles noisy documents and touching lines by finding valleys in the projection.
    """
    def __init__(self, min_line_h: int = 10, smooth_window: int = 3):
        self.min_line_h = min_line_h
        self.smooth_window = smooth_window

    def segment(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        Segment a document image into text lines.
        Returns list of (cropped_image, (x, y, w, h)) sorted top-to-bottom.
        """
        # Convert to CV2 grayscale
        img_np = np.array(image)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
            
        h_img, w_img = gray.shape

        # 1. Binarize (Adaptive Thresholding)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 25, 10
        )

        # 2. Horizontal Projection Profile
        raw_hist = np.sum(binary, axis=1) # Shape (h,)
        
        # 3. Smooth the histogram
        if self.smooth_window > 1:
            kernel = np.ones(self.smooth_window) / self.smooth_window
            hist = np.convolve(raw_hist, kernel, mode='same')
        else:
            hist = raw_hist

        # 4. Dynamic Thresholding using Histogram Stats
        # Identify "text rows" vs "gap rows"
        # We assume gaps have significantly lower pixel density than text.
        # Use a low percentile of non-zero values as the noise floor base.
        
        non_zero_hist = hist[hist > 0]
        if len(non_zero_hist) == 0:
            return []
            
        # "Gap" is usually near 0. "Text" is high.
        # We define a gap as anything below 5% of the AVERAGE text density.
        # This scales with font weight/size automatically.
        mean_density = np.mean(non_zero_hist)
        gap_threshold = mean_density * 0.05 
        
        is_text = hist > gap_threshold
        
        lines = []
        start = None
        
        for i, val in enumerate(is_text):
            if val and start is None:
                start = i
            elif not val and start is not None:
                # End of a text block
                end = i
                
                # Check if it's tall enough to be a line (filter dots/noise)
                if (end - start) >= self.min_line_h:
                    self._extract_line(binary, gray, start, end, image, lines)
                start = None
                
        # Handle last line
        if start is not None:
             if (h_img - start) >= self.min_line_h:
                self._extract_line(binary, gray, start, h_img, image, lines)

        return lines

    def _extract_line(self, binary, gray, r_start, r_end, source_image, lines_list):
        """Helper to crop the line horizontally and add to list"""
        # Crop the horizontal strip
        line_slice = binary[r_start:r_end, :]
        
        # Find horizontal boundaries (cropping left/right whitespace)
        col_sums = np.sum(line_slice, axis=0)
        col_indices = np.where(col_sums > 0)[0]
        
        if len(col_indices) == 0:
            return
            
        x_start, x_end = col_indices[0], col_indices[-1]
        
        # Add padding
        pad = 4
        y1 = max(0, r_start - pad)
        y2 = min(gray.shape[0], r_end + pad)
        x1 = max(0, x_start - pad)
        x2 = min(gray.shape[1], x_end + pad)
        
        w = x2 - x1
        h = y2 - y1
        
        crop = source_image.crop((x1, y1, x2, y2))
        lines_list.append((crop, (x1, y1, w, h)))

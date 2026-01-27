import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple

class LineSegmenter:
    """
    Robust line segmenter using Horizontal Projection Profiles.
    Better for handling tight line spacing than morphological ops.
    """
    def __init__(self, min_line_h: int = 15, min_gap: int = 5):
        self.min_line_h = min_line_h
        self.min_gap = min_gap

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

        # 1. Binarize (Otsu + Invert so text is white)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 2. Horizontal Projection Profile
        # Sum of pixels in each row
        hist = np.sum(binary, axis=1)
        
        # 3. Find gaps
        # A line exists where the sum > threshold
        threshold = 0  # Can be slightly > 0 for noisy images
        
        line_indices = np.where(hist > threshold)[0]
        
        lines = []
        if len(line_indices) == 0:
            return lines

        # Group consecutive indices into regions
        regions = []
        if len(line_indices) > 0:
            start = line_indices[0]
            for i in range(1, len(line_indices)):
                # If gap > min_gap, treat as break
                if line_indices[i] > line_indices[i-1] + self.min_gap:
                    end = line_indices[i-1]
                    regions.append((start, end))
                    start = line_indices[i]
            regions.append((start, line_indices[-1]))

        # 4. Extract lines
        for r_start, r_end in regions:
            # Refine vertical boundaries
            h_segment = r_end - r_start
            if h_segment < self.min_line_h:
                continue
            
            # Find horizontal boundaries (cropping left/right whitespace)
            line_slice = binary[r_start:r_end, :]
            col_sums = np.sum(line_slice, axis=0)
            col_indices = np.where(col_sums > 0)[0]
            
            if len(col_indices) == 0:
                continue
                
            x_start, x_end = col_indices[0], col_indices[-1]
            
            # Add padding
            pad = 4
            y1 = max(0, r_start - pad)
            y2 = min(gray.shape[0], r_end + pad)
            x1 = max(0, x_start - pad)
            x2 = min(gray.shape[1], x_end + pad)
            
            w = x2 - x1
            h = y2 - y1
            
            crop = image.crop((x1, y1, x2, y2))
            lines.append((crop, (x1, y1, w, h)))

        return lines

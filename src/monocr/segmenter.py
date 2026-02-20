import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple

class LineSegmenter:
    """
    Robust line segmenter using Horizontal Projection Profiles with Smoothing.
    Handles noisy documents and touching lines by finding valleys in the projection.
    """
    def __init__(self, smooth_kernel: int = 5, threshold_ratio: float = 0.02):
        self.smooth_kernel = smooth_kernel
        self.threshold_ratio = threshold_ratio

    def segment(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        Segment a document image into text lines.
        Returns list of (cropped_image, (0, y1, w, h-y1)) sorted top-to-bottom.
        """
        # 1. Convert to Grayscale & Numpy
        img_np = np.array(image.convert("L"))
        
        # 2. Binarize (Adaptive Thresholding)
        # Invert so text is white, background black
        binary = cv2.adaptiveThreshold(
            img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 25, 10
        )
        
        # 3. Horizontal Projection Profile
        hist = np.sum(binary, axis=1) # Shape (H,)
        
        # 4. Smooth Histogram
        if self.smooth_kernel > 1:
            kernel = np.ones(self.smooth_kernel) / self.smooth_kernel
            hist = np.convolve(hist, kernel, mode='same')
            
        # 5. Find Text Regions vs Spaces
        max_val = np.max(hist)
        threshold = max_val * self.threshold_ratio
        
        is_text_row = hist > threshold
        
        lines = []
        start_y = None
        h_img, w_img = img_np.shape
        
        for y, is_text in enumerate(is_text_row):
            if is_text and start_y is None:
                start_y = y # Start of line
            elif not is_text and start_y is not None:
                # End of line (found a gap)
                end_y = y
                
                # Check height to ignore tiny noise speckles
                if (end_y - start_y) > 8: 
                    # Add generous padding to avoid cutting loose ascenders/descenders
                    pad = 4
                    y1 = max(0, start_y - pad)
                    y2 = min(h_img, end_y + pad)
                    
                    crop = image.crop((0, y1, w_img, y2))
                    # Return with bounding box relative to original image
                    # BB Format: (x, y, w, h)
                    lines.append((crop, (0, y1, w_img, y2 - y1)))
                
                start_y = None
                
        # Handle last block
        if start_y is not None:
            if (h_img - start_y) > 8:
                pad = 4
                y1 = max(0, start_y - pad)
                y2 = min(h_img, h_img)
                crop = image.crop((0, y1, w_img, y2))
                lines.append((crop, (0, y1, w_img, y2 - y1)))
                
        return lines

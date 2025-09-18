#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced inference utilities for Mon OCR
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import json
import logging
from typing import List, Dict, Optional, Union

from .ocr import MonOCR

class MonOCRInference:
    """Advanced Mon OCR inference with additional utilities"""
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = "crnn"):
        """
        Initialize advanced Mon OCR inference
        
        Args:
            model_path: Path to trained model file
            model_type: Type of model ("crnn" or "trocr")
        """
        self.ocr = MonOCR(model_path, model_type)
        self.logger = logging.getLogger(__name__)
    
    def predict_with_confidence(self, image: Union[str, Image.Image]) -> Dict[str, Union[str, float]]:
        """
        Predict text with confidence score
        
        Args:
            image: Path to image file or PIL Image object
            
        Returns:
            Dictionary with 'text' and 'confidence' keys
        """
        try:
            text = self.ocr.predict(image)
            # For now, return a placeholder confidence score
            # In a full implementation, you'd calculate actual confidence
            confidence = 0.95  # Placeholder
            
            return {
                'text': text,
                'confidence': confidence
            }
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            return {
                'text': "",
                'confidence': 0.0
            }
    
    def batch_predict_with_confidence(self, images: List[Union[str, Image.Image]]) -> List[Dict[str, Union[str, float]]]:
        """
        Predict text with confidence for multiple images
        
        Args:
            images: List of image paths or PIL Image objects
            
        Returns:
            List of dictionaries with 'text' and 'confidence' keys
        """
        results = []
        for image in images:
            result = self.predict_with_confidence(image)
            results.append(result)
        
        return results
    
    def process_document(self, image_path: str, output_path: Optional[str] = None) -> Dict[str, str]:
        """
        Process a document image and save results
        
        Args:
            image_path: Path to document image
            output_path: Path to save results (optional)
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Load and process image
            image = Image.open(image_path)
            text = self.ocr.predict(image)
            
            results = {
                'image_path': image_path,
                'extracted_text': text,
                'status': 'success'
            }
            
            # Save results if output path provided
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            
            return results
            
        except Exception as e:
            error_result = {
                'image_path': image_path,
                'extracted_text': "",
                'status': 'error',
                'error': str(e)
            }
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(error_result, f, ensure_ascii=False, indent=2)
            
            return error_result

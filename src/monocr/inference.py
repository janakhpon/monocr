#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
advanced inference utilities for mon ocr
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import json
from typing import List, Dict, Optional, Union

from .ocr import MonOCR

class MonOCRInference:
    """advanced mon ocr inference with additional utilities"""
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = "crnn"):
        """initialize advanced mon ocr inference"""
        self.ocr = MonOCR(model_path, model_type)
    
    def predict_with_confidence(self, image: Union[str, Image.Image]) -> Dict[str, Union[str, float]]:
        """predict text with confidence score"""
        if isinstance(image, str):
            image = Image.open(image).convert("L")
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a file path or PIL Image")
        
        # get prediction
        predicted_text = self.ocr.predict(image)
        
        # calculate confidence (simplified)
        confidence = self._calculate_confidence(image, predicted_text)
        
        return {
            'text': predicted_text,
            'confidence': confidence
        }
    
    def _calculate_confidence(self, image: Image.Image, text: str) -> float:
        """calculate confidence score (simplified implementation)"""
        # simple confidence based on text length and image size
        if not text:
            return 0.0
        
        # normalize confidence based on text length and image dimensions
        text_length = len(text)
        image_area = image.width * image.height
        
        # simple heuristic: longer text on larger images = higher confidence
        confidence = min(1.0, (text_length * 100) / image_area)
        
        return max(0.0, min(1.0, confidence))
    
    def batch_predict_with_confidence(self, images: List[Union[str, Image.Image]]) -> List[Dict[str, Union[str, float]]]:
        """predict text with confidence for multiple images"""
        results = []
        for image in images:
            try:
                result = self.predict_with_confidence(image)
                results.append(result)
            except Exception as e:
                results.append({
                    'text': '',
                    'confidence': 0.0
                })
        
        return results
    
    def save_results(self, results: List[Dict[str, Union[str, float]]], output_path: str):
        """save prediction results to json file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def load_results(self, input_path: str) -> List[Dict[str, Union[str, float]]]:
        """load prediction results from json file"""
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
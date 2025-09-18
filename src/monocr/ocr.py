#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Mon OCR class - Production-ready OCR for Mon text
Supports both CRNN and TrOCR models
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
import json
import logging
from typing import List, Dict, Optional, Union
from torchvision import transforms

# TrOCR imports (optional)
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False

class MonOCR:
    """Production-ready Mon OCR class supporting both CRNN and TrOCR models"""
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = "crnn"):
        """
        Initialize Mon OCR
        
        Args:
            model_path: Path to trained model file (if None, uses bundled model)
            model_type: Type of model ("crnn" or "trocr")
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type.lower()
        self.model = None
        self.processor = None
        self.charset = None
        
        # Load model - use bundled model if no path provided
        if model_path is None:
            from . import get_default_model_path
            model_path = get_default_model_path()
        
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model from file"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if self.model_type == "crnn":
            self._load_crnn_model(model_path)
        elif self.model_type == "trocr":
            self._load_trocr_model(model_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _load_crnn_model(self, model_path: str):
        """Load CRNN model"""
        # Import CRNN model class (this would need to be included in the package)
        from .crnn_model import CRNN, build_charset
        
        # Load model state
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract charset from checkpoint or build from corpus
        if 'charset' in checkpoint:
            self.charset = checkpoint['charset']
        else:
            # Fallback: build charset from default corpus
            self.charset = build_charset("data/raw/corpus")
        
        # Initialize model (add 1 for blank token)
        self.model = CRNN(num_classes=len(self.charset) + 1)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
    
    def _load_trocr_model(self, model_path: str):
        """Load TrOCR model"""
        if not TROCR_AVAILABLE:
            raise ImportError("TrOCR dependencies not available. Install with: pip install transformers")
        
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, image: Union[str, Image.Image]) -> str:
        """
        Predict text from image
        
        Args:
            image: Path to image file or PIL Image object
            
        Returns:
            Predicted text string
        """
        if isinstance(image, str):
            image = Image.open(image).convert("L")
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a file path or PIL Image")
        
        if self.model_type == "crnn":
            return self._predict_crnn(image)
        elif self.model_type == "trocr":
            return self._predict_trocr(image)
    
    def _predict_crnn(self, image: Image.Image) -> str:
        """Predict using CRNN model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess image - match simple_inference.py exactly
        if isinstance(image, str):
            image = Image.open(image).convert('L')
        elif isinstance(image, Image.Image):
            image = image.convert('L')
        
        # Resize image - target_size is (height, width) for the model
        # PIL resize expects (width, height), so we need to swap
        image = image.resize((256, 64), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Apply the same transform as training
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        image_tensor = transform(image_tensor)
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            predicted_text = self._decode_crnn_output(outputs)
        
        return predicted_text
    
    def _predict_trocr(self, image: Image.Image) -> str:
        """Predict using TrOCR model"""
        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        
        # Predict
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
            predicted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return predicted_text
    
    def _decode_crnn_output(self, output: torch.Tensor) -> str:
        """Decode CRNN output to text - match simple_inference.py exactly"""
        if self.charset is None:
            raise ValueError("Charset not loaded")
        
        # Get predictions - same as working version
        preds = output.softmax(2).argmax(2).squeeze(0)  # [seq_len]
        
        # CTC decoding - exact same logic as working simple_inference.py
        decoded = []
        prev_char = None
        
        for idx in preds:
            idx = idx.item()
            if idx == 0:  # blank token
                prev_char = None
            elif idx != prev_char:  # new character
                if idx <= len(self.charset):  # idx is 1-based, charset is 0-based
                    decoded.append(self.charset[idx - 1])
                prev_char = idx
        
        return ''.join(decoded)
    
    def batch_predict(self, images: List[Union[str, Image.Image]]) -> List[str]:
        """
        Predict text from multiple images
        
        Args:
            images: List of image paths or PIL Image objects
            
        Returns:
            List of predicted text strings
        """
        results = []
        for image in images:
            try:
                result = self.predict(image)
                results.append(result)
            except Exception as e:
                logging.warning(f"Error processing image: {e}")
                results.append("")
        
        return results
    
    def read_text(self, image: Union[str, Image.Image]) -> str:
        """
        Read text from image (alias for predict method)
        
        Args:
            image: Path to image file or PIL Image object
            
        Returns:
            Extracted text string
        """
        return self.predict(image)
    
    def read_multiple(self, images: List[Union[str, Image.Image]]) -> List[str]:
        """
        Read text from multiple images (alias for batch_predict method)
        
        Args:
            images: List of image paths or PIL Image objects
            
        Returns:
            List of extracted text strings
        """
        return self.batch_predict(images)
    
    def read_from_folder(self, folder_path: str, extensions: List[str] = None) -> dict:
        """
        Read text from all images in a folder
        
        Args:
            folder_path: Path to folder containing images
            extensions: List of file extensions to process (default: ['.png', '.jpg', '.jpeg'])
            
        Returns:
            Dictionary mapping filename to extracted text
        """
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        results = {}
        image_files = []
        
        for ext in extensions:
            image_files.extend(folder.glob(f"*{ext}"))
            image_files.extend(folder.glob(f"*{ext.upper()}"))
        
        for image_file in image_files:
            try:
                text = self.read_text(str(image_file))
                results[image_file.name] = text
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")
                results[image_file.name] = ""
        
        return results

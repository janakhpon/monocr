"""
Mon OCR - Optical Character Recognition for Mon text
A production-ready OCR package for Mon script text recognition
"""

import os
from pathlib import Path
from .ocr import MonOCR
from .inference import MonOCRInference

__version__ = "0.1.0"
__author__ = "janakhpon"
__email__ = "jnovaxer@gmail.com"

__all__ = ["MonOCR", "MonOCRInference", "read_text", "read_image", "read_folder"]


def get_default_model_path():
    """Get the path to the bundled default model"""
    package_dir = Path(__file__).parent
    model_path = package_dir / "models" / "monocr_v1_best.pt"
    return str(model_path)


# Global OCR instance for simple API
_ocr_instance = None

def _get_ocr():
    """Get or create the global OCR instance"""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = MonOCR()
    return _ocr_instance


def read_text(image_path):
    """
    Read text from a single image - Simple API
    
    Args:
        image_path: Path to image file
        
    Returns:
        Extracted text string
    """
    return _get_ocr().read_text(image_path)


def read_image(image_path):
    """
    Alias for read_text - Read text from a single image
    
    Args:
        image_path: Path to image file
        
    Returns:
        Extracted text string
    """
    return read_text(image_path)


def read_folder(folder_path, extensions=None):
    """
    Read text from all images in a folder - Simple API
    
    Args:
        folder_path: Path to folder containing images
        extensions: List of file extensions to process (default: ['.png', '.jpg', '.jpeg'])
        
    Returns:
        Dictionary mapping filename to extracted text
    """
    return _get_ocr().read_from_folder(folder_path, extensions)


def load_ocr(model_path=None, model_type="crnn"):
    """
    Load OCR model with default settings (Advanced API)
    
    Args:
        model_path: Path to trained model file (if None, uses bundled model)
        model_type: Type of model ("crnn" or "trocr")
    
    Returns:
        MonOCR instance
    """
    if model_path is None:
        model_path = get_default_model_path()
    
    return MonOCR(model_path, model_type)

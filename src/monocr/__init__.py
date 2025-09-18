"""
mon ocr - optical character recognition for mon text
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
    """get bundled model path"""
    package_dir = Path(__file__).parent
    model_path = package_dir / "models" / "monocr_v1_best.pt"
    return str(model_path)


# global ocr instance for simple api
_ocr_instance = None

def _get_ocr():
    """get or create global ocr instance"""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = MonOCR()
    return _ocr_instance


def read_text(image_path):
    """read text from single image"""
    return _get_ocr().read_text(image_path)


def read_image(image_path):
    """alias for read_text"""
    return read_text(image_path)


def read_folder(folder_path, extensions=None):
    """read text from all images in folder"""
    return _get_ocr().read_from_folder(folder_path, extensions)


def load_ocr(model_path=None, model_type="crnn"):
    """load ocr model with custom settings"""
    if model_path is None:
        model_path = get_default_model_path()
    
    return MonOCR(model_path, model_type)
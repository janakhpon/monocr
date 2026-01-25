"""
mon ocr - optical character recognition for mon text
"""

import logging
from pathlib import Path
from .ocr import MonOCR
from .config import DEFAULT_MODEL_PATH
from .exceptions import MonOCRError, ModelNotFoundError, ImageLoadError

__version__ = "0.1.5"

# Set up null handler to prevent "No handler found" warnings
logging.getLogger(__name__).addHandler(logging.NullHandler())

def get_default_model_path():
    """get bundled v3 model path"""
    return str(DEFAULT_MODEL_PATH)

# Global instance for easy access
_ocr = None

def _get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = MonOCR(get_default_model_path())
    return _ocr

def read_text(image):
    """Recognize text from an image (supports single/multi-line)"""
    return _get_ocr().predict(image)

def read_folder(folder_path):
    """Recognize text from all images in a folder"""
    return _get_ocr().read_from_folder(folder_path)
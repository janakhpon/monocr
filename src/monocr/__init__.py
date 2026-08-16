"""
mon ocr - optical character recognition for mon text
"""

import logging
from pathlib import Path
from .ocr import MonOCR
from .config import DEFAULT_MODEL_PATH
from .exceptions import (
    MonOCRError,
    ModelNotFoundError,
    CharsetNotFoundError,
    ImageLoadError,
)

# Kept equal to [project.version] in pyproject.toml. `monocr --version` reads
# the installed metadata, not this string, so the two can drift apart silently —
# tests/test_packaging.py is what stops that.
__version__ = "2.3.0"

__all__ = [
    "MonOCR",
    "MonOCRError",
    "ModelNotFoundError",
    "CharsetNotFoundError",
    "ImageLoadError",
    "read_text",
    "read_folder",
    "get_default_model_path",
    "__version__",
]

# Set up null handler to prevent "No handler found" warnings
logging.getLogger(__name__).addHandler(logging.NullHandler())

def get_default_model_path():
    """Where a downloaded model ends up on this machine.

    Deprecated as an argument to MonOCR. huggingface_hub writes its own cache
    layout under CACHE_DIR, so this flat path is not where the download lands —
    passing it made MonOCR skip the download branch entirely and then fail to
    find the file, or worse, load an unrelated model left there by an older
    release. Call MonOCR() with no argument instead.
    """
    return str(DEFAULT_MODEL_PATH)

# Global instance for easy access
_ocr = None

def _get_ocr():
    global _ocr
    if _ocr is None:
        # No path: let MonOCR resolve and download the pinned revision.
        _ocr = MonOCR()
    return _ocr

def read_text(image):
    """Recognize text from an image (supports single/multi-line)"""
    return _get_ocr().predict(image)

def read_folder(folder_path):
    """Recognize text from all images in a folder"""
    return _get_ocr().read_from_folder(folder_path)
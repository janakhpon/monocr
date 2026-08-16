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

# Read from the installed metadata, so pyproject.toml is the only place a
# version is written.
#
# This was a hand-maintained literal until 2.3.1, kept equal to
# [project.version] by a test. That worked, but it made the documented release
# procedure wrong: `uv version --bump` edits pyproject.toml and nothing else, so
# following the README left this string behind, and the guard fired *after* the
# tag had been pushed. Deriving it removes the class of mistake instead of
# detecting it.
try:
    from importlib.metadata import PackageNotFoundError, version as _installed_version

    __version__ = _installed_version("monocr")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0+unknown"

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
#!/usr/bin/env python3
"""Model download utilities for monocr."""

import logging
from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download
from .config import CACHE_DIR

logger = logging.getLogger(__name__)


def get_cached_model_path(
    repo_id: str,
    filename: str,
    cache_dir: Optional[Path] = None,
    force_download: bool = False
) -> Path:
    """
    Get path to cached model, downloading from Hugging Face Hub if necessary.
    
    Args:
        repo_id: Hugging Face repository ID (e.g. 'janakhpon/monocr')
        filename: Filename in the repository (e.g. 'monocr.ckpt')
        cache_dir: Directory to cache models (optional)
        force_download: Force re-download even if file exists
        
    Returns:
        Path to the model file
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR
        
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            force_download=force_download
        )
        return Path(model_path)
    except Exception as e:
        logger.error(f"oops, cannot download from hugging face: {e}")
        raise e


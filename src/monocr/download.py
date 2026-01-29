#!/usr/bin/env python3
"""Model download utilities for monocr."""

import hashlib
import logging
import urllib.request
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_with_progress(url: str, dest_path: Path) -> None:
    """Download file with simple progress indication."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    temp_path = dest_path.with_suffix('.tmp')
    
    try:
        logger.info(f"Downloading model from {url}")
        print(f"Downloading model from {url}")
        
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            block_size = 8192
            
            with open(temp_path, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    
                    downloaded += len(buffer)
                    f.write(buffer)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
        
        print()  # New line after progress
        
        # Move temp file to final destination
        shutil.move(str(temp_path), str(dest_path))
        logger.info(f"Download complete: {dest_path}")
        
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise Exception(f"Download failed: {e}")


def verify_checksum(file_path: Path, expected_sha256: str) -> bool:
    """Verify file checksum."""
    if not file_path.exists():
        return False
    
    actual = compute_sha256(file_path)
    return actual == expected_sha256


def download_model(
    url: str,
    dest_path: Path,
    expected_sha256: Optional[str] = None,
    force: bool = False
) -> Path:
    """
    Download model file if not cached.
    
    Args:
        url: URL to download from
        dest_path: Destination path for the model
        expected_sha256: Expected SHA256 hash for verification
        force: Force re-download even if file exists
        
    Returns:
        Path to the downloaded model
    """
    # Check if model already exists and is valid
    if dest_path.exists() and not force:
        if expected_sha256:
            if verify_checksum(dest_path, expected_sha256):
                logger.info(f"Using cached model: {dest_path}")
                return dest_path
            else:
                logger.warning(f"Cached model checksum mismatch. Re-downloading.")
        else:
            logger.info(f"Using cached model: {dest_path}")
            return dest_path
    
    # Download the model
    download_with_progress(url, dest_path)
    
    # Verify checksum if provided
    if expected_sha256:
        if not verify_checksum(dest_path, expected_sha256):
            dest_path.unlink()
            raise Exception(f"Downloaded file checksum mismatch. Download may be corrupted.")
        logger.info("Checksum verified")
    
    return dest_path


def get_cached_model_path(
    cache_dir: Path,
    model_name: str,
    model_url: str,
    model_sha256: Optional[str] = None
) -> Path:
    """
    Get path to cached model, downloading if necessary.
    
    Args:
        cache_dir: Directory to cache models
        model_name: Name of the model file
        model_url: URL to download from if not cached
        model_sha256: Expected SHA256 hash
        
    Returns:
        Path to the model file
    """
    model_path = cache_dir / model_name
    
    if not model_path.exists():
        print(f"Model not found in cache. Downloading to {cache_dir}")
        download_model(model_url, model_path, model_sha256)
    
    return model_path

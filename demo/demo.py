#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mon OCR - Demonstration Script
Version 0.1.2
"""

import os
import sys
from pathlib import Path
from typing import Dict

try:
    from monocr import read_text, read_folder, __version__
except ImportError:
    print("Error: MonOCR is not installed in the current environment.")
    print("Please run 'uv sync' or 'pip install -e .'")
    sys.exit(1)

def print_header(title: str):
    print(f"\n{'=' * 40}")
    print(f" {title}")
    print(f"{'=' * 40}")

def main():
    print_header(f"Mon OCR v{__version__} Demo")
    
    demo_dir = Path(__file__).parent
    images_dir = demo_dir / "images"
    
    if not images_dir.exists():
        print(f"Error: Demo images directory not found at {images_dir}")
        return
    
    # Supported image extensions
    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in valid_extensions]
    
    if not image_files:
        print(f"No valid images found in {images_dir}")
        return
    
    print(f"[*] Found {len(image_files)} sample images.")
    
    # 1. Single Image Recognition
    print_header("1. Single Image Recognition")
    sample_img = image_files[0]
    print(f"[*] Processing: {sample_img.name}")
    
    try:
        text = read_text(str(sample_img))
        print(f"[+] Result: {text}")
    except Exception as e:
        print(f"[!] Error processing image: {e}")
    
    # 2. Batch Processing
    print_header("2. Batch Processing (Folder)")
    print(f"[*] Processing folder: {images_dir.name}/")
    
    try:
        results: Dict[str, str] = read_folder(str(images_dir))
        for filename, text in sorted(results.items()):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                print(f"  - {filename:30} : {text}")
    except Exception as e:
        print(f"[!] Error in batch processing: {e}")
    
    # Documentation & Quick Start
    print_header("Quick Start Guide")
    print("To use Mon OCR in your project:")
    print("-" * 30)
    print("from monocr import read_text")
    print("text = read_text('path/to/image.png')")
    print("\nCommand Line Interface:")
    print("-" * 30)
    print("monocr read path/to/image.png")
    print("monocr batch path/to/folder/")
    print("=" * 40)

if __name__ == "__main__":
    main()
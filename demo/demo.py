#!/usr/bin/env python3
"""
Mon OCR Demo
"""

import os
from pathlib import Path
from monocr import read_text, read_folder

def main():
    print("Mon OCR Demo")
    print("=" * 20)
    
    demo_dir = Path(__file__).parent
    images_dir = demo_dir / "images"
    
    if not images_dir.exists():
        print("No demo images found")
        return
    
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    
    if not image_files:
        print("No images found")
        return
    
    print(f"Found {len(image_files)} images")
    print()
    
    # Read single image
    first_image = image_files[0]
    try:
        text = read_text(str(first_image))
        print(f"Image: {first_image.name}")
        print(f"Text: {text}")
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Read all images
    try:
        results = read_folder(str(images_dir))
        for filename, text in results.items():
            print(f"{filename}: {text}")
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    print("Usage:")
    print("  from monocr import read_text, read_folder")
    print("  text = read_text('image.png')")
    print("  results = read_folder('images/')")
    print()
    print("CLI:")
    print("  monocr read image.png")
    print("  monocr batch images/")

if __name__ == "__main__":
    main()
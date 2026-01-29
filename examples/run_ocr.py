#!/usr/bin/env python3
"""
Mon OCR - Inference Example
Supports Images and PDFs
"""

import sys
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
from monocr import MonOCR

def process_image(ocr, img, name):
    print(f"  Running {name}")
    try:
        result = ocr.predict_with_confidence(img)
        print(f"  Conf: {result['confidence']:.1%}")
        # Show first line if long
        text = result['text'].strip()
        lines = text.split('\n')
        print(f"  Text: {lines[0] if lines else '[Empty]'}")
        if len(lines) > 1:
            print(f"        ... ({len(lines)-1} more)")
    except Exception as e:
        print(f"  Failed: {e}")

def main():
    print("-" * 60)
    print("Mon OCR (Images & PDFs)".center(60))
    print("-" * 60)
    
    # Init OCR
    try:
        ocr = MonOCR()
        print(f"Ready on {ocr.device}")
    except Exception as e:
        print(f"Init failed: {e}")
        sys.exit(1)
    
    # Find files
    images_dir = Path(__file__).parent / "images"
    if not images_dir.exists():
        print(f"No folder: {images_dir}")
        return

    # Collect files
    files = sorted(
        list(images_dir.glob("*.png")) + 
        list(images_dir.glob("*.jpg")) + 
        list(images_dir.glob("*.jpeg")) +
        list(images_dir.glob("*.pdf"))
    )
    
    if not files:
        print("No files")
        return

    print(f"Found {len(files)} files\n")
    
    for f in files:
        print(f"File: {f.name}")
        
        if f.suffix.lower() == '.pdf':
            try:
                print(f"  Converting PDF")
                pages = convert_from_path(str(f))
                print(f"  {len(pages)} pages")
                for i, page in enumerate(pages, 1):
                    process_image(ocr, page, f"Page {i}")
            except Exception as e:
                print(f"  PDF failed: {e}")
        else:
            # Regular Image
            process_image(ocr, str(f), "Image")
            
        print("-" * 60)

if __name__ == "__main__":
    main()
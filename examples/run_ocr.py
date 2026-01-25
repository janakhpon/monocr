#!/usr/bin/env python3
"""
Mon OCR - Inference Example
"""

import sys
from pathlib import Path
from monocr import MonOCR

def main():
    print("-" * 50)
    print("Mon OCR Inference".center(50))
    print("-" * 50)
    
    # Initialize with default model
    try:
        ocr = MonOCR()
        print(f"Model loaded on {ocr.device}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Setup images directory
    images_dir = Path(__file__).parent / "images"
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return

    # Process all images
    image_files = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))
    
    if not image_files:
        print("No images found to process.")
        return

    print(f"Found {len(image_files)} images.\n")
    
    for img_path in image_files:
        print(f"File: {img_path.name}")
        try:
            # Run prediction
            result = ocr.predict_with_confidence(str(img_path))
            
            # Print simplified output
            print(f"Conf: {result['confidence']:.1%}")
            print(f"Text: {result['text']}")
            
            # Ground truth verification (optional)
            gt_path = img_path.with_suffix('.txt')
            if not gt_path.exists():
                gt_path = img_path.parent / (img_path.stem + ".gt.txt")
                
            if gt_path.exists():
                with open(gt_path, 'r', encoding='utf-8') as f:
                    gt = f.read().strip()
                match = "MATCH" if result['text'].strip() == gt else "MISMATCH"
                print(f"GT:   {gt} [{match}]")
                
        except Exception as e:
            print(f"Error: {e}")
            
        print("-" * 50)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
from monocr import read_text, read_folder

def main():
    print("Mon OCR Example")
    print("=" * 20)
    
    # Read single image
    print("1. Single image:")
    try:
        text = read_text("demo/images/word_003_ဘာသာမန်.png")
        print(f"   Text: {text}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print()
    
    # Read folder
    print("2. Folder:")
    try:
        results = read_folder("demo/images/")
        for filename, text in results.items():
            if filename.endswith('.png'):
                print(f"   {filename}: {text}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print()
    print("Usage:")
    print("  from monocr import read_text")
    print("  text = read_text('image.png')")

if __name__ == "__main__":
    main()
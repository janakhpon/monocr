#!/usr/bin/env python3
"""
Command Line Interface for Mon OCR
"""

import click
import json
from pathlib import Path
from typing import List

from .ocr import MonOCR
from .inference import MonOCRInference
from . import get_default_model_path

@click.group()
@click.version_option()
def main():
    """Mon OCR - Optical Character Recognition for Mon text"""
    pass

@main.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--model', '-m', help='Path to trained model file (default: uses bundled model)')
@click.option('--model-type', type=click.Choice(['crnn', 'trocr']), default='crnn', help='Type of model to use')
@click.option('--output', '-o', help='Output file to save results')
def read(image_path: str, model: str, model_type: str, output: str):
    """Read text from a single image"""
    try:
        if model is None:
            model = get_default_model_path()
        ocr = MonOCR(model, model_type)
        
        print("Processing image...")
        text = ocr.read_text(image_path)
        
        print(f"\nExtracted text:")
        print(text)
        
        if output:
            result = {
                'image_path': image_path,
                'extracted_text': text,
                'model_type': model_type
            }
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {output}")
            
    except Exception as e:
        print(f"Error: {e}")
        raise click.Abort()

@main.command()
@click.argument('folder_path', type=click.Path(exists=True, file_okay=False))
@click.option('--model', '-m', help='Path to trained model file (default: uses bundled model)')
@click.option('--model-type', type=click.Choice(['crnn', 'trocr']), default='crnn', help='Type of model to use')
@click.option('--output', '-o', help='Output file to save results')
@click.option('--extensions', default='png,jpg,jpeg', help='File extensions to process (comma-separated)')
def batch(folder_path: str, model: str, model_type: str, output: str, extensions: str):
    """Read text from all images in a folder"""
    try:
        if model is None:
            model = get_default_model_path()
        ocr = MonOCR(model, model_type)
        ext_list = [f'.{ext.strip()}' for ext in extensions.split(',')]
        
        print("Processing folder...")
        results = ocr.read_from_folder(folder_path, ext_list)
        
        print("\nOCR Results:")
        print("-" * 40)
        for filename, text in results.items():
            print(f"{filename}: {text}")
        
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {output}")
            
    except Exception as e:
        print(f"Error: {e}")
        raise click.Abort()

@main.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--model', '-m', help='Path to trained model file (default: uses bundled model)')
@click.option('--model-type', type=click.Choice(['crnn', 'trocr']), default='crnn', help='Type of model to use')
def confidence(image_path: str, model: str, model_type: str):
    """Read text with confidence score"""
    try:
        ocr = MonOCRInference(model, model_type)
        
        print("Processing image...")
        result = ocr.predict_with_confidence(image_path)
        
        print(f"\nExtracted text:")
        print(result['text'])
        print(f"\nConfidence: {result['confidence']:.2%}")
            
    except Exception as e:
        print(f"Error: {e}")
        raise click.Abort()

if __name__ == '__main__':
    main()
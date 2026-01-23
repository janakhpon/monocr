#!/usr/bin/env python3
"""
command line interface for mon ocr
"""

import click
import json
from pathlib import Path
from typing import List

from .ocr import MonOCR
from . import get_default_model_path

@click.group()
@click.version_option()
def main():
    """mon ocr - optical character recognition for mon text"""
    pass

@main.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--model', '-m', help='Path to trained model file.')
@click.option('--model-type', type=click.Choice(['crnn', 'trocr']), default='crnn', help='Type of model.')
@click.option('--confidence', '-c', is_flag=True, help='Return confidence score.')
def read(image_path: str, model: str, model_type: str, confidence: bool):
    """Read text from an image."""
    try:
        model = model or get_default_model_path()
        ocr = MonOCR(model, model_type)
        
        if confidence:
            result = ocr.predict_with_confidence(image_path)
            print(f"Text: {result['text']}")
            print(f"Confidence: {result['confidence']:.2%}")
        else:
            print(ocr.predict(image_path))
            
    except Exception as e:
        print(f"Error: {e}")
        raise click.Abort()

@main.command()
@click.argument('folder_path', type=click.Path(exists=True, file_okay=False))
@click.option('--model', '-m', help='Path to trained model file.')
def batch(folder_path: str, model: str):
    """Batch process a folder of images."""
    try:
        model = model or get_default_model_path()
        ocr = MonOCR(model)
        results = ocr.read_from_folder(folder_path)
        
        for filename, text in sorted(results.items()):
            print(f"{filename:30}: {text}")
            
    except Exception as e:
        print(f"Error: {e}")
        raise click.Abort()

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
import click
import logging
import sys
from .ocr import MonOCR
from .config import DEFAULT_MODEL_PATH, HF_REPO_ID, HF_FILENAME
from .download import get_cached_model_path
from .exceptions import MonOCRError

@click.group()
@click.version_option()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(verbose):
    """mon ocr - simple and effective text recognition for Mon language"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

@main.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--confidence', '-c', is_flag=True, help='Show confidence score')
def read(image_path, confidence):
    """Read text from an image."""
    try:
        ocr = MonOCR(str(DEFAULT_MODEL_PATH))
        if confidence:
            res = ocr.predict_with_confidence(image_path)
            click.echo(f"Text:\n{res['text']}")
            click.echo(f"Confidence: {res['confidence']:.2%}")
        else:
            text = ocr.predict(image_path)
            if text.strip():
                click.echo(text)
            else:
                click.echo("Hmm, couldn't find any text there.")
    except MonOCRError as e:
        logging.error(f"OCR Error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Unexpected error: {e}")
        sys.exit(1)

@main.command()
def download():
    """Download the model manually."""
    try:
        click.echo("Checking for the latest model... wait ah.")
        path = get_cached_model_path(repo_id=HF_REPO_ID, filename=HF_FILENAME, force_download=True)
        click.echo(f"Done! Model is safe at: {path}")
    except Exception as e:
        logging.error(f"cannot download: {e}")
        sys.exit(1)

@main.command()
@click.argument('folder_path', type=click.Path(exists=True, file_okay=False))
def batch(folder_path):
    """Process a folder of images."""
    try:
        ocr = MonOCR(str(DEFAULT_MODEL_PATH))
        results = ocr.read_from_folder(folder_path)
        if not results:
            click.echo("No images found in that folder, buddy.")
            return
        
        click.echo(f"Found {len(results)} images. Starting now...")
        for path, text in sorted(results.items()):
            if text.strip():
                click.echo(f"--- {path} ---")
                click.echo(text)
                click.echo("")
        click.echo("All done!")
    except MonOCRError as e:
        logging.error(f"Batch processing error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
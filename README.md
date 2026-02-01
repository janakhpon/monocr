# Mon OCR

Optical Character Recognition for Mon (mnw) text.

## Installation

```bash
pip install monocr | uv add monocr
```

## Quick Start

### Python Usage

```python
from monocr import MonOCR

# Initialize
model = MonOCR()

# 1. Read an Image
text = model.read_text("image.png")
print(text)

# 2. Read with Confidence
result = model.predict_with_confidence("image.png")
print(f"Text: {result['text']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Examples

See the [`examples/`](examples/) folder to learn more.

- **`examples/run_ocr.py`**: A complete script that can process a folder of images or read a full PDF book.

### CLI Usage

You can also use the command line interface:

```bash
# Process a single image
monocr read image.png

# Process a folder of images
monocr batch folder/path

# Manually download the model
monocr download
```

## License

MIT - do whatever you want with it.

## Dev Setup

```bash
git clone git@github.com:janakhpon/monocr.git
cd monocr
uv sync --dev
```

### Release Workflow

```bash
uv version --bump patch
uv build
git add .
git commit -m "bump version"
git tag v0.1.17
git push origin main --tags
```

## Resources

- [monocr on pypi](https://pypi.org/project/monocr/)
- [monocr on hugging face](https://huggingface.co/janakhpon/monocr)

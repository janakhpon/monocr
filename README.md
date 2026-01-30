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

## Related Tools

- [mon_tokenizer](https://github.com/Code-Yay-Mal/mon_tokenizer)
- [hugging face mon_tokenizer model](https://huggingface.co/janakhpon/mon_tokenizer)
- [Mon corpus collection in unicode](https://github.com/MonDevHub/MonCorpusCollection)

## License

MIT - do whatever you want with it.

## Dev Setup

```bash
git clone git@github.com:janakhpon/monocr.git
cd monocr
uv sync --dev
```

### Update Model in Hugging Face

To update the model weights:

```bash
# 1. Login to HF
hf auth login

# 2. Upload from your local model folder
hf upload janakhpon/monocr path/to/model_dir --repo-type model
```

### Release Workflow

```bash
uv version --bump patch
git add .
git commit -m "bump version"
git tag v0.1.16
git push origin main --tags
```

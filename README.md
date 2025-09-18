# Mon OCR

Optical Character Recognition for Mon (mnw) text.

## Installation

```bash
pip install monocr | uv add monocr
```

## Quick Start

```python
from monocr import read_text, read_folder

# Read text from a single image
text = read_text("image.png")
print(text)

# Read all images in a folder
results = read_folder("images/")
for filename, text in results.items():
    print(f"{filename}: {text}")
```

## Command Line

```bash
# Read single image
monocr read image.png

# Process folder
monocr batch images/ --output results.json
```

## Dev Setup

```bash
git clone git@github.com:janakhpon/monocr.git
cd monocr
uv sync --dev

# Release workflow
uv version --bump patch
git add .
git commit -m "bump version"
git tag v0.1.5
git push origin main --tags
```

## Related tools
- [mon_tokenizer](https://github.com/Code-Yay-Mal/mon_tokenizer)
- [hugging face mon_tokenizer model](https://huggingface.co/janakhpon/mon_tokenizer)
- [Mon corpus collection in unicode](https://github.com/MonDevHub/MonCorpusCollection)

## License

MIT - do whatever you want with it.

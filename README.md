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

## License

MIT License

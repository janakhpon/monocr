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

# Initialize. Downloads the model pinned in config.HF_REVISION on first use.
model = MonOCR()

# A page: segmented into lines, joined with newlines
print(model.predict_page("page.png"))

# A line crop: read whole, never split
print(model.predict_line("line.png"))

# With a confidence score
result = model.predict_with_confidence("page.png")
print(f"Text: {result['text']}")
print(f"Confidence: {result['confidence']:.2%}")
```

`predict` and `read_text` are aliases for `predict_page`. If you are passing
single-line crops, call `predict_line` — it is the only path that cannot split
one line into several.

### Examples

See the [`examples/`](examples/) folder to learn more.

- **`examples/run_ocr.py`**: A complete script that can process a folder of images or read a full PDF book.
- Or a demo notebook to play around with the package [`notebooks/demo.ipynb`](https://github.com/janakhpon/preview_monocr/blob/main/notebooks/demo.ipynb)

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

## Resources

- [monocr on pypi](https://pypi.org/project/monocr/)
- [monocr on hugging face](https://huggingface.co/janakhpon/monocr)
- [Changelog](CHANGELOG.md)

## Development

```bash
uv sync
uv run pytest
```

### Release Workflow

The tag has to match `[project.version]` in `pyproject.toml`. The release
workflow checks, and fails the build if it does not — this section used to
show `git tag v2.2.3` next to a version of `2.2.0`, and nothing would have
stopped that going out.

```bash
uv version --bump patch          # or minor / major
git commit -am "release: $(uv version --short)"
git tag "v$(uv version --short)"
git push origin HEAD --tags
```

Publishing runs from the tag, through GitHub Actions, using PyPI trusted
publishing. It installs from the lockfile and runs the test suite first.

## License

MIT - do whatever you want with it.

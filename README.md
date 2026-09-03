# Mon OCR

[![PyPI](https://img.shields.io/pypi/v/monocr.svg)](https://pypi.org/project/monocr/)

Optical Character Recognition for Mon (mnw) text.

Mon is classified as **vulnerable** in UNESCO's Atlas of the World's Languages
in Danger. General OCR covers its *script* but not its *language*, and the
difference is the reason this package exists.

Mon is written in the Mon–Burmese script, so a Burmese recogniser will read a
Mon page and return text. Google Cloud Vision lists Burmese (`my`, `Mymr`) as an
experimental OCR language and has no entry for Mon; Cloud Translation and
Google Translate's 2024 expansion both ship Burmese and not Mon; Tesseract's
`Myanmar.traineddata` is a script model, which by design covers languages it was
never trained on.

What that misses is specific and checkable. Ten of the 276 characters this model
emits are named for Mon in Unicode precisely because Burmese does not use them:
`U+1028` MON E, `U+1033` MON II, `U+1034` MON O, and `U+105A`–`U+1060` (MON NGA,
JHA, BBA, BBE, and the MON MEDIAL NA, MA and LA signs). A Burmese-trained model
has no output class for any of them, so it emits the nearest Burmese glyph
instead. The result is fluent-looking Burmese-Mon hybrid text with nothing
raised — wrong in a way that is hard to notice unless you read Mon.

This package is a Mon recogniser you can `pip install`, running locally on ONNX
Runtime. It reads printed Mon: rendered pages and scans of them. Handwriting is
out of scope. The held-out result and, more importantly, the four things it does
not cover are on the [model card](https://huggingface.co/janakhpon/monocr).

## Installation

```bash
pip install monocr
```

or, in a uv project:

```bash
uv add monocr
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

`pyproject.toml` holds the only version in the tree; `monocr.__version__` reads
it back from the installed metadata. The tag has to match it, and the release
workflow fails the build if it does not.

```bash
uv version --bump patch          # or minor / major
git commit -am "release: $(uv version --short)"
git tag "v$(uv version --short)"
git push origin HEAD --tags
```

Two things this section got wrong before, both caught by running it:

- It showed `git tag v2.2.3` beside a version of `2.2.0`. Nothing would have
  stopped that going out; `release.yml` now compares them.
- These four commands alone used to leave a release broken. `__version__` was a
  separate literal in `__init__.py`, `uv version --bump` does not touch it, and
  the test that noticed ran *after* the tag was already pushed. The version is
  derived now, so the commands above are the whole procedure.

Publishing runs from the tag, through GitHub Actions, using PyPI trusted
publishing. It installs from the lockfile and runs the test suite first.

## License

MIT - do whatever you want with it.

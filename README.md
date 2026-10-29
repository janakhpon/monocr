# Mon OCR

[![PyPI](https://img.shields.io/pypi/v/monocr.svg)](https://pypi.org/project/monocr/)

Optical character recognition for Mon (mnw) text: a Python package that runs
locally on ONNX Runtime. It is part of the Mon OCR toolchain
([how the pieces fit](https://github.com/MonDevHub)) and has its own line
segmentation.

Mon was listed as **vulnerable** in UNESCO's 2010 Atlas of the World's Languages
in Danger. It is written in the Mon–Burmese script, so a Burmese recogniser will
read a Mon page and return text, but not the right text. Ten of the 276
characters this model emits are named for Mon in Unicode because Burmese does
not use them: `U+1028` MON E, `U+1033` MON II, `U+1034` MON O, and
`U+105A`–`U+1060` (MON NGA, JHA, BBA, BBE, and the MON MEDIAL NA, MA and LA
signs). A Burmese-trained model has no output class for any of them, so it emits
the nearest Burmese glyph instead: fluent-looking Burmese-Mon hybrid text with
nothing raised, wrong in a way that is hard to notice unless you read Mon.

## What to expect

- It reads printed Mon. Accuracy is measured on rendered lines only; scans are
  unmeasured, and camera photographs are out of domain and fail with high
  confidence. Handwriting is out of scope.
- The held-out result and, more importantly, the four things it does not cover
  are on the [model card](https://huggingface.co/janakhpon/monocr).
- The model is pinned to revision `d3d9d5e` of
  [janakhpon/monocr](https://huggingface.co/janakhpon/monocr)
  (`config.HF_REVISION`) and downloaded on first use, so an upload to `main`
  cannot change what an installed copy reads.

## Install

Requires Python 3.11+.

```bash
pip install monocr
# or, in a uv project
uv add monocr
```

## Quick start

```python
from monocr import MonOCR

# Downloads the pinned model on first use.
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
single-line crops, call `predict_line`; it is the only path that cannot split
one line into several.

The confidence is the mean top-class probability over every output step of a
line, blanks and padding included, averaged over the lines that produced text
(0.0 if none did). Padding scores close to 1, so short lines score higher, and it
stays high on out-of-domain input such as photographs. It is not a measured
accuracy, and not a usable filter across lines of different length or on
out-of-domain input.

## CLI

```bash
monocr read image.png        # add --confidence for the score
monocr batch folder/path     # top-level *.png, *.jpg, *.jpeg (lowercase only on macOS and Linux)
monocr download              # fetch the model ahead of time
```

Both `read` and `batch` segment each image as a page. For line crops, call
`predict_line` from Python.

## Examples

- [`examples/run_ocr.py`](https://github.com/janakhpon/monocr/blob/main/examples/run_ocr.py)
  reads every image and PDF in `examples/images/` of a clone of this repository;
  it takes no arguments. PDFs need `pip install "monocr[examples]"` and poppler.
- A demo notebook,
  [`notebooks/demo.ipynb`](https://github.com/janakhpon/preview_monocr/blob/main/notebooks/demo.ipynb),
  written for monocr 0.1.17, an earlier PyTorch release, not this one.

## How it works

The page is converted to grayscale, inverted if it is light text on a dark
background, and segmented into lines with a horizontal projection profile. Each
line is resized to the model's input height, padded to 1024 pixels wide, read by
the ONNX graph and decoded greedily against the charset published beside the
weights at the same revision, or, when it was not downloaded, against the copy
bundled with the package, which is byte-identical. Loading refuses a model and
charset whose sizes disagree, because that pair would decode to the wrong
characters without raising.

A line wider than the model window (1024 by 160 pixels, so 6.4 times as wide as
it is tall) is squeezed to fit rather than split into tiles. The model card
measures squeezing as better up to about twice that width, level at three times,
and worse from four; for longer lines, cut them at word gaps and call
`predict_line` on each piece.

## Related

- [monocr-onnx](https://github.com/MonDevHub/monocr-onnx) — the same model with
  Python (`pip install monocr-onnx`), JavaScript, Go and Rust bindings. Its line
  segmenters differ from this package's, so page output can differ.
- [The apps](https://github.com/MonDevHub/monocr) built on the toolchain.
- [Changelog](https://github.com/janakhpon/monocr/blob/main/CHANGELOG.md)

## Development

```bash
uv sync
uv run pytest
```

- A change that can alter the text this package returns (segmentation,
  preprocessing, decoding or a new model revision) ships only with an A/B
  comparison that shows it is better:
  [docs/changing-output.md](https://github.com/janakhpon/monocr/blob/main/docs/changing-output.md).
- Releases are cut from a version tag and published by GitHub Actions:
  [docs/releasing.md](https://github.com/janakhpon/monocr/blob/main/docs/releasing.md).

## License

MIT

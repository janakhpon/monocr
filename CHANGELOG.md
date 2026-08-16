# Changelog

## 2.3.0 — 2026-08-16

The 2.2.0 release fixed a decode defect that this repository had no way of
catching. This one builds the way of catching it.

### The hole that let 2.1.2 out

`release.yml` was checkout, build, publish, with nothing in between, and there
was no test workflow at all — so no change to this repository could produce a
red build. The suite it would have run was nine tests, of which four could not
execute at all: `unittest.main()` sat at line 80 of `tests/test_monocr.py`,
above them.

- Added `.github/workflows/test.yml`, running pytest on 3.11 and 3.13.
- `release.yml` now installs from the lockfile and runs the suite before
  publishing, and refuses a tag that disagrees with `pyproject.toml`.
- Added a CI job that installs the built wheel in `python:3.12-slim` and
  imports it, which is the only place the headless-OpenCV claim below is
  actually testable.
- Tests: 9 to 89, none of which reach the network or load a model. Verified by
  mutation: 18 deliberate reversions of the fixes in this release, all caught.
  The harness is committed at `scripts/mutate.py`, so the claim can be re-run
  rather than taken on trust.

### Fixed

- **`opencv-python` → `opencv-python-headless`.** `cv2` is imported
  unconditionally by `segmenter.py`, so the GUI build made `import monocr` fail
  outright on any image missing the X libraries OpenCV links against — which is
  most containers. Measured on `python:3.12-slim`: 2.2.0 raises `ImportError:
  libxcb.so.1`, 2.3.0 imports.
- **Line segmentation cropped the full page width.** `_extract_line`, which
  trims left and right whitespace and pads relative to the line's own height,
  was defined in `segmenter.py` and never called; `segment` used a full-width
  crop with a fixed 4-pixel pad instead. That handed the model a strip with the
  page's aspect ratio, and the resize then squeezed the text to fit. The live
  path now calls it, matching `monocr-onnx`, which has always done this.
- **`predict` dispatched on `img.height > 100`.** The model takes a 160-pixel
  input, so an ordinary line crop at native resolution was tall enough to be
  routed down the page path. There is no height that separates a line from a
  page, so the choice is now explicit: `predict_line` for a crop,
  `predict_page` for a page.
- **An empty segmentation returned `""`.** Reporting no text for a page the
  segmenter could not split is indistinguishable from reporting a blank page.
  It now falls back to reading the image whole.
- **`examples/run_ocr.py` had never run.** It printed `ocr.device`, which does
  not exist, from inside a `try` whose handler then reported "Init failed" for
  an init that had succeeded. It also imported `pdf2image` — an optional extra
  needing poppler — at module scope, so the image path failed on a plain
  install too.
- **`CharsetNotFoundError` was not importable from the top level**, despite
  being the exception that signals a charset/model mismatch.

### Changed

- `predict` is now an alias for `predict_page`. Callers passing single-line
  crops should move to `predict_line`, which cannot split one line into
  several.
- Removed `PROJECTION_THRESHOLD`, `MIN_LINE_GAP` and `BINARY_THRESHOLD` from
  `config.py`. They described an algorithm this package does not run —
  `LineSegmenter` reads nothing from that module — and three numbers that look
  authoritative and are never used are worse than none. `IMAGE_NORM_MEAN` and
  `IMAGE_NORM_STD` are now used by the preprocessor rather than duplicated
  there as a literal.
- Untracked `src/monocr/models/monocr.ckpt`, an LFS pointer to 177MB with no
  reader: the branch that would have loaded it looks for a `.onnx`.

## 2.2.0 — 2026-08-15

Moved to the v3.5 model and made the charset/model pair impossible to get
wrong.

- Pinned the download to Hugging Face revision `d3d9d5e`. `hf_hub_download`
  resolved `main` until this release, so any upload to the model repository
  changed what already-installed copies fetched, with no version bump and no
  way for a user to tell.
- The charset is now fetched from the same revision as the weights, with the
  bundled copy as an offline fallback.
- `load_model` refuses a charset whose length cannot match the graph's class
  count, rather than decoding into the wrong alphabet.
- The input height is read off the graph instead of a constant.

### The defect this fixed

2.1.2 shipped a 914-character `valid_chars.txt` against a 316-class model, read
with `.strip()` — which removes the charset's leading U+0020 and shifts every
index by one. Both faults were independently sufficient. All 315 decodable
indices returned the wrong character, so the package answered Mon input with
fluent-looking text in the wrong alphabet and raised nothing. Anyone who
installed 2.1.0 through 2.1.2 got that.

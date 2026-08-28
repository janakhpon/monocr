# Changelog

## Unreleased

Three correctness fixes to the inference path. None is released yet, and the
newest released section below describes a segmenter with no suppression pass and an
inference path with no polarity probe — which is why this section exists rather
than waiting for a version bump.

### The model was fed pages it could not read

**Input polarity is now detected.** `_prepare_image` did `image.convert("L")` and
nothing else, so a light-on-dark scan or a dark-mode screenshot reached a model
trained only on dark text on a light background.

Measured over 300 labelled crops, same graph, only the polarity of the input
changed:

| input | with a probe | without |
|---|---:|---:|
| upright | 0.0000 | 0.0036 |
| inverted | 0.0000 | **0.0342** |

9.5x worse on inverted input — degradation rather than the total failure it might
sound like, and cheap to close for four corner patches. Corner-median rather than a
global mean, because a page 64% covered in ink has a mean below 128 and a global
test would invert an ordinary dense page.

The probe runs in `_prepare_image`, i.e. **before** `_segment_lines`. That ordering
matters: the segmenter treats dark as ink, so handed a light-on-dark page it would
segment the background and return the gaps between lines. Three sibling bindings got
this wrong by putting the probe in their per-crop preprocessing; this package did
not.

**Printed page rules are now suppressed** before the projection profile. A page
border adds a constant ink floor to every row it spans, and once that floor clears
the gap threshold no in-frame row reads as a gap: the page returns as one band and
is squeezed into the model window.

Measured through this segmenter over the twelve renderable MNEC page-ones: bands
returned went from 124 to 169, with four of the twelve collapsing to three bands or
fewer without it. The per-page table is in
`tests/test_page_rules.py::test_segment_wires_in_the_suppression`, which is the
record of this measurement — this line said 131 to 197 until 2026-08-28, a roll-up
figure taken from a sibling repository's header that nothing itemises and that
disagrees with the numbers this package landed beside its own code. Pages carrying
no rules are untouched to the pixel, which is what makes the pass safe to run
unconditionally.

Two guards, both with their measurements in the source: `RULE_SPAN = 0.5`, because
no Mon, Burmese or Latin glyph holds an unbroken stroke half a page long; and
`RULE_MAX_INK_SHARE = 0.80`, because `RULE_SPAN` is a fraction of the page, so on a
short page a tall text block exceeds it vertically and every glyph column reads as a
rule.

**The row scan no longer runs past the bottom of the page.**
`np.convolve(..., mode="same")` never returns fewer elements than its kernel, so on
a page shorter than `smooth_kernel` the smoothed profile described rows that do not
exist. A 3-row page produced a run of length 4: it satisfied a `min_line_h` of 4
that no 3-row page can, and that inflated height was what `_extract_line` took its
padding from. `monocr-onnx` bounds the same loop with `range(h_img)`; this package
scanned the profile instead.

Reachable without a synthetic fixture, because `smooth_kernel` is a constructor
argument — at the mon_OCR reference's 15 it caught any crop under 15 rows tall.
Default construction on a page taller than 5 rows was never affected.

### Lineage, stated where it is read

`LineSegmenter`'s docstring claimed step with `monocr_onnx` and said nothing about
the mon_OCR reference. The first half checks out: every constant is equal, adaptive
block 25 and C 10, 0.02, 10, 5, and pads of 0.20 and 0.15. The silence was the
costly half. This segmenter thresholds at 0.02 of the profile **max** where the
reference takes 0.12 of the **mean of its non-zero rows** — a different algorithm at
any number, not a different tuning — detects runs on the smoothed profile where the
reference detects on the raw one, and has no pre-blur, no smear, no gap merge, no
outlier rejection and no tiling.

The docstring now says all of that, plus the two ways it differs from
`monocr_onnx` itself (PIL-only input, and `smooth_kernel` where the port says
`smooth_window`), and the precondition it relies on: `MonOCR._prepare_image`, not
this class, is what guarantees the dark-on-light input its `THRESH_BINARY_INV`
needs.

No constant was changed. Which set is right is a measurement question and neither
repository has anything that scores a segmenter.

### Not changed

Nothing in the model contract: 160x1024 input, `pixel / 127.5 - 1.0`, the 276-character
charset and the pinned revision are all as they were.

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

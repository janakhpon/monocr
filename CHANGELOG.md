# Changelog

## Unreleased

Five correctness fixes to the inference path. None is released yet, and the
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
no Mon, Burmese or Latin glyph holds an unbroken stroke half a page long -- the
interior case, since OpenCV counts the out-of-image overhang as ink and the bar is
roughly half that at a page edge, which is what makes a border findable; and
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
argument — at the mon_OCR reference's 15 any crop under 15 rows tall is in
range, and drawn fixtures at that window differ from 3 rows to 10.
Default construction on a page of 5 rows or taller was never affected.

**Line boundaries now come off the raw profile.** The gap threshold is still
calibrated on the smoothed profile's max — that statistic and that basis are
unchanged — but the run boundaries are read off the raw row sums. Smoothing
averages over `smooth_kernel` rows, so a gap narrower than the whole window never
reaches zero in the smoothed profile: the ink either side bleeds into it, the bled
rows clear the 2% threshold, and two distinct lines come back as one band.

Measured at this package's own parameters — `min_line_h` 10, `smooth_kernel` 5,
`threshold_ratio` 0.02 — on 29 drawn glyph-blob bands of 12 rows each:

| gap between bands | detecting on the smoothed profile | detecting on the raw profile |
|---|---:|---:|
| 1, 2, 3, 4 px | 1 band of 29 drawn | 29 of 29 |
| 5 px and up | 29 of 29 | 29 of 29 |

The break point is the smoother's full width — exactly `smooth_kernel`, at every
window from 1 to 16, **even ones included**. numpy's `mode='same'` is a true k-tap
box at both parities, so a zero-run of length g drives an output row to zero iff
`g >= k`; at `g == k - 1` the smoothed minimum is `band_sum / k`, far above a 2%
threshold, which is why the count jumps 1 to 29 with no intermediate value.

That is this package's law and not the siblings'. The JS, Go and Rust bindings
hand-roll `[i - k//2, i + k//2]`, which is `2 * (w // 2) + 1` taps, so their break
points run 1,3,3,5,5,7,… The two laws agree on odd windows and part company on every
even one, where this package breaks a row **earlier**: window 4 fused gaps of 1-3 px
here against 1-4 there, window 6 fused 1-5 against 1-6. The absence of an
even-window case is why this went unnoticed. `smooth_kernel` is a constructor
argument, so a caller who raised it widened the failure with it.

The calibration stays on the smoothed profile while detection moves to the raw one.
**That split is a trade, not a correctness result**, and it was written up here as a
correctness result on the strength of a single fixture. Both directions are measured,
on the same family of page. Take a 1000-px page carrying a dense 1-px spike row: the
raw max is 127,500 and the smoothed max 25,500, giving thresholds of 2,550 and 510.

- Add a faint 20-row band summing 2,295 a row, which sits between them. Smoothed
  calibration finds it; raw calibration returns **zero** lines and loses the only
  real line on the page.
- Instead add two 12-row bands at rows 40-51 and 70-81 with a faint 4-column bridge
  across the gap summing 1,020 a row. Smoothed calibration returns **one** band,
  rows 32-89, the two lines fused; raw calibration returns **two**, correctly split.

A lower threshold keeps faint lines and fuses across faint bridges; a higher one
splits correctly and drops faint lines. Which is better is a measurement question,
and it is the same one this file already leaves open below: nothing in any of these
repositories scores a segmenter against labelled pages. The split is kept because
every sibling implementation does it that way, so changing it here alone would break
parity for no measured gain — not because the trade has been settled.

The **detection** half is not a trade: reading boundaries off the smoothed profile
fuses lines a kernel apart with no compensating benefit, which is why all six
implementations moved.

On an ordinary page the two calibrations are indistinguishable, which needs saying
because it means a test at the default ratio proves nothing about them. Smoothing
does not lower the peak of a band at least as tall as the window — height 5 at
kernel 5 already preserves it — so `max(smoothed) == max(raw)` in exact float
equality on every drawn page tried: 29 bands, gaps 1-20, band heights 5 and up,
64,260.0 either way. Band heights 1-4 at kernel 5 gave smoothed maxima of 12,852,
25,704, 38,556 and 51,408, which is `min(band_h, kernel) * peak / kernel`, so the
separations run the other way — 51,408 down to 12,852. Not "exactly `peak / kernel`":
that figure is 12,852, the separation at height 4 and the smoothed max at height 1
and neither elsewhere. That is why the calibration test needs a sub-kernel-height
spike row.

The `hist[:h_img]` bound on the scan is gone rather than left looking load-bearing.
`np.sum(binary, axis=1)` has exactly `h_img` elements, so detecting on it makes the
phantom rows above structurally impossible instead of guarded.

The two short-page tests defend against a regression to the **unbounded** smoothed
profile only. Measured 2026-08-28: under the bounded variant `smoothed_hist[:h_img]`,
which is the pre-change code, both of them pass — the slice removes the phantom rows
and leaves only the boundary bleed, which their fixtures cannot see. Four other tests
catch that variant, and the scope note on
`test_a_page_cannot_hold_a_line_taller_than_itself` names them.

This closes the divergence `monocr-onnx` opened at its commit a3e3dba. Read on
2026-08-28, this package was the last of six implementations still detecting on the
smoothed profile: `monocr-onnx`'s Python, JS, Go and Rust bindings and the mon_OCR
reference all read boundaries off the raw one. Sibling trees are other agents' work
in progress, so treat that as a dated observation.

**Runs split by a single sub-threshold row are now merged back.** Raw-profile
detection, the fix directly above, is unsafe on its own and shipping it alone was a
regression. Mon stacks diacritics above the base line, and at print resolution one
row between the diacritic zone and the consonant bodies dips below the gap
threshold without reaching zero. Raw detection cuts there: a strip of glyph tops
that decodes to Mon digits, and a decapitated body that decodes without its asats.

**What is ported from where.** The reference's own merge has two clauses — gap at
most `_MIN_GAP_MERGE` (10) rows and the raw minimum in the gap above zero — with no
page median and no ceiling, and it argues against crossing an empty gap at all ("if
in doubt, we keep lines SEPARATE"). What landed here is `monocr-onnx`'s three-clause
superset from `9135cab`; from the reference come the constant 10 and the ordering
(its merge is step 8, its height filter step 9). The fragment clause and the ceiling
are the Rust binding's.

`merge_runs` fuses a run into the previous one when the gap is at most
`MIN_GAP_MERGE` (10) rows, **and** either every row in the gap carries ink or one of
the two runs is at most half a typical line, **and** the result is at most twice a
typical line. `typical` is the page's own median run height. The merge runs
**before** the `min_line_h` filter, because a diacritic strip can be shorter than
the minimum and filtering first discards the strip and leaves the decapitated body
behind as a whole line.

Measured at this package's own parameters — `min_line_h` 10, `smooth_kernel` 5,
`threshold_ratio` 0.02 — over 55 real pages, 49 PDF pages rendered at 300 DPI plus
6 photographed pages, decoded with the pinned model revision `d3d9d5e`. Same
pipeline, same corpus, one function swapped:

| | runs | bands | garbage bands | clean characters |
|---|---:|---:|---:|---:|
| raw detection, no merge | 1,779 | 1,245 | 65 (5.2%) | 32,290 |
| raw detection + this merge | 1,570 | **1,266** | **58 (4.6%)** | **32,559** |

Band count *rises* while run count falls, because a merged strip-plus-body clears
`min_line_h` where the strip alone did not. 34 pages gained characters and **none
lost any**, so the garbage figure is not bought by discarding text. Garbage is a
band over half Mon digits and longer than 3 characters — the definition `mon_OCR`
`docs/AUDIT-2026-08-B.md` gives in **F-70**, not F-69. The length clause keeps
correctly-read page numbers out of the count.

**These numbers are smaller than the sibling figures** usually quoted beside this
change — 1.2% garbage for shipping nothing, 26.6% for raw detection alone, 0.7% for
the complete pair. That pair belongs to `se-brain rules/standards/testing.md` §24,
which draws it from F-69 **and** F-70 together over 24 scanned book pages plus three
photographs, measured through a sibling CLI. **It is not in F-69**, which is status
"reported, not fixed" and carries no after-merge measurement at all; an earlier draft
of this entry and of the source comment cited it there, which was wrong.

The corpus here is mostly digitally typeset PDF, whose inter-line gaps are clean and
wide; the 145-page image scan F-69 measured is not in this workspace. The mechanism
and the direction are the same on both metrics. Do not carry the sibling figures into
this package's source.

The concrete case, on page 1 of `party_mission.pdf` at 300 DPI: the smoothed max was
84,100, so the threshold was `0.02 * 84,100 = 1,682`, or 6.6 ink pixels a row. Rows
303-308 each carried exactly **5** ink pixels — under the threshold, over zero — and
split one line into an 8-row strip and a 32-row body. The 8-row strip is under
`min_line_h`, so before this change it was dropped and only the decapitated body
reached the model.

Three design points, each of which cost a rebuild upstream and each of which has a
test and a mutation here:

- **The height test is against the page median, not the neighbouring run.** Against
  the neighbour it cascades: the merge grows the accumulated run, a taller run makes
  the next line look more like a fragment, and upstream one page went from 36 bands
  to 10 with single bands of 534, 632 and 732 rows and lost 92% of its readable
  characters.
- **The ceiling of twice a typical line is load-bearing, not decoration.** Over the
  55 pages the gap bound and the ink-or-fragment clause together admitted 691
  candidate merges and the ceiling refused 482 of them.
- **A vertical smear is not a substitute.** At reach 1 it closes 2-row gaps, the
  same as the tightest real line spacing on these pages, so it fuses lines that are
  genuinely separate.

This closes the incomplete port `monocr-onnx` fixed in Rust at `9135cab`. Two of
its test fixtures do not transfer and were not copied, both for the same reason —
another clause absorbs the case the fixture claims to isolate, so the corresponding
mutation survives. Measured here 2026-08-28:

- `a_dip_between_equal_halves_merges_on_ink_alone` uses 40-row halves against 82-row
  lines, which puts the page median at 82 and makes `2 * 40 <= 82` true, so the
  fragment clause fires as well and dropping the ink clause **survives**. The
  fixture here uses 60-row lines instead, giving a median of 60.
- `a_wide_gap_is_a_line_boundary_however_much_ink_it_holds` uses two 40-row runs 15
  rows apart, whose merged band would be 95 rows against a ceiling of 80 — so the
  ceiling refuses it and dropping the gap bound **survives**. The fixture here uses
  a 24-row pair on a page whose median line is 40, giving a merged band of 64 rows,
  inside the ceiling.

Both were written that way here first and both mutations survived the first harness
run, which is how they were found.

### Recorded, not fixed

**The merge's `typical` is a median over runs that `min_line_h` will discard.** It is
computed before the height filter — which is the whole point of the ordering — so
speckles count toward it. Measured over the 55 pages above: 534 of 1,779 runs (30%)
are under `min_line_h`, and on 8 of the 55 that drives `typical` below 10.
`mon_e_lib.pdf` page 41 reaches `typical` 2, so a ceiling of 4, against a median of
35 over its runs that survive the filter — the merge is effectively off on exactly
the most fragmented pages.

Taking the median over runs already at or above `min_line_h`, falling back to all
runs when none qualify, fixes it and changes none of the ten new fixtures. Left alone
because `monocr-onnx` and the reference both compute over the unfiltered list, so
changing it here alone is a parity divergence and an owner decision — the same
argument as the `_extract_line` pad below. Found by an independent review of this
change, not by a test.

`_extract_line` mixes an inclusive last-ink-column index with PIL's exclusive
`crop`, so every crop it returns is one pixel short on the right: the left pad is
`pad_x` columns and the right pad `pad_x - 1`, and the reported width understates
the ink by one. Ink is only *lost* at `pad_x == 0`, which needs a caller passing
`min_line_h` below 7 — `pad_x = int(h_raw * 0.15)` and `h_raw >= min_line_h`, so the
default 10 forces `pad_x >= 1` and the last ink column is always inside. The
defaults cannot reach the data-loss path.

Unfixed on purpose. The same arithmetic is in `monocr_onnx` and in the mon_OCR
reference, whose `pad_x` floor of an absolute 10 px means no setting there can lose
ink. Correcting it in one place shifts every crop by a pixel and breaks parity with
two published packages and the corpus every page-level CER in this ecosystem was
measured against — an owner decision, not a cleanup. The reachability, the
asymmetry and the sibling and reference line numbers are recorded in the
`_extract_line` docstring, and the one test that pins a crop geometry says its
value is measured rather than correct.

### Lineage, stated where it is read

`LineSegmenter`'s docstring claimed step with `monocr_onnx` and said nothing about
the mon_OCR reference. The first half checks out: every constant is equal, adaptive
block 25 and C 10, 0.02, 10, 5, and pads of 0.20 and 0.15. The silence was the
costly half. This segmenter thresholds at 0.02 of the profile **max** where the
reference takes 0.12 of the **mean of its non-zero rows** — a different algorithm at
any number, not a different tuning — and has no pre-blur, no smear, no outlier
rejection and no tiling (the reference has all four). It also detected runs on the smoothed profile where the
reference detects on the raw one, and had no gap merge; both of those are now
closed, see above.

The docstring now says all of that, plus the four ways it differs from
`monocr_onnx` itself and the precondition it relies on: `MonOCR._prepare_image`,
not this class, is what guarantees the dark-on-light input its
`THRESH_BINARY_INV` needs.

The four are led by the one that breaks a call site hardest: `segment` here
returns `(crop, bbox)` tuples where the port returns `{'img', 'bbox'}` dicts, and
only one direction fails loudly — `crop, bbox = line` against a dict unpacks its
two keys and feeds the model the string `'img'`. Then `smooth_kernel` against
`smooth_window`, PIL-only input, and no `tile_line`/`cut_column`.

Four more divergences from the mon_OCR reference are now recorded, all
previously unlisted, and the first is the same class as the max-versus-mean
one. `pad_x` here is a
fraction of the line HEIGHT; the reference takes a fraction of the line WIDTH
floored at an absolute 10px, so a short tall word and a long thin line are padded
the opposite way round. Beside it: 0.40 against 0.20 vertically, rounding up
against truncating, and column extents off a dilated mask against the plain
binary one.

And the parity guard no longer defends only numbers. It now names the four things
that must agree — which profile the boundaries come off, which statistic the
threshold is calibrated on, which quantity each constant is a fraction of, and
only then the value — because a formula change moves the cuts with every number
left equal and a constants diff will not see it. The first of the four was live
when this was written — observed 2026-08-28, `monocr-onnx`'s Python binding
detecting boundaries on the raw profile while this package detected on the
smoothed one, every constant still equal — and is now closed. Items 2 to 4 are
open, and item 2 is the one to leave alone: the max-versus-mean split is a tuning
constant the reference's spec header forbids reconciling.

The reason `np.max` stays unsliced and on the smoothed profile is now recorded at
the line, because slicing it or moving it to the raw profile — either obvious
tidy-up — would break calibration parity with `monocr-onnx`, which also takes its
max unsliced off the smoothed profile.

No constant was changed. Which set is right is still a measurement question:
individual changes on both sides carry their own measurements, but nothing in
either repository scores a segmenter against labelled pages.

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

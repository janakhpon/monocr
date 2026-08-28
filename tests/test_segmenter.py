"""Line segmentation, and which path an image takes through it.

Fixtures here draw stroke patterns rather than solid blocks. Adaptive
thresholding responds to local contrast, so the middle of a filled rectangle
reads as background and only its edges light up — a solid bar is not a fixture
for text, it is a fixture for two thin lines.

Expected values below were measured against this segmenter on 2026-08-16, not
derived on paper.
"""

import numpy as np
from PIL import Image, ImageDraw

import pytest

from monocr.segmenter import LineSegmenter, smooth_profile


def strokes(draw, y0, y1, x0, x1, pitch=8, width=3):
    """Vertical strokes, standing in for glyphs on one text line."""
    for x in range(x0, x1, pitch):
        draw.rectangle([x, y0, x + width, y1], fill=0)


def page(bands, size=(600, 300), x0=50, x1=550):
    img = Image.new("L", size, 255)
    draw = ImageDraw.Draw(img)
    for top, bottom in bands:
        strokes(draw, top, bottom, x0, x1)
    return img


def logits_for(indices, num_classes):
    """A logits tensor whose greedy decode is exactly `indices`."""
    array = np.zeros((1, len(indices), num_classes), dtype=np.float32)
    for t, idx in enumerate(indices):
        array[0, t, idx] = 1.0
    return array


# ---------------------------------------------------------------------------
# LineSegmenter
# ---------------------------------------------------------------------------


def test_a_blank_page_yields_no_lines():
    """A page with no ink has a projection max of 0, so every row clears a
    threshold of 0 and the whole sheet used to come back as one blank "line"."""
    assert LineSegmenter().segment(Image.new("L", (600, 300), 255)) == []


def test_one_band_is_one_line():
    assert len(LineSegmenter().segment(page([(120, 150)]))) == 1


def test_three_bands_are_three_lines_top_to_bottom():
    lines = LineSegmenter().segment(page([(40, 70), (130, 160), (220, 250)]))
    assert len(lines) == 3
    tops = [bbox[1] for _, bbox in lines]
    assert tops == sorted(tops)


def test_the_crop_is_trimmed_to_the_ink_not_the_page_width():
    """The fix that folding `_extract_line` into the live path buys.

    A full-width crop hands the model a strip with the page's aspect ratio, and
    the resize then squeezes the text to fit 1024 columns. An indented line on a
    600px page must come back near its own left edge, not at 0.
    """
    lines = LineSegmenter().segment(page([(120, 150)], x0=300, x1=550))
    assert len(lines) == 1
    crop, (x, _y, w, _h) = lines[0]
    assert 280 <= x <= 300, f"left edge {x} is not near the ink at x=300"
    assert w < 300, f"crop is {w}px wide on a 600px page — it was not trimmed"
    assert crop.size == (w, _h)


def test_a_speckle_shorter_than_the_minimum_line_height_is_ignored():
    """The mark has to clear the ink threshold, or min_line_h is not what
    excludes it and the test guards nothing.

    A single dot is filtered out two steps earlier, by `hist > 2% of max`. This
    band spans the full text width, so its row sums match a real line's; only
    its height — 4 pixels against a minimum of 10 — keeps it out.
    """
    img = page([(120, 150)])
    strokes(ImageDraw.Draw(img), 20, 24, 50, 550)

    assert len(LineSegmenter().segment(img)) == 1
    assert len(LineSegmenter(min_line_h=1).segment(img)) == 2, (
        "with the minimum removed the speckle must become a second line, or "
        "this fixture is being filtered by something else"
    )


def test_a_page_cannot_hold_a_line_taller_than_itself():
    """The row scan must be bounded by real rows, not by the smoothed profile.

    `np.convolve(..., mode="same")` never returns fewer elements than its
    kernel, so on a page shorter than `smooth_kernel` the profile describes rows
    the page does not have. Unbounded, this 3-row page produced a run of length
    4: it passed a `min_line_h` of 4 that no 3-row page can satisfy, and handed
    `_extract_line` an inflated height to take its padding from.

    Since 2026-08-28 the scan reads the RAW profile, which is exactly `h_img`
    long, so the over-long run is structurally impossible rather than clipped by
    a slice. This test therefore now guards against a regression to the smoothed
    profile, which is the only way the phantom rows come back.

    Not a synthetic-only case. `smooth_kernel` is a constructor argument, so at
    the mon_OCR reference's 15 any crop under 15 rows tall is in range.
    """
    strip = np.full((3, 40), 255, dtype=np.uint8)
    strip[0, :] = 0
    img = Image.fromarray(strip)

    assert len(LineSegmenter(min_line_h=1).segment(img)) == 1, (
        "the fixture has to be segmentable at all, or the next assert is vacuous"
    )
    assert LineSegmenter(min_line_h=4).segment(img) == [], (
        "a 3-row page returned a line against a 4-row minimum, so the scan ran "
        "past the bottom of the page"
    )


def test_no_short_page_reports_a_line_it_is_too_short_to_hold():
    """The same bound, swept over the pairs that can actually catch it.

    Every case asks a page of `h` rows for a minimum of `h + 1`, which no page
    of that height can satisfy whatever the smoothing window is. The sweep used
    to run all 28 pairs of h 1-7 against kernel 3/5/9/15 and report that as its
    coverage. Measured against the pre-fix scan on 2026-08-28, only 10 of those
    28 returned a line at all: the other 18 passed without the fix, which makes
    them decoration, and the whole h=1 row asserted `[] == []` because a 1-row
    image is uniform and adaptive thresholding finds no ink in it.

    Two structural reasons a pair cannot discriminate, and the predicate below
    is both of them:

    * The phantom run has to *terminate* inside the smoothed profile. The
      trailing branch of the scan was always bounded by `h_img`, so a run that
      reaches the end of the profile was never over-long to begin with.
      For a window wider than the page — the only case with phantom rows in it
      — `np.convolve(..., mode="same")` starts its returned window at full index
      `(h - 1) // 2`, so the run from this fixture's single inked row ends at
      index `kernel - (h - 1) // 2` — inside the profile only from h=3 up.
    * That run then has to beat the minimum, i.e. exceed `h`.

    Together: `h + 1 <= kernel - (h - 1) // 2 <= kernel - 1`. That predicate
    reproduced pre-fix behaviour exactly over h 1-8 against kernel 3/5/9/15/21,
    0 mismatches, so nothing excluded here would have caught the bug. Each
    surviving case is guarded for non-vacuity as well, the same way
    `test_a_page_cannot_hold_a_line_taller_than_itself` is.

    The predicate is kept as-is now that the scan reads the raw profile, because
    it still selects exactly the pairs that discriminate against a regression to
    the smoothed profile — bounded or not. On the raw profile every one of these
    fixtures yields a run of length 1, so they pass structurally.
    """
    cases = [
        (h, kernel)
        for h in range(1, 8)
        for kernel in (3, 5, 9, 15)
        if h + 1 <= kernel - (h - 1) // 2 <= kernel - 1
    ]
    assert len(cases) == 10, (
        f"the sweep selected {len(cases)} pairs, not the 10 measured to "
        "discriminate — the predicate and the grid have drifted apart"
    )

    for h, kernel in cases:
        strip = np.full((h, 40), 255, dtype=np.uint8)
        strip[0, :] = 0
        img = Image.fromarray(strip)

        assert LineSegmenter(min_line_h=1, smooth_kernel=kernel).segment(img), (
            f"a {h}-row page with smooth_kernel={kernel} yields no line even at "
            "min_line_h=1, so the next assert is vacuous"
        )
        lines = LineSegmenter(min_line_h=h + 1, smooth_kernel=kernel).segment(img)
        assert lines == [], (
            f"a {h}-row page with smooth_kernel={kernel} returned "
            f"{len(lines)} lines against a minimum of {h + 1}"
        )


def test_a_short_page_pads_its_crop_from_the_line_not_the_smoothing_window():
    """The other half of the same bug, and the half no line count can see.

    `_extract_line` pads by a fraction of `r_end - r_start`. When the scan read
    the smoothed profile, `r_end` came off it too, so on a page shorter than the
    smoothing window the pad was a fraction of the profile length instead of the
    line's own height. The line still existed and still counted as one line —
    only its shape was wrong, which is why the count sweeps above cannot catch
    this and the CHANGELOG claim about it went untested until 2026-08-28.

    Fixture: 3 rows, ink in row 0 across columns 8-20, `smooth_kernel=15`.
    Three measured values, all on 2026-08-28, all on this same fixture:

    * Unbounded scan on the smoothed profile — the original bug. The run
      terminated at profile index 14, so `pad_x` was `int(14 * 0.15) == 2` and
      the crop came back (6, 0, 16, 3).
    * Bounded scan on the smoothed profile — the 2026-08-28 morning fix. The run
      was clipped at `h_img`, giving (8, 0, 12, 3).
    * Scan on the RAW profile — current. The ink is in row 0 only, so the run is
      one row long and the crop is (8, 0, 12, 1). Every number here now comes
      off the line itself.

    (8, 0, 12, 1) is the MEASURED value, not an endorsement of the width. The
    ink spans 13 columns, 8 to 20 inclusive, and `_extract_line` mixes an
    inclusive `x_end` with PIL's exclusive crop, so the last ink column is
    dropped at zero pad. That off-by-one is shared with `monocr_onnx` and with
    the mon_OCR reference and is deliberately unfixed — see the
    `_extract_line` docstring. This pin records what the code does; it does not
    bless the 12.
    """
    strip = np.full((3, 40), 255, dtype=np.uint8)
    strip[0, 8:21] = 0
    img = Image.fromarray(strip)

    lines = LineSegmenter(min_line_h=1, smooth_kernel=15).segment(img)
    assert len(lines) == 1, (
        "the fixture has to segment at all, or the next assert is vacuous"
    )
    crop, bbox = lines[0]
    assert bbox == (8, 0, 12, 1), (
        f"crop is {bbox}; the line is one inked row, so it pads by "
        "int(1 * 0.15) == 0 in x and int(1 * 0.20) == 0 in y — any margin or "
        "extra height here was taken from a profile, not from the line"
    )
    assert crop.size == (12, 1)


def test_a_single_line_crop_at_the_model_height_is_not_shredded():
    """160 pixels tall is one line to this model, and it must stay one line.

    Until 2.3.0 `predict` sent anything over 100 pixels down the page path, and
    the model's own input is 160 — so an ordinary line crop was treated as a
    page. That dispatch is gone; this pins the segmenter half of the claim.
    """
    tall = Image.new("L", (900, 160), 255)
    strokes(ImageDraw.Draw(tall), 40, 120, 20, 880)
    assert len(LineSegmenter().segment(tall)) == 1


# ---------------------------------------------------------------------------
# Which profile the boundaries come off, and which the threshold does
# ---------------------------------------------------------------------------


def two_bands(gap, band_h=12, width=600, x0=50, x1=550, top=20):
    """Two glyph-blob bands separated by exactly `gap` blank rows.

    Blobs, not a solid bar: a solid bar spanning a text column is a printed rule
    by this module's own definition and `suppress_page_rules` deletes it before
    the profile sees it.
    """
    img = Image.new("L", (width, top * 2 + 2 * band_h + gap), 255)
    draw = ImageDraw.Draw(img)
    strokes(draw, top, top + band_h - 1, x0, x1)
    strokes(draw, top + band_h + gap, top + 2 * band_h + gap - 1, x0, x1)
    return img


def test_line_boundaries_come_off_the_raw_profile():
    """A gap narrower than the smoothing window must still split two lines.

    The threshold is calibrated on the smoothed profile, but the boundaries are
    read off the raw one. Smoothing averages over `smooth_kernel` rows, so a gap
    narrower than the whole window never reaches zero in the smoothed profile:
    the ink either side bleeds into it, the bled rows clear the 2% threshold, and
    two lines come back as one band. The raw profile needs one clean row.

    Measured 2026-08-28 at this module's parameters (min_line_h 10,
    smooth_kernel 5, threshold_ratio 0.02) on 29 drawn bands of 12 rows:
    detecting on the smoothed profile returned 1 band against 29 drawn at gaps
    of 1, 2, 3 and 4 px and matched the raw profile from 5 px up; the raw profile
    returned all 29 from 1 px. This is the two-band corner of that table — gaps
    1-4 fuse pre-fix, gap 5 does not, so the pair below is what discriminates.
    """
    for gap in (1, 2, 3, 4):
        assert len(LineSegmenter().segment(two_bands(gap))) == 2, (
            f"a {gap}px gap fused two lines, so the boundaries are being read "
            "off the smoothed profile"
        )
    assert len(LineSegmenter().segment(two_bands(5))) == 2, (
        "a 5px gap must split on either profile — if this fails the fixture "
        "itself is broken, not the profile choice"
    )


def test_an_even_smoothing_window_does_not_widen_to_the_odd_one_above_it():
    """The even-kernel case, which no sibling binding has.

    numpy's `mode='same'` is a true even-width box: `smooth_profile(raw, 4)` has
    exactly 4 taps and divides by 4. The hand-rolled loops in the JS, Go and
    Rust bindings span `2 * (w // 2) + 1`, so an even window there behaves as the
    odd window above it. Measured here 2026-08-28: the smoothed profile's fusion
    break point is exactly `smooth_kernel` at every kernel from 1 to 16, and a
    second from-scratch measurement the same day agreed — kernel 4 fused gaps
    1-3 and split at 4, kernel 6 fused 1-5 and split at 6, where the hand-rolled
    `2 * (k // 2) + 1` law would have given 1-4 and 1-6. The two laws agree on
    odd kernels and part company on every even one.

    Post-fix the raw profile splits at every gap whatever the kernel, which is
    what this asserts. The parity law itself is only observable on the smoothed
    profile now, and `test_smooth_profile_is_a_true_window_tap_box` pins it
    there.
    """
    for kernel, gap in ((2, 1), (4, 2), (4, 3), (6, 5), (12, 11)):
        lines = LineSegmenter(smooth_kernel=kernel).segment(two_bands(gap))
        assert len(lines) == 2, (
            f"smooth_kernel={kernel} fused a {gap}px gap into "
            f"{len(lines)} line(s)"
        )


def test_the_gap_threshold_is_calibrated_on_the_smoothed_profile():
    """The other half: the threshold basis did NOT move to the raw profile.

    The smoothed max is the LOWER of the two whenever the page's tallest peak is
    narrower than the smoothing window, so calibrating on it is what keeps faint
    lines visible. This fixture makes that observable. A dense 1-px row on a
    1000-px page gives a raw max of 127,500 and a smoothed max of exactly a
    fifth of it, 25,500 — the smoother flattens a peak it can average over.
    Beside it sits a faint 20-row band whose every row sums to 2,295, which
    lands between the two thresholds: 2% of 25,500 is 510, 2% of 127,500 is
    2,550.

    Measured 2026-08-28, twice and independently: calibrating on the smoothed
    profile returns this one band, calibrating on the raw profile returns ZERO
    lines and loses the only real line on the page. So moving the calibration
    along with the detection would trade a fused-line bug for a dropped-line
    one.

    An ordinary page cannot tell the two calibrations apart, which is why the
    fixture is this strange. Smoothing does not lower the peak of a band taller
    than the window, so `max(smoothed) == max(raw)` in exact float equality on
    every drawn page tried — 29 bands, gaps 1-20, band heights 5 and up,
    64,260.0 either way.

    The spike row itself is never a line, being 1 row against a minimum of 10.
    """
    strip = np.full((120, 1000), 255, dtype=np.uint8)
    strip[10, ::2] = 0
    for x in range(100, 136, 4):
        strip[40:60, x] = 0

    lines = LineSegmenter().segment(Image.fromarray(strip))
    assert len(lines) == 1, (
        f"expected the faint band and nothing else, got {len(lines)} lines "
        f"{[b for _, b in lines]} — calibrating on the raw profile loses it, "
        "and counting the 1px spike as a line means min_line_h stopped working"
    )
    _crop, (x, y, w, h) = lines[0]
    assert (x, y, w, h) == (97, 36, 38, 28), (
        f"the one line is {(x, y, w, h)}, not the faint band at rows 40-59, "
        "columns 100-132 padded by int(20 * 0.15) == 3 and int(20 * 0.20) == 4"
    )


@pytest.mark.parametrize("window", range(1, 17))
def test_smooth_profile_is_a_true_window_tap_box(window):
    """Span and divisor, for even windows as well as odd.

    Three properties, each measured over windows 1-16 on 2026-08-28:

    * The divisor is exactly `window`, so an isolated spike's peak is
      `spike / window` — 1/2, 1/3, 1/4, ... with no parity step.
    * A run of exactly `window` zero rows drives at least one output row to
      zero, and a run of `window - 1` never does. That is the fusion break
      point `segment` step 5 measures, and it is where the sibling bindings
      part company: their `2 * (w // 2) + 1` loops span one row more than asked
      at even windows, so a run of 4 zeros does NOT clear their window 4.
    * `mode='same'` never returns fewer elements than the kernel, so a window
      wider than the page yields a profile longer than the page. Only the
      threshold is read off it.
    """
    spike = np.zeros(200, dtype=np.float32)
    spike[100] = 1000.0
    out = smooth_profile(spike, window)

    if window <= 1:
        assert out is spike, "window <= 1 must hand back the raw profile itself"
    assert len(out) == len(spike)
    short = np.zeros(3, dtype=np.float32)
    short[0] = 300.0
    assert len(smooth_profile(short, window)) == max(3, window), (
        "mode='same' never returns fewer elements than the kernel, so a window "
        "wider than the page must yield a profile longer than the page — which "
        "is why only the threshold is read off it"
    )
    assert out.max() == pytest.approx(1000.0 / window), (
        f"peak is {out.max()}, so the divisor is not exactly {window}"
    )

    def has_a_zero_row(zeros):
        profile = np.zeros(200, dtype=np.float32)
        profile[20:60] = 500.0
        profile[60 + zeros:100] = 500.0
        return bool(np.any(smooth_profile(profile, window)[20:100] == 0.0))

    assert has_a_zero_row(window), (
        f"a run of {window} zero rows never reaches zero at window {window}, "
        "so the box spans more than its width"
    )
    if window > 1:
        assert not has_a_zero_row(window - 1), (
            f"a run of {window - 1} zero rows reached zero at window {window}, "
            "so the box spans less than its width"
        )


# ---------------------------------------------------------------------------
# Which path an image takes
# ---------------------------------------------------------------------------


def test_predict_page_reads_every_line(make_ocr):
    ocr = make_ocr()
    ocr.predict_page(page([(40, 70), (130, 160), (220, 250)]))
    assert ocr.fake_session.calls == 3


def test_predict_line_reads_the_image_once_however_tall_it_is(make_ocr):
    """The escape hatch. A caller holding a line crop can guarantee one read."""
    ocr = make_ocr()
    ocr.predict_line(page([(40, 70), (130, 160), (220, 250)]))
    assert ocr.fake_session.calls == 1


def test_predict_is_the_page_path(make_ocr):
    ocr = make_ocr()
    ocr.predict(page([(40, 70), (130, 160)]))
    assert ocr.fake_session.calls == 2


def test_an_unsegmentable_image_is_read_whole_rather_than_reported_empty(make_ocr):
    """Returning "" for a page the segmenter could not split reports success for
    a failure — the caller cannot tell it from a genuinely blank page."""
    ocr = make_ocr()
    assert ocr.predict(Image.new("L", (600, 300), 255)) == ""
    assert ocr.fake_session.calls == 1, "the whole image must still reach the model"


def test_lines_are_joined_with_newlines(make_ocr):
    ocr = make_ocr(charset=" ab", num_classes=4, logits=logits_for([2], 4))
    assert ocr.predict_page(page([(40, 70), (130, 160), (220, 250)])) == "a\na\na"


def test_blank_lines_are_dropped_from_the_join(make_ocr):
    """A line that decodes to nothing must not leave an empty row in the output."""
    ocr = make_ocr(charset=" ab", num_classes=4, logits=logits_for([0], 4))
    assert ocr.predict_page(page([(40, 70), (130, 160)])) == ""


def test_confidence_is_averaged_over_the_lines_that_decoded(make_ocr):
    ocr = make_ocr(charset=" ab", num_classes=4, logits=logits_for([2], 4))
    result = ocr.predict_with_confidence(page([(40, 70), (130, 160)]))
    assert result["text"] == "a\na"
    assert 0.0 < result["confidence"] <= 1.0


def test_confidence_is_zero_when_nothing_decoded(make_ocr):
    """Not a crash on an empty mean, and not a confident report of no text."""
    ocr = make_ocr()
    result = ocr.predict_with_confidence(Image.new("L", (600, 300), 255))
    assert result == {"text": "", "confidence": 0.0}

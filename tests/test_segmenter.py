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

from monocr.segmenter import LineSegmenter


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
    `_extract_line` an inflated height to take its padding from. The monocr_onnx
    port bounds the same loop with `range(h_img)`; this one did not.

    Not a synthetic-only case. `smooth_kernel` is a constructor argument, so at
    the mon_OCR reference's 15 the bug reaches any crop under 15 rows tall.
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
      `np.convolve(..., mode="same")` starts its returned window at full index
      `(h - 1) // 2`, so the run from this fixture's single inked row ends at
      index `kernel - (h - 1) // 2` — inside the profile only from h=3 up.
    * That run then has to beat the minimum, i.e. exceed `h`.

    Together: `h + 1 <= kernel - (h - 1) // 2 <= kernel - 1`. That predicate
    reproduced pre-fix behaviour exactly over h 1-8 against kernel 3/5/9/15/21,
    0 mismatches, so nothing excluded here would have caught the bug. Each
    surviving case is guarded for non-vacuity as well, the same way
    `test_a_page_cannot_hold_a_line_taller_than_itself` is.
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

    `_extract_line` pads by a fraction of `r_end - r_start`. With the scan
    unbounded, `r_end` came off the smoothed profile, so on a page shorter than
    the smoothing window the pad was a fraction of the profile length instead of
    the line's own height. The line still existed and still counted as one line
    — only its shape was wrong, which is why the count sweep above cannot catch
    this and the CHANGELOG claim about it went untested until 2026-08-28.

    Measured 2026-08-28 on this fixture: 3 rows, ink in row 0 across columns
    8-20, `smooth_kernel=15`. The run terminated at profile index 14, so `pad_x`
    was `int(14 * 0.15) == 2` rather than `int(3 * 0.15) == 0`, and the crop came
    back as (6, 0, 16, 3) around a line that is (8, 0, 12, 3) — four columns of
    padding the ink never asked for.
    """
    strip = np.full((3, 40), 255, dtype=np.uint8)
    strip[0, 8:21] = 0
    img = Image.fromarray(strip)

    lines = LineSegmenter(min_line_h=1, smooth_kernel=15).segment(img)
    assert len(lines) == 1, (
        "the fixture has to segment at all, or the next assert is vacuous"
    )
    crop, bbox = lines[0]
    assert bbox == (8, 0, 12, 3), (
        f"crop is {bbox}; a 3-row line pads by int(3 * 0.15) == 0 in x and "
        "int(3 * 0.20) == 0 in y, so any margin here was taken from a height "
        "the page does not have"
    )
    assert crop.size == (12, 3)


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

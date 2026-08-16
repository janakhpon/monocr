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

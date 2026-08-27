"""The model is trained on dark text on a light background; check what we feed it.

Measured 2026-08-27 over 300 labelled crops from mon_OCR's `data/real/digits/val`,
same graph, only the polarity of the input changed:

    upright, with the probe      CER 0.0000   300/300 exact
    inverted, with the probe     CER 0.0000   300/300 exact
    upright, without it          CER 0.0036   296/300
    inverted, without it         CER 0.0342   288/300   <- 9.5x worse

Degradation rather than total failure, closed for the cost of four corner patches.
Background levelling — step 4 of the reference `to_normalized_grayscale` — is what
the 0.0036 upright row costs and is deliberately not ported here.
"""

import numpy as np
import pytest
from PIL import Image

from monocr.ocr import normalize_polarity


def _page(bg: int, ink: int, w: int = 200, h: int = 60) -> Image.Image:
    a = np.full((h, w), bg, dtype=np.uint8)
    a[h // 3 : 2 * h // 3, w // 5 : 4 * w // 5] = ink
    return Image.fromarray(a, mode="L")


def test_a_dark_on_light_page_is_returned_unchanged():
    """THE NO-OP. Every input passes through this, so an ordinary page must come
    back byte-identical or the probe is a regression rather than a fix."""
    page = _page(bg=255, ink=0)
    assert np.array_equal(np.asarray(normalize_polarity(page)), np.asarray(page))


def test_a_light_on_dark_page_is_inverted():
    out = np.asarray(normalize_polarity(_page(bg=0, ink=255)))
    assert out[0, 0] == 255
    assert out[30, 100] == 0


def test_a_dense_page_is_not_mistaken_for_a_dark_one():
    """Why corner-median and not a global mean: this page is ~64% ink, so its mean
    luminance is below 128 and a global test would invert an ordinary dense page."""
    a = np.full((60, 200), 255, dtype=np.uint8)
    a[6:54, 20:180] = 0
    page = Image.fromarray(a, mode="L")
    assert np.asarray(page).mean() < 128, "fixture must actually be mean-dark"
    assert np.array_equal(np.asarray(normalize_polarity(page)), np.asarray(page))


@pytest.mark.parametrize(
    "shape", [(8, 120), (60, 8)], ids=["short-crop", "narrow-crop"]
)
def test_the_corner_floor_covers_both_axes(shape):
    """`_POLARITY_CORNER_FLOOR` guards height and width separately, and a test for
    one does not cover the other. Without a floor the patch is empty for anything
    under 10px, `np.median([])` is nan, `nan < 128` is False, and a dark crop is
    silently left inverted — a wrong answer rather than a crash."""
    h, w = shape
    a = np.full((h, w), 0, dtype=np.uint8)
    a[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = 255
    assert np.asarray(normalize_polarity(Image.fromarray(a, mode="L")))[0, 0] == 255


def test_the_probe_runs_on_the_real_entry_point():
    """The unit tests above are worthless if `_prepare_image` does not call it."""
    from monocr.ocr import MonOCR

    inverted = _page(bg=0, ink=255)
    out = np.asarray(MonOCR._prepare_image(object.__new__(MonOCR), inverted))
    assert out[0, 0] == 255, "_prepare_image must normalise polarity"

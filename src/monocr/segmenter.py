import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple

# Printed-rule suppression. A page border adds a constant ink floor to every row
# it spans, and once that floor clears the gap threshold no in-frame row reads as
# a gap: the page comes back as one band and is squeezed into the model window.
#
# Measured 2026-08-27 on the reference implementation over twelve real MNEC
# papers: nine collapsed to a single band without this, seven of them returning
# 0-2 characters, and the twelve together went from 3,846 characters to 5,924.
# Pages carrying no rules come back byte-identical.
#
# A rule is an unbroken ink run spanning at least RULE_SPAN of the page in one
# direction. Morphological opening with a line kernel keeps exactly those runs;
# subtracting them leaves the text. No Mon, Burmese or Latin glyph holds an
# unbroken stroke half a page long, so the false-positive risk against text is
# structural rather than merely small.
#
# There is deliberately NO thickness test. "A rule is long AND thin" was written,
# measured and deleted upstream: across twelve real pages the rule pixels found
# with a thickness limit and with none were identical to the pixel, and it cannot
# work anyway -- adaptiveThreshold compares against a LOCAL mean, so the interior
# of a thick ink region is not ink and only its edges are. Every thick region
# arrives already presented as a pair of thin bands.
RULE_SPAN = 0.5

# Suppression that would remove more than this share of the page ink has found
# text, not rules, and is abandoned. RULE_SPAN is a fraction of the page, so on a
# SHORT page a tall block of text exceeds it vertically and every glyph column
# reads as a rule. Upstream this was caught by an existing test losing 98.7% of
# its ink and returning zero lines. The threshold sits in a measured gap: real
# framed pages classify 21.5%-58.8% of their ink as rules, rule-free pages 0.00%,
# and that false positive 98.7%.
RULE_MAX_INK_SHARE = 0.80


def suppress_page_rules(binary):
    """Remove printed rules from a text mask, leaving glyphs untouched.

    Returns `binary` unchanged when the page carries no rules, and also when
    suppression would remove more ink than RULE_MAX_INK_SHARE.

    Two behaviours the constants above do not state, both measured 2026-08-28.
    The kernels are floored at 15 px, so below 30 px of page in a direction the
    span test is an absolute 15-px run rather than RULE_SPAN of the page. And
    OpenCV erodes with out-of-image pixels treated as ink, so a run touching a
    page edge is credited with the overhang and roughly half of RULE_SPAN is
    enough there: a 10-px run on a 20-px-wide page is a rule at the edge and is
    not one in the middle.

    That second one is not a leak, it is why a printed border is found at all --
    a border touches all four edges. It does mean the "half a page long"
    argument above is the interior case; at an edge the bar is half that. An
    all-ink page comes back unchanged through RULE_MAX_INK_SHARE, not because
    the kernel outgrew the page: opening does not clear a mask it overhangs.
    """
    h, w = binary.shape
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(w * RULE_SPAN)), 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(h * RULE_SPAN)))),
    )
    rules = cv2.bitwise_or(horizontal, vertical)
    ink = int(np.count_nonzero(binary))
    if not ink or np.count_nonzero(rules) > ink * RULE_MAX_INK_SHARE:
        return binary
    return cv2.subtract(binary, rules)


class LineSegmenter:
    """
    Robust line segmenter using Horizontal Projection Profiles with Smoothing.
    Handles noisy documents and touching lines by finding valleys in the projection.

    Lineage, checked against both siblings on 2026-08-28 rather than assumed.

    This is a port of ``monocr_onnx.segmenter.LineSegmenter`` and that claim
    holds: every constant is equal -- adaptive block 25 and C 10,
    threshold_ratio 0.02, min_line_h 10, smoothing 5, pads of 0.20 and 0.15 of
    the core line height -- and the printed-rule block above is the same code
    with the same RULE_SPAN and RULE_MAX_INK_SHARE. Changing a number here
    without changing it there is what makes two bindings disagree on one input.

    Two differences that are not cuts. This class takes PIL only, where the port
    also accepts a bare ndarray; and its smoothing argument is ``smooth_kernel``
    where the port says ``smooth_window``, so a keyword call does not carry
    across. The port also ships ``tile_line``/``cut_column`` for a line too wide
    for the model window. This module has neither, so an over-wide line is
    squeezed to fit instead of being split -- see ``ocr._predict_single_line``.

    It is NOT a port of the mon_OCR reference, and the distance is wider than
    tuning. The gap threshold here is a fraction of the profile MAX; the
    reference takes a fraction of the MEAN of its non-zero rows. Max and mean
    part company as lines are added to a page, so no choice of ratio reconciles
    them: 0.02 of the max and 0.12 of the mean are different algorithms at any
    number. The reference also detects run boundaries on the raw profile while
    calibrating the threshold on the smoothed one, and runs four stages absent
    here entirely -- a pre-blur, a morphological smear, a bounded gap merge, and
    outlier rejection. Expect different line counts on the same page.

    Which set is right is unmeasured: nothing in either repository scores a
    segmenter, so do not close the question by editing a number here.

    Polarity is a precondition, not a stage. This class treats dark pixels as
    ink and never probes; ``MonOCR._prepare_image`` runs ``normalize_polarity``
    before handing a page over. Called directly it must be given
    dark-text-on-light, or it segments the background. The reference probes
    inside ``segment`` itself and so carries no such precondition.
    """
    def __init__(self, min_line_h: int = 10, smooth_kernel: int = 5, threshold_ratio: float = 0.02):
        self.min_line_h = min_line_h
        self.smooth_kernel = smooth_kernel
        self.threshold_ratio = threshold_ratio

    def segment(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        Segment a document image into text lines.
        Returns list of (cropped_image, (x, y, w, h)) sorted top-to-bottom.
        """
        # 1. Convert to Grayscale & Numpy
        img_np = np.array(image.convert("L"))
        h_img = img_np.shape[0]

        # 2. Binarize (Adaptive Thresholding)
        # THRESH_BINARY_INV makes ink white and paper black. That picks out the
        # text only if the page arrives dark-on-light, which this class requires
        # of its caller rather than checking -- see the class docstring.
        binary = cv2.adaptiveThreshold(
            img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 10
        )

        # 2.5 Printed-rule suppression
        binary = suppress_page_rules(binary)

        # 3. Horizontal Projection Profile
        hist = np.sum(binary, axis=1).astype(np.float32)  # Shape (H,)

        # 4. Smooth Histogram
        if self.smooth_kernel > 1:
            kernel = np.ones(self.smooth_kernel) / self.smooth_kernel
            hist = np.convolve(hist, kernel, mode='same')

        # 5. Find Text Regions vs Spaces
        #
        # A blank page needs no special case: its maximum is 0, so the threshold
        # is 0 and the strict `>` excludes every row. An explicit early return
        # here was dead code — mutation testing on 2026-08-16 removed it and no
        # test changed.
        max_val = np.max(hist)
        threshold = max_val * self.threshold_ratio

        # Bound the scan to real rows. `np.convolve(..., mode="same")` returns
        # max(len(hist), smooth_kernel) elements, so on a page shorter than the
        # smoothing window `hist` describes rows that do not exist. The
        # monocr_onnx port bounds the same loop with `range(h_img)`; without it a
        # 3-row page produced a run of length 4, which both passed a min_line_h
        # of 4 that no 3-row page can satisfy and inflated the h_raw that
        # `_extract_line` derives its padding from. Reachable in normal use,
        # because `smooth_kernel` is a constructor argument: at the reference's
        # 15 it catches any crop under 15 rows tall.
        is_text_row = hist[:h_img] > threshold

        lines: List[Tuple[Image.Image, Tuple[int, int, int, int]]] = []
        start_y = None

        for y, is_text in enumerate(is_text_row):
            if is_text and start_y is None:
                start_y = y # Start of line
            elif not is_text and start_y is not None:
                # End of line (found a gap). Anything shorter than min_line_h is
                # a speckle, not a line.
                if (y - start_y) >= self.min_line_h:
                    self._extract_line(binary, img_np, start_y, y, image, lines)
                start_y = None

        # Handle last block
        if start_y is not None and (h_img - start_y) >= self.min_line_h:
            self._extract_line(binary, img_np, start_y, h_img, image, lines)

        return lines

    def _extract_line(self, binary, gray, r_start, r_end, source_image, lines_list):
        """Crop one detected line region and append it as (crop, bbox).

        Trims left and right whitespace and pads relative to the line's own
        height. A full-width crop with a fixed 4-pixel pad — what this class did
        until 2.3.0, with this method sitting unused beside it — hands the model
        a strip whose aspect ratio is the page's, not the line's, and the resize
        then squeezes the text horizontally to fit the window.
        """
        # Find horizontal boundaries (cropping left/right whitespace)
        line_slice = binary[r_start:r_end, :]
        col_sums = np.sum(line_slice, axis=0)
        col_indices = np.where(col_sums > 0)[0]

        if len(col_indices) == 0:
            return

        x_start, x_end = col_indices[0], col_indices[-1]

        # Relative padding based on line height
        h_raw = r_end - r_start
        pad_y = int(h_raw * 0.20)
        pad_x = int(h_raw * 0.15)

        y1 = max(0, r_start - pad_y)
        y2 = min(gray.shape[0], r_end + pad_y)
        x1 = max(0, x_start - pad_x)
        x2 = min(gray.shape[1], x_end + pad_x)

        crop = source_image.crop((x1, y1, x2, y2))
        lines_list.append((crop, (int(x1), int(y1), int(x2 - x1), int(y2 - y1))))


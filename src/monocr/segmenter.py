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


def smooth_profile(raw_hist, window):
    """Box-filter the row profile. Returns `raw_hist` ITSELF for window <= 1.

    Not a copy. The caller keeps both profiles alive and neither is written to,
    so the aliasing is harmless -- but do not start mutating either in place.
    `monocr_onnx`'s Python and JS bindings alias the same way; its Go binding
    returns a fresh slice on purpose. Do not read the Go note as a bug report
    about this one.

    A TRUE `window`-tap box. The kernel is `np.ones(window) / window`, so it has
    exactly `window` taps whatever the parity and carries the division itself --
    `mode='same'` only selects which slice of the full convolution comes back.
    A run of `window` zero rows therefore always drives at least one output row
    to zero, and a run of `window - 1` never does. Measured 2026-08-28 over windows 1-16: the zero-run
    break point is exactly `window` at every one of them, and the peak of an
    isolated spike is exactly `spike / window`.

    That is the same law as `monocr_onnx.segmenter.smooth_profile`, read on
    2026-08-28, and NOT the law of that repository's JS, Go and Rust bindings:
    those loop `[-window//2, +window//2]`, which is `2 * (window // 2) + 1`
    taps, so an even window there spans one row MORE than asked. Their break
    points run 1,3,3,5,5,7,... Do not carry a table across in either direction.

    "and behaves as the odd window above it" was written here and is only true
    of JS and Rust, which divide by the number of rows actually visited and so
    really are a `2 * (window // 2) + 1`-tap box. Go divides by the REQUESTED
    `window` (`go/pkg/segmenter/segmenter.go`, read 2026-08-28: it sums over
    `[-overflow, +overflow]` with `overflow := window / 2`, then
    `sum / float64(window)`). At an even window it therefore sums `window + 1`
    terms and divides by `window`, inflating every interior row by
    `(window + 1) / window` -- enough that its smoothed peak CLEARS the raw
    peak. Its break point still matches, because that is set by the span and the
    threshold scales with the inflation; its VALUES do not.

    `mode='same'` returns `max(len(raw_hist), window)` elements, so for a window
    wider than the page this profile is LONGER than the page. That is one reason
    only the threshold is read off it; see `segment`.
    """
    if window <= 1:
        return raw_hist
    kernel = np.ones(window) / window
    return np.convolve(raw_hist, kernel, mode="same")


# Two runs separated by at most this many rows are one text line, provided the
# raw profile never reaches zero inside the gap.
#
# WHY THIS EXISTS, measured 2026-08-28 at THIS module's parameters (min_line_h
# 10, smooth_kernel 5, threshold_ratio 0.02) over 55 real pages -- 49 PDF pages
# rendered at 300 DPI plus 6 photographed pages -- decoded with the pinned model
# revision d3d9d5e.
#
# Detecting boundaries on the raw profile (2026-08-28, one commit before this
# one) splits a single line wherever one row dips below the gap threshold, and in
# Mon that happens between the upper diacritic zone and the consonant bodies.
# Observed on page 1 of `party_mission.pdf` at 300 DPI: the threshold was
# `0.02 * 84,100 = 1,682`, which is 6.6 ink pixels a row, and the six rows 303-308
# each carried exactly 5 ink pixels -- under the threshold, over zero. That cut
# one line into an 8-row strip of glyph tops (rows 295-303) and a 32-row
# decapitated body (rows 309-341). The strip is shorter than `min_line_h`, so it
# was DISCARDED and only the body reached the model, missing its asats.
#
# What the merge buys, same 55 pages, same model, one function swapped:
#
#                       runs   bands   garbage bands   clean characters
#   raw, no merge       1,779  1,245    65  (5.2%)     32,290
#   raw + this merge    1,570  1,266    58  (4.6%)     32,559
#
# 209 run boundaries closed; band count RISES because a merged strip-plus-body
# clears `min_line_h` where the strip alone did not. Garbage is a band over half
# Mon digits and longer than 3 characters, which is the metric `mon_OCR`
# `docs/AUDIT-2026-08-B.md` defines in F-70, not F-69 -- the length clause keeps
# correctly-read page numbers out of the count. 34 pages gained characters and
# NONE lost any, so the garbage number is not bought by dropping text.
#
# These are this module's numbers and they are SMALLER than the sibling figures
# they are often quoted beside -- 1.2% garbage for shipping nothing, 26.6% for raw
# detection alone, 0.7% for the complete pair. Do not substitute those, and do not
# cite them to F-69: that pair is in `se-brain rules/standards/testing.md` §24,
# which draws it from F-69 AND F-70 together over 24 scanned book pages plus three
# photographs, measured through a sibling CLI. F-69 itself is status "reported, not
# fixed" and carries no after-merge measurement at all. Checked 2026-08-28.
#
# The corpus here is mostly digitally typeset PDF, whose inter-line gaps are clean
# and wide; the 145-page image scan F-69 measured is not in this workspace. The
# mechanism is the same and the direction is the same on both metrics.
#
# A 1-row gap holding ink is not a line boundary at any resolution. That much is
# the reference's rule, but be precise about WHAT is ported from where, because an
# earlier version of this comment credited all of it to the reference and that is
# wrong. Read 2026-08-28 in `mon_OCR/src/monocr/segmenter.py` step 8, the
# reference's merge has exactly TWO clauses -- gap <= `_MIN_GAP_MERGE` AND the raw
# minimum in the gap above zero -- with no fragment clause, no page median and no
# ceiling. It argues explicitly against crossing an empty gap: "If in doubt, we
# keep lines SEPARATE -- text is NEVER lost this way."
#
# So what this module runs is `monocr-onnx`'s three-clause SUPERSET, added in Rust
# at `9135cab`. From the reference come the constant 10 and the ORDER -- its merge
# is step 8 and its minimum-height filter step 9. The fragment clause and the
# ceiling are the Rust binding's, and the fragment clause is a deliberate
# departure from the reference's own advice, justified by the cascade measurement
# in `merge_runs` rather than by the reference.
#
# Raw detection needs a merge to be safe and the first ports took detection
# without it. That is now closing: read 2026-08-28, `monocr-onnx`'s Rust, Python,
# JS and Go segmenters all reference a merge. Those trees were being edited at the
# time, so treat this as dated rather than current.
#
# The value 10 is carried across rather than re-derived, which the reference's own
# header forbids doing with a tuning constant. It survives here on the evidence
# above, and on where this module's threshold sits relative to the two it is being
# borrowed from. Measured over the same 55 pages, as a multiple of this module's
# `0.02 * max(smoothed)`:
#
#     Rust binding, 0.05 * mean(non-zero)     0.55x mean, 0.49x median
#     mon_OCR reference, 0.12 * mean          1.31x mean, 1.18x median
#
# So this module's threshold is ROUGHLY BETWEEN THEM -- about twice the Rust
# binding's and about three quarters of the reference's. A higher threshold makes
# more rows read as gaps and so needs the merge more; 10 rows is not at the edge
# of either neighbour's regime, which is the most this measurement supports. It
# does NOT establish that 10 is optimal here, and nothing in these repositories
# can: no corpus scores a segmenter against labelled lines.
#
# The three clauses do different jobs. The gap bound refuses to merge real
# inter-line spacing even when overlapping diacritics hold the raw profile above
# zero across it -- upstream that unmerged case collapsed 3 PDF lines into 1. The
# ink test refuses to merge across a genuine clean break, which always has at
# least one empty row. The ceiling bounds the damage when both are satisfied
# wrongly; see `merge_runs`.
MIN_GAP_MERGE = 10


def merge_runs(runs, raw_hist, max_gap, min_line):
    """Fuse runs that a single sub-threshold row split apart.

    Merges ``runs[i]`` into ``runs[i-1]`` when the gap between them is at most
    ``max_gap`` rows AND (every row in the gap carries ink OR the shorter of the
    two runs is at most half a typical line while the taller one is at least
    ``min_line``) AND the result is at most twice a typical line. See
    ``MIN_GAP_MERGE`` for why, and for the measurement.

    ``raw_hist`` must be the RAW profile. The smoothed one bleeds ink into a gap
    that is genuinely empty, so every gap would read as ink-holding and the ink
    clause would stop refusing anything.

    A module-level function taking the profile rather than a method, so the
    arithmetic is testable without a page, a mask or a model.

    ``typical`` is the page's own MEDIAN run height and both height tests are
    relative to it rather than to the neighbouring run. That is a correction, not
    a preference: judging a fragment against its neighbour CASCADES. The merge
    mutates the accumulated run, so every merge makes it taller, and a taller run
    makes the next line look more like a fragment. Measured upstream 2026-08-28 on page 47 of a 56-page book: 36 bands
    collapsed to 10, with single bands of 534, 632 and 732 rows holding a dozen
    text lines each, and the page lost 92% of its readable characters.

    ``ceiling`` is the backstop for that cascade -- the fragment test alone cannot
    bound the result, and one runaway band costs a whole page. Twice rather than
    tighter because a legitimate merge of two halves lands at about one typical
    line and must not be refused. It is load-bearing here and not only in theory:
    over the 55 pages measured for ``MIN_GAP_MERGE`` the gap bound and the
    ink-or-fragment clause together admitted 691 candidate merges and the ceiling
    refused 482 of them. Both of the other clauses earn their place on the same
    input: 670 of those candidates had ink across the gap and 21 passed on the
    fragment test alone.

    Do NOT reach for a vertical smear instead. At reach 1 it closes 2-row gaps,
    which is the tightest real line spacing on these pages, so it fuses lines
    that are genuinely separate -- the case
    ``test_two_real_lines_two_rows_apart_stay_separate`` pins.

    ``typical`` is medianed over the runs that could BE a line -- height at least
    ``min_line`` -- and falls back to the unfiltered median when none qualify. The
    merge deliberately runs before the height filter, so ``runs`` still holds
    every speckle the profile picked up, and medianing over all of them lets noise
    decide what a typical line is. Measured here 2026-08-29 over the 55 pages
    behind ``MIN_GAP_MERGE``: 458 of 1,579 collected runs (29%) are under
    ``min_line_h``, and over the unfiltered list that drove ``typical`` under 10 on
    6 of the 55 -- ``mon_e_lib.pdf`` page 41 reached ``typical`` 4 and so a ceiling
    of 8, against a filtered median of 23. The ceiling then refuses every merge
    worth making, so the pass switched itself off on the most fragmented pages on
    the sheet. The fallback is safe rather than principled: on a page where nothing
    clears the minimum the height filter discards everything anyway, so no crop
    depends on the value.

    This was recorded here as a KNOWN LIMITATION and left for parity, on the
    grounds that ``monocr-onnx`` and the reference both medianed over the
    unfiltered list. Read again 2026-08-29, that justification is gone: three of
    the four ``monocr-onnx`` bindings filter, and the Rust binding is where the
    filtered form was designed.

    KNOWN LIMITATION, measured here 2026-08-29 and NOT fixed. The ``min_line``
    guard is on the FRAGMENT clause only; ``gap_has_ink`` carries no such guard, so
    a run of speckle whose every gap row holds ink still chains. Over the same 55
    pages, bands made entirely of sub-``min_line_h`` runs that nonetheless clear
    the height filter went from 28 to 48 -- every one of them admitted by the ink
    clause, none by the fragment clause in either form. They land on the two most
    speckled pages, ``mon_e_lib.pdf`` pages 11 and 41 (2 -> 13 and 0 -> 11), where
    the collapsed ceiling used to refuse them for the wrong reason. Guarding the
    ink clause the same way is NOT an obvious fix: an ink-bridged gap means the
    profile never reached zero, which is the signal that the split was the
    threshold's doing rather than a real break, and that is the case the clause
    exists to rescue. Left as measured rather than guessed at. All four
    ``monocr-onnx`` bindings share it.
    """
    if not runs:
        return []

    heights = [h for h in (r1 - r0 for r0, r1 in runs) if h >= min_line]
    if not heights:
        heights = [r1 - r0 for r0, r1 in runs]
    heights.sort()
    typical = max(1, heights[len(heights) // 2])
    ceiling = typical * 2

    merged = []
    for r0, r1 in runs:
        if merged:
            last0, last1 = merged[-1]
            gap_size = max(0, r0 - last1)
            # An out-of-range row counts as NO ink, not as skipped. Indexing
            # past the profile cannot happen from the run collector, but a
            # caller passing its own runs must not get a merge out of a row that
            # does not exist.
            gap_has_ink = all(
                0 <= y < len(raw_hist) and raw_hist[y] > 0
                for y in range(last1, r0)
            )
            # A run at most half a typical line is a fragment of a line, not a
            # line -- and a fragment attaches to a LINE, never to another
            # fragment. Without `max(...) >= min_line` a run of speckle merges
            # with itself: measured upstream on a 12-speck fixture, twelve 2-row
            # specks fused into one 46-row band, which CLEARS the height filter
            # and is handed to the recogniser as a line. Two pieces that are both
            # too short to be a line do not become one by being adjacent.
            ha, hb = last1 - last0, r1 - r0
            fragment = 2 * min(ha, hb) <= typical and max(ha, hb) >= min_line
            if (
                gap_size <= max_gap
                and (gap_has_ink or fragment)
                and (r1 - last0) <= ceiling
            ):
                merged[-1] = (last0, r1)
                continue
        merged.append((r0, r1))
    return merged


class LineSegmenter:
    """
    Robust line segmenter using Horizontal Projection Profiles with Smoothing.
    Handles noisy documents and touching lines by finding valleys in the projection.

    Lineage, checked against both siblings on 2026-08-28 rather than assumed.

    This is a port of ``monocr_onnx.segmenter.LineSegmenter`` and the constants
    claim holds: every one is equal -- adaptive block 25 and C 10,
    threshold_ratio 0.02, min_line_h 10, smoothing 5, pads of 0.20 and 0.15 of
    the core line height -- and the printed-rule block above is the same code
    with the same RULE_SPAN and RULE_MAX_INK_SHARE.

    Divergences from the port that are not constants, as read on 2026-08-28 in
    a tree another agent was editing. The one that changes what the two return
    for the same page -- which profile the boundaries come off -- is newest and
    has its own paragraph below. These four change how a caller talks to them,
    worst first. Treat the list as what was observed, not as closed.

    ``segment`` here returns a list of ``(crop, bbox)`` tuples; the port returns
    a list of ``{'img': ..., 'bbox': ...}`` dicts. The bbox is the same
    ``(x, y, w, h)`` either way, so the damage is entirely in the container:
    port code doing ``line['img']`` raises against this module, and code here
    doing ``crop, bbox = line`` against the port silently unpacks the dict's two
    KEYS and hands the model the string ``'img'``. That is the one that breaks a
    call site hardest, because only one direction fails loudly.

    The smoothing argument is ``smooth_kernel`` here and ``smooth_window``
    there, so a keyword call does not carry across either. This class takes PIL
    only, where the port also accepts a bare ndarray. And the port ships
    ``tile_line``/``cut_column`` for a line too wide for the model window; this
    module has neither, so an over-wide line is squeezed to fit instead of being
    split -- see ``ocr._predict_single_line``.

    It is NOT a port of the mon_OCR reference, and the distance is wider than
    tuning. The gap threshold here is a fraction of the profile MAX; the
    reference takes a fraction of the MEAN of its non-zero rows. Max and mean
    part company as lines are added to a page, so no choice of ratio reconciles
    them: 0.02 of the max and 0.12 of the mean are different algorithms at any
    number. The reference detects run boundaries on the raw profile while
    calibrating the threshold on the smoothed one, which as of 2026-08-28 this
    module does too, and as of 2026-08-28 a bounded gap merge is here as well --
    see ``MIN_GAP_MERGE``, which takes the reference's constant 10 and its
    ordering but the Rust binding's three clauses. Three of the reference's
    profile stages are still absent entirely: a pre-blur, a morphological smear,
    and outlier rejection. Its tiling is absent too and is counted separately
    below, with the other call-shape divergences. Expect different line counts on
    the same page.

    The crop geometry parts company the same way, and this list omitted it until
    2026-08-28. ``_extract_line`` here pads horizontally by
    ``int(h_raw * 0.15)``, a fraction of the line's HEIGHT. Read in
    ``mon_OCR/src/monocr/segmenter.py`` on 2026-08-28, in a tree another agent
    was editing at the time, so re-read it rather than trusting this line:
    ``pad_x = max(self.pad_x_floor_px, int(np.ceil(coreW * self.pad_x_factor)))``
    -- a fraction of the line's WIDTH, 0.05 of it, floored at an absolute 10 px.
    Both are constructor arguments there, so those are defaults a caller can
    override, not module constants. A
    short tall word and a long thin line get opposite treatment, so no constant
    reconciles those two either. The reference's header makes the same argument
    in the same terms, but scopes it to a six-implementation roster it says
    explicitly does not yet include this package, so read it as corroborating
    the shape of the claim and not the comparison. Beside it, three more in the
    same method: the
    vertical factor is 0.40 there against 0.20 here, both pads round up there
    and truncate here, and the column extents come off a dilated mask there
    against the plain binary one here, which widens the crop before any pad.

    Parity is not only about which numbers. Four things must agree before two
    bindings cut a page alike, and a value is the last of them:

    1. Which profile the run boundaries are read off -- raw or smoothed.
    2. Which statistic the threshold is calibrated on -- max, or mean of the
       non-zero rows.
    3. Which quantity each constant is a fraction of -- line height, line width,
       page extent, or absolute pixels.
    4. Only then the value itself.

    A change to any of the first three moves the cuts with every number left
    equal, and a review that diffs constants will not see it. So: changing a
    number here without changing it there is what makes two bindings disagree on
    one input, and so is changing a basis, a statistic or a profile.

    Item 1 was a live divergence and is now closed. Observed 2026-08-28 in
    ``monocr-onnx`` at commit a3e3dba, "fix(python): detect line boundaries on
    the raw profile, not the smoothed" -- local and unpushed at the time, so
    confirm before relying on it -- the port's Python binding moved its boundary
    detection to the RAW profile while still calibrating on the smoothed one.
    Read on 2026-08-28, this module was the last of six implementations still
    detecting on the smoothed profile: the port's Python, JS, Go and Rust
    bindings and the mon_OCR reference all read boundaries off the raw one. As of
    2026-08-28 ``segment`` does the same: raw for boundaries, smoothed for the
    threshold. The measurement behind it is in
    ``segment`` step 5 and is this module's own, taken at this module's
    parameters -- do not substitute the port's table, because the two round
    their smoothing window differently from the sibling bindings.

    Items 2-4 are still open, and 2 is the one to leave alone: the max-versus-
    mean split is a tuning constant the reference's spec header forbids
    reconciling.

    Which segmenter is right is still unmeasured where it counts. The raw-
    profile change carries a measurement on drawn bands at fixed gap widths, and
    the reference's gap ratio carries one too; what no repository here has is
    anything that scores a segmenter against labelled pages. So do not close the
    remaining questions by editing a number -- or a formula -- here.

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

        # 3. Horizontal projection profiles, raw and smoothed
        #
        # Both stay alive. The threshold is calibrated on the smoothed profile
        # (step 4) and the run boundaries are read off the raw one (step 5).
        # Until 2026-08-28 the convolution overwrote `hist` in place, so there
        # was no raw profile left to detect on.
        raw_hist = np.sum(binary, axis=1).astype(np.float32)  # Shape (H,)
        smoothed_hist = smooth_profile(raw_hist, self.smooth_kernel)

        # 4. Calibrate the gap threshold on the SMOOTHED profile's max
        #
        # A blank page needs no special case: its maximum is 0, so the threshold
        # is 0 and the strict `>` excludes every row. An explicit early return
        # here was dead code — mutation testing on 2026-08-16 removed it and no
        # test changed.
        #
        # Keep the statistic and the profile. This module takes a fraction of
        # the profile MAX, not of the mean of its non-zero rows, and it takes it
        # on the SMOOTHED profile. `monocr_onnx` does both the same way, read on
        # 2026-08-28, so moving either would be a formula divergence with every
        # constant still equal -- the class of change the class docstring warns
        # about.
        #
        # Whether the two calibrations are even distinguishable was measured
        # here on 2026-08-28, twice and independently, and on an ordinary page
        # they are NOT. Smoothing does not lower the peak of a band AT LEAST AS
        # TALL AS the window -- height 5 at kernel 5 already preserves it -- so
        # `max(smoothed) == max(raw)` in exact float equality on every drawn page
        # tried: 29 glyph-blob bands at gaps 1-20 and band heights 5 and up, all
        # 64,260 either way.
        #
        # Only a peak NARROWER than the window separates them, and the law is
        # `smoothed_max = min(band_h, kernel) * peak / kernel`. Band heights 1 to
        # 4 at kernel 5 gave smoothed maxima of 12,852, 25,704, 38,556 and 51,408
        # against a raw 64,260 -- so the SEPARATIONS run the other way, 51,408
        # down to 12,852. It is not "`peak / kernel` per step": that figure is
        # 12,852, which is the separation only at band height 4 and the smoothed
        # max only at height 1. Stated wrongly here until 2026-08-28, where it
        # invited a reader to re-derive 25,704 as 12,852 and conclude the
        # measurement was broken.
        #
        # Why the two decisions are SPLIT rather than moved together. This is a
        # TRADE, not a correctness result, and it was written here as a
        # correctness result until 2026-08-28 on the strength of one fixture.
        # Both directions are measured, on the same family of page:
        #
        #   A faint line and NO bridge. 1000-px page, dense 1-px spike row (raw
        #   max 127,500, smoothed max 25,500 -> thresholds 2,550 and 510), plus a
        #   faint 20-row band summing 2,295 a row, which sits between them.
        #   Smoothed calibration finds the band; raw calibration returns ZERO
        #   lines and loses the only real line on the page.
        #
        #   The same page WITH a faint bridge. Two 12-row bands at rows 40-51 and
        #   70-81 and a faint 4-column bridge across the gap summing 1,020 a row.
        #   Smoothed calibration returns ONE band, rows 32-89 -- the two lines
        #   fused. Raw calibration returns TWO, correctly split.
        #
        # So the lower threshold keeps faint lines and fuses across faint
        # bridges, and the higher one splits correctly and drops faint lines.
        # Which is better is a measurement question, and it is the same one this
        # class's docstring and the CHANGELOG both leave open: nothing in any of
        # these repositories scores a segmenter against labelled pages. The split
        # is kept because it is what every sibling implementation does, so
        # changing it here alone would break parity for no measured gain -- not
        # because the trade has been settled.
        #
        # The DETECTION half is not a trade and is not in question: reading
        # boundaries off the smoothed profile fuses lines a kernel apart with no
        # compensating benefit, which is why all six implementations moved.
        #
        # That is also why nothing at the default ratio on an ORDINARY page can
        # tell the two calibrations apart, and why
        # `test_the_gap_threshold_is_calibrated_on_the_smoothed_profile` needs a
        # sub-kernel-height spike row rather than a normal fixture.
        #
        # The max is deliberately UNSLICED, which is also safe rather than
        # merely conventional. Row sums are non-negative and the kernel is
        # uniform, so when the window is wider than the page every
        # full-convolution index in `[h_img - 1, kernel - 1]` covers the whole
        # profile and holds the same, maximal value; in that regime `mode="same"`
        # returns the window starting at full index `(h_img - 1) // 2`, so full
        # index `h_img - 1` lands inside the first `h_img` elements. Measured
        # 2026-08-28 over 138,658 exhaustive binary profiles (h 1-12, kernel
        # 1-17) and 297,712 random float32 ones: the sliced and unsliced maxima
        # never differed in value, only ever by one ULP of summation order in
        # the float64 the convolution upcasts to (worst relative gap 3.8e-16).
        max_val = np.max(smoothed_hist)
        threshold = max_val * self.threshold_ratio

        # 5. Find text rows on the RAW profile, not the smoothed one
        #
        # The smoother averages over `smooth_kernel` rows, so a gap narrower
        # than the whole window never reaches zero in the smoothed profile: the
        # ink either side bleeds into it, the bled rows clear the threshold, and
        # two distinct lines come back as one band. The raw profile needs one
        # clean row.
        #
        # MEASURED HERE on 2026-08-28, at this module's own parameters
        # (min_line_h 10, smooth_kernel 5, threshold_ratio 0.02), on 29 drawn
        # glyph-blob bands of 12 rows each, and reproduced by a second
        # from-scratch implementation of the pipeline the same day. Detecting on
        # the smoothed profile returned 1 band against 29 drawn at gaps of 1, 2,
        # 3 and 4 px, and matched the raw profile from 5 px up. Detecting on the
        # raw profile returned all 29 at every gap from 1 px.
        #
        # The break point is the smoother's FULL width -- exactly
        # `smooth_kernel`, at every kernel from 1 to 16, EVEN ONES INCLUDED.
        # `smooth_profile` is a true k-tap box at both parities, so a zero-run of
        # length g drives an output row to zero iff `g >= k`; at `g == k - 1` the
        # smoothed minimum is `band_sum / k`, far above a 2% threshold, which is
        # why the count jumps 1 -> 29 with no intermediate value.
        #
        # This is the LAW OF THIS MODULE and it is not the siblings'. The JS, Go
        # and Rust bindings hand-roll `[i - k//2, i + k//2]`, which is
        # `2 * (k // 2) + 1` taps, so their break points run 1,3,3,5,5,7,... The
        # two laws agree on odd kernels and part company on every even one,
        # where this module breaks a row EARLIER: kernel 4 fused gaps 1-3 and
        # kernel 6 fused 1-5 here, against 1-4 and 1-6 there. The absence of an
        # even-window case is why this went unnoticed. Do not carry a table
        # across in either direction. `smooth_kernel` is a constructor argument,
        # so a caller who raises it widens the failure with it.
        #
        # The `[:h_img]` bound this line used to carry is gone rather than left
        # looking load-bearing. `raw_hist` is `np.sum(binary, axis=1)` and so
        # has exactly `h_img` elements; it was the SMOOTHED profile that could
        # be longer, because `np.convolve(..., mode="same")` returns
        # max(len(hist), smooth_kernel) elements. Unbounded on the smoothed
        # profile a 3-row page produced a run of length 4, which passed a
        # min_line_h of 4 that no 3-row page can satisfy and inflated the h_raw
        # `_extract_line` derives its padding from. Detecting on the raw profile
        # makes that structurally impossible instead of guarded, so the two
        # short-page tests now defend against a regression to the smoothed
        # profile rather than against a missing slice.
        is_text_row = raw_hist > threshold

        runs: List[Tuple[int, int]] = []
        start_y = None

        for y, is_text in enumerate(is_text_row):
            if is_text and start_y is None:
                start_y = y # Start of line
            elif not is_text and start_y is not None:
                runs.append((start_y, y))
                start_y = None

        # Handle last block
        if start_y is not None:
            runs.append((start_y, h_img))

        # 6. Fuse runs a single sub-threshold row split apart, BEFORE the
        # height filter. See `MIN_GAP_MERGE` for the measurement.
        #
        # The order is the reference's and it matters. A diacritic strip can be
        # shorter than `min_line_h` -- on the measured page it was 8 rows against
        # a minimum of 10 -- so filtering first discards the strip and leaves the
        # decapitated body behind as a whole line, which is the worst of the
        # three outcomes: no error, no missing band, and a line read without its
        # asats.
        runs = merge_runs(runs, raw_hist, MIN_GAP_MERGE, self.min_line_h)

        # 7. Extract. Anything still shorter than min_line_h is a speckle, not a
        # line.
        lines: List[Tuple[Image.Image, Tuple[int, int, int, int]]] = []
        for r_start, r_end in runs:
            if (r_end - r_start) >= self.min_line_h:
                self._extract_line(binary, img_np, r_start, r_end, image, lines)

        return lines

    def _extract_line(self, binary, gray, r_start, r_end, source_image, lines_list):
        """Crop one detected line region and append it as (crop, bbox).

        Trims left and right whitespace and pads relative to the line's own
        height. A full-width crop with a fixed 4-pixel pad — what this class did
        until 2.3.0, with this method sitting unused beside it — hands the model
        a strip whose aspect ratio is the page's, not the line's, and the resize
        then squeezes the text horizontally to fit the window.

        KNOWN OFF-BY-ONE ON THE RIGHT EDGE. Recorded 2026-08-28, deliberately
        not fixed.

        ``x_end`` is the INCLUSIVE index of the last ink column, and PIL's
        ``crop`` is EXCLUSIVE on ``x2``. So the crop spans
        ``x_start - pad_x`` .. ``x_end + pad_x - 1``. Three consequences:

        * The pads are asymmetric: ``pad_x`` columns of margin on the left and
          ``pad_x - 1`` on the right. Universal, on every crop that is not
          clipped by a page edge.
        * The crop is one column short, and the bbox reports that honestly:
          ``x2 - x1`` is exactly ``crop.size[0]``, so nothing is MISreported.
          Away from the page edges that width is
          ``x_end - x_start + 2 * pad_x`` where the ink spans
          ``x_end - x_start + 1`` columns; at an edge the ``max(0, ...)`` and
          ``min(w, ...)`` clamps make even that identity not hold.
        * Ink is only LOST at ``pad_x == 0``, where the last ink column falls
          outside the crop outright -- and on a line only ONE column wide that
          is the whole line. Measured 2026-08-28 at ``min_line_h=6`` with a
          single ink column over 6 rows: bbox width ``0``, crop ``0x8``. That
          crop is then fatal rather than merely lossy, because
          ``ocr.py::_predict_single_line`` computes ``ratio = w / h = 0``, hence
          ``new_w = 0``, and ``Image.resize((0, 160))`` raises ``ValueError``.
          Unreachable through ``MonOCR``, which constructs ``LineSegmenter()``
          at the default. "One pixel short on the right" is the wrong
          description of this branch. That needs a caller passing ``min_line_h``
          below 7: ``pad_x = int(h_raw * 0.15)`` and ``h_raw >= min_line_h``, so
          ``min_line_h >= 7`` forces ``pad_x >= 1`` and the last ink column is
          always inside. At the default 10 no ink can be lost, so the data-loss
          path is not reachable through the defaults.

        It is not fixed here because it is not this module's alone. The same
        arithmetic is in ``monocr_onnx.segmenter.LineSegmenter._extract_line``
        (read 2026-08-28 in a tree another agent was editing: ``x_start,
        x_end`` at line 208, ``x2`` at 217, ``crop`` at 226) and in the
        reference ``mon_OCR/src/monocr/segmenter.py`` at HEAD 8f645ffa on
        2026-08-28 (``x0, x1`` at 841, ``coreW = x1 - x0`` at 843, ``xb`` at
        853, ``crop`` at 856), whose ``coreW`` carries the same one-column
        understatement. In the reference no ink can be lost at any DEFAULT: its
        ``pad_x`` is floored at ``_PAD_X_FLOOR_PX``, 10 px. Not "at any setting"
        -- ``pad_x_floor_px`` is a constructor argument there too
        (``segmenter.py:369, 377``), so ``pad_x_floor_px=0`` plus a one-column
        line reaches ``pad_x == 0`` in the reference as well. This class's own
        docstring already says both values are caller-overridable there.

        Shifting every crop by a pixel across three implementations -- two of
        them published packages, and one the corpus every page-level CER in this
        ecosystem was measured against -- is an owner decision, not a cleanup.
        Do not do it in one place alone: that breaks parity rather than
        restoring it.
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


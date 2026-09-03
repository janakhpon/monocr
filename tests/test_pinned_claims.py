"""Claims this package states in prose and enforced with nothing.

Every test here was written on 2026-09-03 against a *surviving mutation*: the
value was changed, the whole suite was re-run, and 142 tests stayed green. A
constant that the code argues for at length and no test reads is a comment, not
a contract.

Method note, because it cost a pass. Mutating a module and restoring it within
the same second leaves a `__pycache__` entry whose mtime still matches, and
Python keeps serving the stale bytecode -- so an *unmutated* copy failed six
tests and a mutated one passed. Every figure above was measured with
`PYTHONDONTWRITEBYTECODE=1` and `__pycache__` cleared between runs.
"""

import ast
import re
from pathlib import Path

from monocr import config
from monocr.segmenter import LineSegmenter

SEGMENTER_SOURCE = Path(config.PACKAGE_ROOT) / "segmenter.py"


def test_the_model_revision_is_a_pin_and_never_a_branch():
    """`config.py:29-33` argues this at length and nothing checked it.

    Setting `HF_REVISION = "main"` left all 142 tests passing. The comment above
    it says why that matters: without a pin `hf_hub_download` resolves `main`,
    so any re-upload to the model repository changes what an already-installed
    copy downloads -- "including changing the class count out from under the
    bundled charset". Three different networks have been served from the single
    filename `onnx/monocr.onnx` at that repository.

    `monocr-onnx` guards exactly this
    (`python/tests/test_model_manager.py::test_urls_carry_the_pinned_revision_and_never_main`).
    This package makes the same argument and shipped without the same test.
    """
    revision = config.HF_REVISION
    assert re.fullmatch(r"[0-9a-f]{7,40}", revision), (
        f"HF_REVISION is {revision!r}, which is not a commit hash. A branch or "
        "tag name here means an installed copy re-downloads whatever the model "
        "repository points at today."
    )
    assert revision not in {"main", "master", "HEAD"}


def test_the_fallback_geometry_is_v35_and_not_the_retired_network():
    """`TARGET_HEIGHT` 160 -> 128 left 142 tests green.

    128 is the v2 input height, and `ocr.py:313-314` records that naming it in a
    comment "invited someone to hardcode it back". This constant is the fallback
    used when the graph reports a dynamic height (`ocr.py:188-192`), a branch no
    fixture exercises -- every test model reports an integer height -- so the
    fallback is both unpinned and untravelled.
    """
    assert config.TARGET_HEIGHT == 160, (
        f"TARGET_HEIGHT is {config.TARGET_HEIGHT}; 160 is v3.5 and 128 is the "
        "retired v2 network. This value is only reached on a dynamic-height "
        "graph, so a wrong one fails silently rather than loudly."
    )
    assert config.TARGET_WIDTH == 1024


def test_the_segmenter_defaults_match_the_onnx_python_binding():
    """`config.py:9-15` states this contract and deliberately holds no constants.

    It says LineSegmenter's "smoothing window, threshold ratio and minimum line
    height ... have to stay equal to monocr-onnx's". Three of those survived
    mutation with 142 green, including two that silently retune every page:
    `smooth_kernel` 5 -> 3 and `threshold_ratio` 0.02 -> 0.05.

    Those two mutations are not arbitrary. 3 is the JavaScript and Go default
    and 0.05-of-non-zero-mean is the Rust one, so each mutation turns this
    binding into a different *sibling* -- which is why the suite passing is so
    misleading.

    **The contract as written is not well-formed, and this test resolves it
    narrowly.** monocr-onnx's four bindings do not agree with each other on
    these numbers, and each records why; `go/monocr.go` names Python's
    0.02-of-max against its own 0.05-of-mean. So "equal to monocr-onnx's" has
    four answers. Pinned here against
    `monocr-onnx/python/monocr_onnx/segmenter.py:297`, which is this package's
    actual sibling: `min_line_h=10, smooth_window=5, threshold_ratio=0.02`.
    Reconciling the four is a measurement question and not this test's business.
    """
    seg = LineSegmenter()
    assert (seg.min_line_h, seg.smooth_kernel, seg.threshold_ratio) == (10, 5, 0.02), (
        f"defaults are {(seg.min_line_h, seg.smooth_kernel, seg.threshold_ratio)}, "
        "which no longer match monocr_onnx's Python segmenter. Changing them "
        "changes what every page reads on; if it is deliberate, update this "
        "test and the sibling in the same change."
    )


def test_the_min_line_height_default_stays_above_the_zero_width_crop():
    """A separate assertion from the one above, because the reason is a crash.

    `segment`'s docstring derives the bound: `pad_x = int(h_raw * 0.15)` and
    `h_raw >= min_line_h`, so `min_line_h >= 7` forces `pad_x >= 1` and the
    last ink column is always inside the crop. Below 7, a one-column line over
    six rows yields bbox width 0 and a 0x8 crop; `_predict_single_line` then
    computes `ratio = w / h = 0`, calls `Image.resize((0, 160))` and raises
    `ValueError`. Measured 2026-08-28.

    The docstring says that path is "unreachable through MonOCR, which
    constructs LineSegmenter() at the default" -- so the default is the whole
    guard, and `min_line_h` 10 -> 6 left 142 tests green. This test is the
    guard the sentence assumes.
    """
    assert LineSegmenter().min_line_h >= 7, (
        f"min_line_h defaults to {LineSegmenter().min_line_h}. Below 7 a "
        "one-column line produces a zero-width crop and MonOCR raises "
        "ValueError from inside Image.resize. See segment()'s docstring."
    )


def test_segment_uses_the_result_of_the_page_rule_suppression():
    """`test_page_rules.py`'s structural test can pass while the call is a no-op.

    That test asserts `segment` *calls* `suppress_page_rules`, and its own
    message says "the unit tests above pass regardless, so this is the only
    thing pinning the wiring". Measured: rewriting
    `binary = suppress_page_rules(binary)` as a bare `suppress_page_rules(binary)`
    keeps the call, satisfies the ast test, and leaves 142 tests green -- the
    function is pure, so discarding its return value discards the suppression
    entirely.

    This checks the stronger property: the call's value is bound or passed on,
    not thrown away.
    """
    tree = ast.parse(SEGMENTER_SOURCE.read_text(encoding="utf-8"))
    segment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "segment"
    )

    discarded = [
        node.lineno
        for node in ast.walk(segment)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "suppress_page_rules"
    ]
    assert not discarded, (
        f"segment() calls suppress_page_rules at line(s) {discarded} and throws "
        "the result away. The function returns a new array and mutates nothing, "
        "so this removes printed-rule suppression while every existing test, "
        "including the structural one, still passes."
    )

    used = [
        node
        for node in ast.walk(segment)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "suppress_page_rules"
    ]
    assert used, (
        "segment() no longer calls suppress_page_rules at all. Measured "
        "2026-08-27 over the twelve MNEC page-ones: without it, nine collapsed "
        "to a single band."
    )

"""Break each fix on purpose and confirm the suite goes red.

A green suite is evidence only if it can go red. Each mutation below reverts one
of the 2.3.0 fixes to the state that shipped, or to a plausible careless edit,
and the run is expected to fail. A SURVIVED line means that fix has no guard.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MUTATIONS = [
    (
        "charset loaded with a bare .strip() (the 2.1.2 defect)",
        "src/monocr/ocr.py",
        'f.read().strip("\\n\\r")',
        "f.read().strip()",
    ),
    (
        "_decode indexes charset[idx] instead of charset[idx-1]",
        "src/monocr/ocr.py",
        "text.append(self.charset[idx-1])",
        "text.append(self.charset[idx])",
    ),
    (
        "_decode stops collapsing adjacent repeats",
        "src/monocr/ocr.py",
        "if idx != 0 and idx != prev_idx:",
        "if idx != 0:",
    ),
    (
        "_decode drops its range check",
        "src/monocr/ocr.py",
        "if 0 < idx <= len(self.charset):",
        "if True:",
    ),
    (
        "the contract check is skipped",
        "src/monocr/ocr.py",
        "if isinstance(num_classes, int) and num_classes != len(self.charset) + 1:",
        "if False:",
    ),
    (
        "input height comes from the constant, not the graph",
        "src/monocr/ocr.py",
        "self.input_height = model_height",
        "self.input_height = TARGET_HEIGHT",
    ),
    (
        "the canvas is padded black instead of white",
        "src/monocr/ocr.py",
        'new_img = Image.new("L", (target_w, target_h), 255)',
        'new_img = Image.new("L", (target_w, target_h), 0)',
    ),
    (
        "normalisation loses its sign",
        "src/monocr/ocr.py",
        "img_norm = (img_arr - IMAGE_NORM_MEAN) / IMAGE_NORM_STD",
        "img_norm = img_arr / 255.0",
    ),
    (
        "an empty segmentation returns nothing instead of the whole image",
        "src/monocr/ocr.py",
        "            return [img]\n        return lines",
        "            return []\n        return lines",
    ),
    (
        "predict goes back to the img.height > 100 dispatch",
        "src/monocr/ocr.py",
        "        return self.predict_page(image)",
        "        img = self._load(image)\n"
        "        lines = self._segment_lines(img) if img.height > 100 else [img]\n"
        '        return "\\n".join(\n'
        "            t for t in (self._predict_single_line(l) for l in lines) if t.strip()\n"
        "        )",
    ),
    (
        "predict_line segments after all",
        "src/monocr/ocr.py",
        "        return self._predict_single_line(self._load(image))",
        "        return self.predict_page(image)",
    ),
    (
        "the line crop goes back to full page width",
        "src/monocr/segmenter.py",
        "        x1 = max(0, x_start - pad_x)\n"
        "        x2 = min(gray.shape[1], x_end + pad_x)",
        "        x1 = 0\n        x2 = gray.shape[1]",
    ),
    (
        # Search string updated 2026-08-28: the scan now collects every run and
        # filters after the merge, so the comparison is on the run tuple. A
        # mutation whose string no longer matches ABORTS this harness rather
        # than failing, so an edit that moves one has to move it here too.
        "the minimum line height is dropped, so speckles become lines",
        "src/monocr/segmenter.py",
        "if (r_end - r_start) >= self.min_line_h:",
        "if (r_end - r_start) >= 1:",
    ),
    (
        "rule detection pins the erosion border, losing the edge overhang",
        "src/monocr/segmenter.py",
        "        cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(w * RULE_SPAN)), 1)),\n    )",
        "        cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(w * RULE_SPAN)), 1)),\n"
        "        borderValue=0,\n    )",
    ),
    (
        "the 15-px kernel floor drops to 1, so short strokes become rules",
        "src/monocr/segmenter.py",
        "(max(15, int(w * RULE_SPAN)), 1)",
        "(max(1, int(w * RULE_SPAN)), 1)",
    ),
    (
        "boundaries come off the smoothed profile again, bounded",
        "src/monocr/segmenter.py",
        "        is_text_row = raw_hist > threshold",
        "        is_text_row = smoothed_hist[:h_img] > threshold",
    ),
    (
        "boundaries come off the smoothed profile, unbounded (pre-2026-08-28)",
        "src/monocr/segmenter.py",
        "        is_text_row = raw_hist > threshold",
        "        is_text_row = smoothed_hist > threshold",
    ),
    (
        "the gap threshold is calibrated on the raw profile, not the smoothed",
        "src/monocr/segmenter.py",
        "        max_val = np.max(smoothed_hist)",
        "        max_val = np.max(raw_hist)",
    ),
    (
        "the row smoother loses its divisor, so the profile is a sum not a mean",
        "src/monocr/segmenter.py",
        "    kernel = np.ones(window) / window",
        "    kernel = np.ones(window)",
    ),
    (
        # The JS and RUST span law. Not Go's: Go spans the same rows but divides
        # by the requested window rather than the rows visited, so its values are
        # inflated by (window + 1) / window at an even window and this mutant does
        # not reproduce them. Named for all three until 2026-08-28.
        "an even smoothing window widens to the odd one above it (the JS/Rust law)",
        "src/monocr/segmenter.py",
        "    kernel = np.ones(window) / window\n"
        '    return np.convolve(raw_hist, kernel, mode="same")',
        "    span = 2 * (window // 2) + 1\n"
        "    kernel = np.ones(span) / span\n"
        '    return np.convolve(raw_hist, kernel, mode="same")',
    ),
    (
        "smooth_profile stops short-circuiting a window of 1",
        "src/monocr/segmenter.py",
        "    if window <= 1:\n        return raw_hist",
        "    if window <= 0:\n        return raw_hist",
    ),
    # An off-by-one on the old `hist[: h_img]` scan bound was carried here as an
    # equivalent mutant until 2026-08-28. The bound itself is gone now: the scan
    # reads `raw_hist`, which is `np.sum(binary, axis=1)` and so has exactly
    # `h_img` elements, and a slice that can never bite is worse than no slice.
    # The property it defended is covered by the two smoothed-profile mutations
    # above, which are the only way phantom rows come back.

    # The bounded gap merge. Raw-profile detection without it splits Mon lines at
    # the diacritic zone; see `MIN_GAP_MERGE` for the 55-page measurement.
    (
        "the merge call is deleted from the pipeline, leaving raw detection alone",
        "src/monocr/segmenter.py",
        "        runs = merge_runs(runs, raw_hist, MIN_GAP_MERGE, self.min_line_h)",
        "        runs = runs",
    ),
    (
        "the merge runs AFTER the height filter, so a short strip is discarded",
        "src/monocr/segmenter.py",
        "        runs = merge_runs(runs, raw_hist, MIN_GAP_MERGE, self.min_line_h)",
        "        runs = merge_runs(\n"
        "            [(a, b) for a, b in runs if (b - a) >= self.min_line_h],\n"
        "            raw_hist,\n"
        "            MIN_GAP_MERGE,\n"
        "            self.min_line_h,\n"
        "        )",
    ),
    (
        "the merge drops its ink clause",
        "src/monocr/segmenter.py",
        "                and (gap_has_ink or fragment)",
        "                and fragment",
    ),
    (
        "the merge drops its fragment clause",
        "src/monocr/segmenter.py",
        "                and (gap_has_ink or fragment)",
        "                and gap_has_ink",
    ),
    (
        "the merge drops its gap bound, so any ink-bridged gap fuses",
        "src/monocr/segmenter.py",
        "                gap_size <= max_gap\n",
        "                True\n",
    ),
    (
        "the merge ceiling drops from two typical lines to one",
        "src/monocr/segmenter.py",
        "    ceiling = typical * 2",
        "    ceiling = typical * 1",
    ),
    (
        "the merge ceiling is removed, so merges cascade unbounded",
        "src/monocr/segmenter.py",
        "                and (r1 - last0) <= ceiling",
        "                and True",
    ),
    (
        "a fragment is judged against its NEIGHBOUR instead of the page median",
        "src/monocr/segmenter.py",
        "            fragment = 2 * min(ha, hb) <= typical and max(ha, hb) >= min_line",
        "            fragment = 2 * min(ha, hb) <= max(ha, hb) and max(ha, hb) >= min_line",
    ),
    (
        "the fragment ratio loosens from half a typical line to a whole one",
        "src/monocr/segmenter.py",
        "            fragment = 2 * min(ha, hb) <= typical and max(ha, hb) >= min_line",
        "            fragment = min(ha, hb) <= typical and max(ha, hb) >= min_line",
    ),
    (
        # The sibling Python binding's defect, faithfully: a numpy slice truncates
        # silently, so a gap running past the profile becomes an EMPTY slice and
        # `np.all` of nothing is True. It merged across rows that do not exist.
        "the ink test reads a numpy slice, so an out-of-range gap counts as inked",
        "src/monocr/segmenter.py",
        "            gap_has_ink = all(\n"
        "                0 <= y < len(raw_hist) and raw_hist[y] > 0\n"
        "                for y in range(last1, r0)\n"
        "            )",
        "            gap_has_ink = bool(np.all(raw_hist[last1:r0] > 0))",
    ),
    # `typical` over the filtered list, and the fragment clause's line guard.
    # Ported 2026-08-29; before that this module had the other two clauses only.
    (
        "typical is medianed over EVERY collected run again, speckle included",
        "src/monocr/segmenter.py",
        "    heights = [h for h in (r1 - r0 for r0, r1 in runs) if h >= min_line]\n"
        "    if not heights:\n"
        "        heights = [r1 - r0 for r0, r1 in runs]\n"
        "    heights.sort()",
        "    heights = sorted(r1 - r0 for r0, r1 in runs)",
    ),
    (
        "the filtered median loses its fallback, so an all-speckle page divides "
        "by an empty list",
        "src/monocr/segmenter.py",
        "    if not heights:\n        heights = [r1 - r0 for r0, r1 in runs]\n",
        "",
    ),
    (
        "the fragment clause loses its line guard, so speckle chains into a band",
        "src/monocr/segmenter.py",
        "            fragment = 2 * min(ha, hb) <= typical and max(ha, hb) >= min_line",
        "            fragment = 2 * min(ha, hb) <= typical",
    ),
    (
        "the line guard tests the SHORTER run instead of the taller one",
        "src/monocr/segmenter.py",
        "            fragment = 2 * min(ha, hb) <= typical and max(ha, hb) >= min_line",
        "            fragment = 2 * min(ha, hb) <= typical and min(ha, hb) >= min_line",
    ),
    (
        "the ratio measures the TALLER run against typical instead of the shorter",
        "src/monocr/segmenter.py",
        "            fragment = 2 * min(ha, hb) <= typical and max(ha, hb) >= min_line",
        "            fragment = 2 * max(ha, hb) <= typical and max(ha, hb) >= min_line",
    ),
    (
        # The pair that masks itself. An unfiltered median collapses `typical` far
        # enough that an unguarded fragment clause reaches the right answer, so
        # reverting either one alone is caught by more cases than reverting both.
        # `test_a_fragment_never_attaches_to_another_fragment` is the case that
        # sees it: its median is 40 filtered or not, so only the guard is left.
        "both 2026-08-29 decisions are reverted together (the state five ports "
        "were in)",
        "src/monocr/segmenter.py",
        [
            "    heights = [h for h in (r1 - r0 for r0, r1 in runs) if h >= min_line]\n"
            "    if not heights:\n"
            "        heights = [r1 - r0 for r0, r1 in runs]\n"
            "    heights.sort()",
            "            fragment = 2 * min(ha, hb) <= typical and max(ha, hb) >= min_line",
        ],
        [
            "    heights = sorted(r1 - r0 for r0, r1 in runs)",
            "            fragment = 2 * hb <= typical or 2 * ha <= typical",
        ],
    ),
    (
        "the merge reads the SMOOTHED profile, so every gap looks ink-holding",
        "src/monocr/segmenter.py",
        "        runs = merge_runs(runs, raw_hist, MIN_GAP_MERGE, self.min_line_h)",
        "        runs = merge_runs(runs, smoothed_hist, MIN_GAP_MERGE, self.min_line_h)",
    ),
    (
        "opencv-python replaces the headless build",
        "pyproject.toml",
        '"opencv-python-headless>=4.0.0",',
        '"opencv-python>=4.0.0",',
    ),
    (
        "__version__ goes back to a hand-maintained literal that drifts",
        "src/monocr/__init__.py",
        '    __version__ = _installed_version("monocr")',
        '    __version__ = "9.9.9"',
    ),
    (
        "CharsetNotFoundError is dropped from __all__",
        "src/monocr/__init__.py",
        '    "CharsetNotFoundError",\n',
        "",
    ),
    (
        "CharsetNotFoundError is no longer imported at the top level",
        "src/monocr/__init__.py",
        "    CharsetNotFoundError,\n",
        "",
    ),
    (
        "the CLI stops fetching the charset beside the weights",
        "src/monocr/cli.py",
        "        charset = get_cached_model_path(\n"
        "            repo_id=HF_REPO_ID,\n"
        "            filename=HF_CHARSET_FILENAME,\n"
        "            revision=HF_REVISION,\n"
        "            force_download=True,\n"
        "        )",
        '        charset = "skipped"',
    ),
]


def run_suite():
    return subprocess.run(
        ["uv", "run", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


# Snapshot the working tree in memory. `git checkout --` restores to HEAD, not
# to the working state, so using it here deletes every uncommitted edit on the
# first iteration and silently invalidates every result after it.
SNAPSHOT = {rel: (ROOT / rel).read_text() for rel in {m[1] for m in MUTATIONS}}


def edits(old, new):
    """One mutation may need more than one edit.

    Two of the four merge decisions MASK each other -- an unfiltered `typical`
    collapses the ceiling far enough that an unguarded fragment clause reaches the
    same bands -- so reverting the pair together is a different mutant from
    reverting either alone, and it cannot be written as a single string swap. A
    mutation gives lists instead of strings when it needs several.
    """
    if isinstance(old, str):
        return [(old, new)]
    assert len(old) == len(new), "an edit list needs a replacement per search"
    return list(zip(old, new))


def restore():
    for rel, text in SNAPSHOT.items():
        (ROOT / rel).write_text(text)


baseline = run_suite()
if baseline.returncode != 0:
    sys.exit(f"baseline is already red, fix that first:\n{baseline.stdout[-2000:]}")
print(f"baseline green: {baseline.stdout.strip().splitlines()[-1]}\n")

survived = []
for label, relpath, old, new in MUTATIONS:
    target = ROOT / relpath
    source = target.read_text()
    for search, replacement in edits(old, new):
        if source.count(search) != 1:
            restore()
            sys.exit(
                f"MUTATION NOT APPLICABLE ({source.count(search)} matches): {label}"
            )
        source = source.replace(search, replacement, 1)
    target.write_text(source)

    result = run_suite()
    restore()

    if result.returncode == 0:
        survived.append(label)
        print(f"  SURVIVED  {label}")
    else:
        first = next(
            (l for l in result.stdout.splitlines() if l.startswith("FAILED")), ""
        )
        print(f"  killed    {label}\n              by {first[7:90] or 'assertion'}")

print()
if survived:
    print(f"{len(survived)} mutation(s) survived:")
    for label in survived:
        print(f"  - {label}")
    sys.exit(1)
print(f"all {len(MUTATIONS)} mutations killed")

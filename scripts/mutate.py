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
        "the minimum line height is dropped, so speckles become lines",
        "src/monocr/segmenter.py",
        "if (y - start_y) >= self.min_line_h:",
        "if (y - start_y) >= 1:",
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
        "the row scan is unbounded again, so it runs past a short page",
        "src/monocr/segmenter.py",
        "        is_text_row = hist[:h_img] > threshold",
        "        is_text_row = hist > threshold",
    ),
    # `hist[: h_img + 1]` was tried here as a second, off-by-one mutation and is
    # not listed because it is an equivalent mutant, not a survivor: index h_img
    # can only ever terminate a run, and terminating there gives r_end == h_img,
    # which is what the trailing branch already does. A run starting at h_img has
    # length 0 and `_extract_line` drops it for having no ink columns. Checked
    # exhaustively on 2026-08-28 over 14,280 cases (h 1-8, kernel 3/5/9/15,
    # min_line_h 0-9): zero differing outcomes. Do not re-add it.
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
    if source.count(old) != 1:
        restore()
        sys.exit(f"MUTATION NOT APPLICABLE ({source.count(old)} matches): {label}")
    target.write_text(source.replace(old, new, 1))

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

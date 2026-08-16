"""The charset/model contract, and getting an image in the door.

Two tests were deleted from this file rather than fixed:

  * `test_api_methods_exist` wrapped its assertions in `except (ModelNotFoundError,
    Exception)`, which catches AssertionError. Every method could be removed from
    MonOCR and it reported "skipped" — green.
  * `test_read_text_function_exists` asserted that an imported function was
    callable.

`unittest.main()` also sat at line 80 of this file, above four more tests. Under
`python tests/test_monocr.py` it exited before reaching them, so the tests
guarding the shipped decode defect did not run in the one way the file invites
you to run it. The suite is pytest now, top to bottom, and there is no early
exit to fall through.
"""

import numpy as np
import pytest
from PIL import Image

from monocr import config
from monocr.exceptions import ImageLoadError
from monocr.ocr import MonOCR

# ---------------------------------------------------------------------------
# The bundled charset, against the pinned revision
# ---------------------------------------------------------------------------


def test_the_bundled_charset_matches_the_pinned_model(bundled_charset):
    """277 classes need exactly 276 characters. 2.1.2 shipped 914 against 316."""
    assert len(bundled_charset) == 276


def test_the_charset_starts_with_a_space_and_stripping_keeps_it():
    raw = open(config.CHARSET_PATH, encoding="utf-8").read()
    assert raw[0] == " ", "the first class is U+0020"
    assert raw.strip("\n\r")[0] == " "
    assert len(raw.strip()) != len(raw.strip("\n\r")), (
        "if these are equal the leading space is already gone and this test no "
        "longer guards anything"
    )


def test_a_revision_is_pinned():
    """Without one, hf_hub_download resolves `main`, so any upload to the model
    repository changes what already-installed copies fetch."""
    assert getattr(config, "HF_REVISION", None)


def test_the_charset_is_fetched_from_the_same_revision_as_the_weights():
    assert config.HF_CHARSET_FILENAME.startswith("onnx/")
    assert config.HF_FILENAME.startswith("onnx/")


# ---------------------------------------------------------------------------
# _prepare_image
# ---------------------------------------------------------------------------


def test_a_pil_image_is_converted_to_grayscale(make_ocr):
    ocr = make_ocr()
    assert ocr._prepare_image(Image.new("RGB", (200, 100), "red")).mode == "L"


def test_an_image_on_disk_is_opened_and_converted(make_ocr, tmp_path):
    ocr = make_ocr()
    path = tmp_path / "line.png"
    Image.fromarray(np.zeros((100, 200), dtype=np.uint8)).save(path)

    assert ocr._prepare_image(str(path)).mode == "L"
    assert ocr._prepare_image(path).mode == "L", "a Path, not only a str"


def test_a_missing_file_raises_image_load_error(make_ocr):
    ocr = make_ocr()
    with pytest.raises(ImageLoadError):
        ocr._prepare_image("no-such-image.png")


def test_a_file_that_is_not_an_image_raises_image_load_error(make_ocr, tmp_path):
    ocr = make_ocr()
    path = tmp_path / "not-an-image.png"
    path.write_text("hello")
    with pytest.raises(ImageLoadError):
        ocr._prepare_image(str(path))


def test_predict_reports_a_bad_image_as_image_load_error(make_ocr):
    """Not as whatever Pillow happened to raise."""
    ocr = make_ocr()
    with pytest.raises(ImageLoadError):
        ocr.predict("no-such-image.png")


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def test_read_from_folder_keys_results_by_filename(make_ocr, tmp_path):
    ocr = make_ocr()
    for name in ("a.png", "b.jpg"):
        Image.new("L", (200, 100), 255).save(tmp_path / name)

    results = ocr.read_from_folder(str(tmp_path))
    assert set(results) == {"a.png", "b.jpg"}


def test_read_from_folder_records_an_empty_string_for_a_file_it_cannot_read(
    make_ocr, tmp_path
):
    ocr = make_ocr()
    (tmp_path / "broken.png").write_text("not a png")
    assert ocr.read_from_folder(str(tmp_path)) == {"broken.png": ""}


def test_predict_batch_returns_one_result_per_image(make_ocr):
    ocr = make_ocr()
    images = [Image.new("L", (200, 100), 255) for _ in range(3)]
    assert len(ocr.predict_batch(images)) == 3


def test_read_text_is_predict(make_ocr):
    ocr = make_ocr()
    assert MonOCR.read_text(ocr, Image.new("L", (200, 100), 255)) == ocr.predict(
        Image.new("L", (200, 100), 255)
    )

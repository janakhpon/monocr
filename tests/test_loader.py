"""Loading a model and its charset.

The 2.1.2 release shipped a 914-character charset against a 316-class model,
read with `.strip()`. Two independent faults, each sufficient on its own to make
every decode wrong, and neither one raised. These tests call `load_model`, so
reverting either fix turns them red — the previous suite asserted what `.strip()`
*would* do to a string without ever calling the loader, which is why it stayed
green through the whole defect.
"""

import pytest

from monocr.exceptions import CharsetNotFoundError, ModelNotFoundError
from monocr.ocr import MonOCR


def test_the_loader_keeps_the_leading_space(make_ocr):
    """`.strip()` here removes U+0020, the first class, and shifts every index."""
    ocr = make_ocr(charset=" abc\n", num_classes=5)
    assert ocr.charset == " abc"
    assert ocr.charset[0] == " "


def test_the_loader_drops_only_line_endings(make_ocr):
    """CRLF as well as LF, and no other whitespace."""
    ocr = make_ocr(charset=" abc\r\n", num_classes=5)
    assert ocr.charset == " abc"


def test_a_charset_that_cannot_match_the_graph_is_refused(make_ocr):
    """3 characters against 277 classes. Decoding would still return text."""
    with pytest.raises(CharsetNotFoundError, match="charset/model mismatch"):
        make_ocr(charset="abc", num_classes=277)


def test_the_exact_off_by_one_is_refused(make_ocr):
    """276 classes needs 275 characters. One character out is the likeliest
    mistake and the hardest to see in the output."""
    with pytest.raises(CharsetNotFoundError):
        make_ocr(charset="x" * 276, num_classes=276)


def test_the_matching_pair_loads(make_ocr):
    ocr = make_ocr(charset="x" * 276, num_classes=277)
    assert len(ocr.charset) == 276


def test_a_dynamic_class_axis_is_not_treated_as_a_count(make_ocr):
    """A string axis is unknown, not a mismatch — refusing it would break any
    model exported with a dynamic vocabulary."""
    ocr = make_ocr()
    ocr.session._outputs[0].shape = [1, "sequence", "classes"]
    ocr.charset = "abc"
    ocr._check_contract()  # must not raise


def test_a_missing_model_file_raises_before_onnxruntime_sees_it():
    with pytest.raises(ModelNotFoundError):
        MonOCR(model_path="no-such-model.onnx")


def test_a_missing_charset_raises_rather_than_decoding_to_nothing(
    make_ocr, monkeypatch, tmp_path
):
    """With no charset there is no alphabet, and every index is out of range —
    `_decode` would return "" for every image and report no error."""
    from monocr import ocr as ocr_module

    monkeypatch.setattr(ocr_module, "CHARSET_PATH", str(tmp_path / "absent.txt"))
    with pytest.raises(CharsetNotFoundError):
        make_ocr()


def test_the_charset_beside_the_weights_wins_over_the_bundled_one(
    make_ocr, monkeypatch, tmp_path
):
    """The downloaded pair is authoritative. A bundled copy left behind from an
    older revision is the failure this ordering exists to prevent."""
    from monocr import ocr as ocr_module

    downloaded = tmp_path / "downloaded.txt"
    downloaded.write_text("y" * 276, encoding="utf-8")

    real_init = ocr_module.MonOCR.__init__

    def init_with_downloaded(self, *args, **kwargs):
        self._downloaded_charset_path = str(downloaded)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(ocr_module.MonOCR, "__init__", init_with_downloaded)
    ocr = make_ocr(charset="x" * 276)
    assert ocr.charset == "y" * 276


def test_predicting_before_a_model_is_loaded_raises(make_ocr):
    ocr = make_ocr()
    ocr.session = None
    with pytest.raises(RuntimeError, match="before loading"):
        ocr.predict("anything.png")

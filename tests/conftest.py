"""Shared fixtures.

Nothing here touches the network or loads a real ONNX file. Every test in this
suite used to begin with "try to build a MonOCR, and skipTest if the model is
not downloadable" — which meant the suite was green on a machine with no model
and green on a machine with a broken one, and the 2.1.2 decode defect passed
through it untouched.

The contract this package has to defend is expressible as two numbers off the
graph, the class count and the input height, so a fake session reporting those
numbers exercises it exactly as a 46MB download would, and does it in
milliseconds with no way to be skipped.
"""

import numpy as np
import pytest

from monocr import ocr as ocr_module


class FakeIO:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class FakeSession:
    """Stands in for ``onnxruntime.InferenceSession``.

    ``shape`` entries that are strings are dynamic axes, matching how
    onnxruntime reports them — the real pinned model reports ``[1, 1, 160,
    1024]`` and ``[1, 'sequence', 277]``.
    """

    def __init__(self, num_classes=277, height=160, width=1024, logits=None):
        self._inputs = [FakeIO("input", [1, 1, height, width])]
        self._outputs = [FakeIO("logits", [1, "sequence", num_classes])]
        self._logits = logits
        self.last_input = None
        self.calls = 0

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feed):
        self.calls += 1
        self.last_input = next(iter(feed.values()))
        if self._logits is None:
            return [np.zeros((1, 4, self._outputs[0].shape[-1]), dtype=np.float32)]
        return [self._logits]


@pytest.fixture
def make_ocr(monkeypatch, tmp_path):
    """Build a MonOCR wired to a FakeSession.

    The model file is real but empty — ``load_model`` only checks that the path
    exists before handing it to onnxruntime, which is patched out. The charset
    is the one bundled in the package unless a replacement is passed, so the
    loader under test is the shipped one.
    """

    def _make(charset=None, **session_kwargs):
        session = FakeSession(**session_kwargs)
        monkeypatch.setattr(
            ocr_module.ort, "InferenceSession", lambda path, providers=None: session
        )
        if charset is not None:
            charset_file = tmp_path / "charset.txt"
            charset_file.write_text(charset, encoding="utf-8")
            monkeypatch.setattr(ocr_module, "CHARSET_PATH", str(charset_file))

        model_file = tmp_path / "monocr.onnx"
        model_file.write_bytes(b"")
        instance = ocr_module.MonOCR(model_path=str(model_file))
        instance.fake_session = session
        return instance

    return _make


@pytest.fixture
def bundled_charset():
    """The charset the package ships, loaded the way load_model loads it."""
    from monocr.config import CHARSET_PATH

    with open(CHARSET_PATH, encoding="utf-8") as f:
        return f.read().strip("\n\r")

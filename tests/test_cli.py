"""The command line interface.

92 shipped lines with no coverage at all, and it is the surface most users of
this package touch first — `monocr read image.png`. click ships a runner and
click is already a dependency, so there was never a cost reason for this gap.
"""

import pytest
from click.testing import CliRunner

from monocr.cli import main
from monocr.exceptions import ModelNotFoundError


class StubOCR:
    """A MonOCR that answers without a model."""

    text = "မန်"
    confidence = 0.91
    folder = {"b.png": "second", "a.png": "first"}

    def __init__(self, *args, **kwargs):
        pass

    def predict(self, path):
        return self.text

    def predict_with_confidence(self, path):
        return {"text": self.text, "confidence": self.confidence}

    def read_from_folder(self, path):
        return self.folder


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "line.png"
    path.write_bytes(b"not really a png, click only checks it exists")
    return str(path)


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr("monocr.cli.MonOCR", StubOCR)
    return StubOCR


def test_version_reports_the_installed_metadata(runner):
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    import monocr

    assert monocr.__version__ in result.output


def test_read_prints_the_recognised_text(runner, stub, image):
    result = runner.invoke(main, ["read", image])
    assert result.exit_code == 0
    assert StubOCR.text in result.output


def test_read_says_so_when_nothing_was_found(runner, monkeypatch, image):
    class Empty(StubOCR):
        text = "   "

    monkeypatch.setattr("monocr.cli.MonOCR", Empty)
    result = runner.invoke(main, ["read", image])
    assert result.exit_code == 0
    assert "couldn't find any text" in result.output


def test_read_with_confidence_prints_a_percentage(runner, stub, image):
    result = runner.invoke(main, ["read", "--confidence", image])
    assert result.exit_code == 0
    assert "91.00%" in result.output
    assert StubOCR.text in result.output


def test_read_rejects_a_path_that_does_not_exist(runner, stub):
    result = runner.invoke(main, ["read", "no-such-file.png"])
    assert result.exit_code != 0


def test_read_exits_nonzero_when_the_model_is_missing(runner, monkeypatch, image):
    def explode(*args, **kwargs):
        raise ModelNotFoundError("no model")

    monkeypatch.setattr("monocr.cli.MonOCR", explode)
    result = runner.invoke(main, ["read", image])
    assert result.exit_code == 1


def test_read_exits_nonzero_on_an_unexpected_error(runner, monkeypatch, image):
    """A bare crash must not look like a successful read of nothing."""

    def explode(*args, **kwargs):
        raise ValueError("something else entirely")

    monkeypatch.setattr("monocr.cli.MonOCR", explode)
    result = runner.invoke(main, ["read", image])
    assert result.exit_code == 1


def test_batch_prints_every_result(runner, stub, tmp_path):
    result = runner.invoke(main, ["batch", str(tmp_path)])
    assert result.exit_code == 0
    assert "first" in result.output and "second" in result.output
    assert result.output.index("a.png") < result.output.index("b.png"), "sorted"


def test_batch_says_so_on_an_empty_folder(runner, monkeypatch, tmp_path):
    class Empty(StubOCR):
        folder = {}

    monkeypatch.setattr("monocr.cli.MonOCR", Empty)
    result = runner.invoke(main, ["batch", str(tmp_path)])
    assert result.exit_code == 0
    assert "No images found" in result.output


def test_batch_rejects_a_file_where_a_folder_is_required(runner, stub, image):
    assert runner.invoke(main, ["batch", image]).exit_code != 0


def test_download_fetches_the_weights_and_the_charset(runner, monkeypatch):
    """Both, at one revision. Fetching weights without the charset beside them
    is how a decode ends up in the wrong alphabet."""
    asked = []

    def fake_download(repo_id, filename, revision, force_download=False):
        asked.append((filename, revision))
        return f"/cache/{filename}"

    monkeypatch.setattr("monocr.cli.get_cached_model_path", fake_download)
    result = runner.invoke(main, ["download"])

    assert result.exit_code == 0
    from monocr.config import HF_CHARSET_FILENAME, HF_FILENAME, HF_REVISION

    assert asked == [(HF_FILENAME, HF_REVISION), (HF_CHARSET_FILENAME, HF_REVISION)]


def test_download_exits_nonzero_when_the_hub_is_unreachable(runner, monkeypatch):
    def explode(**kwargs):
        raise OSError("no network")

    monkeypatch.setattr("monocr.cli.get_cached_model_path", explode)
    assert runner.invoke(main, ["download"]).exit_code == 1

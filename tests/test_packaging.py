"""Version numbers and the public surface.

`pyproject.toml` and `__init__.py` each carry the version, maintained by hand
and independently. `monocr --version` reads the installed metadata, so the CLI
and `monocr.__version__` can report different numbers with nothing to notice —
and a bug report then names a release that was never built from this tree.
"""

import tomllib
from importlib.metadata import version as installed_version
from pathlib import Path

import pytest

import monocr

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def declared_version():
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)["project"]["version"]


def test_the_installed_metadata_matches_the_module_attribute():
    """What `monocr --version` prints, against what `monocr.__version__` says."""
    assert installed_version("monocr") == monocr.__version__


def test_pyproject_matches_the_module_attribute():
    assert declared_version() == monocr.__version__


def test_the_charset_error_is_reachable_from_the_top_level():
    """The package's most important exception. Callers should not have to know
    it lives in a private-looking submodule to catch a charset/model mismatch."""
    assert monocr.CharsetNotFoundError.__name__ == "CharsetNotFoundError"
    assert issubclass(monocr.CharsetNotFoundError, monocr.MonOCRError)


@pytest.mark.parametrize(
    "name",
    ["MonOCR", "MonOCRError", "ModelNotFoundError", "CharsetNotFoundError",
     "ImageLoadError", "read_text", "read_folder"],
)
def test_the_public_surface_is_declared(name):
    """Named explicitly rather than read off `__all__`.

    Parametrising over `monocr.__all__` only tests the names that are in it, so
    deleting an entry deletes its own test case and the suite stays green.
    """
    assert name in monocr.__all__, f"{name} is missing from __all__"
    assert hasattr(monocr, name)


@pytest.mark.parametrize("name", monocr.__all__)
def test_every_exported_name_exists(name):
    assert hasattr(monocr, name), f"__all__ promises {name} and it is not there"


def test_the_dependency_on_opencv_is_the_headless_build():
    """cv2 is imported unconditionally by segmenter.py, so the GUI build makes
    `import monocr` fail on any image lacking the X libraries OpenCV links
    against. This only checks the declaration; the CI job that installs the
    wheel in python:3.12-slim is what checks the behaviour."""
    with open(PYPROJECT, "rb") as f:
        deps = tomllib.load(f)["project"]["dependencies"]
    assert any(d.startswith("opencv-python-headless") for d in deps)
    assert not any(d.startswith("opencv-python>") or d == "opencv-python" for d in deps)

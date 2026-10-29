# Releasing

`pyproject.toml` holds the only version in the tree; `monocr.__version__` reads
it back from the installed metadata. The tag has to match it, and the release
workflow fails the build if it does not.

```bash
uv version --bump patch          # or minor / major
git commit -am "release: $(uv version --short)"
git tag "v$(uv version --short)"
git push origin HEAD --tags
```

Publishing runs from the tag, through GitHub Actions
(`.github/workflows/release.yml`), using PyPI trusted publishing. It installs
from the lockfile and runs the test suite first.

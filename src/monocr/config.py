from pathlib import Path

# Image processing constants
TARGET_WIDTH = 1024
TARGET_HEIGHT = 128
IMAGE_NORM_MEAN = 127.5
IMAGE_NORM_STD = 127.5

# Segmentation constants
PROJECTION_THRESHOLD = 2
MIN_LINE_GAP = 5
BINARY_THRESHOLD = 200

# Paths
PACKAGE_ROOT = Path(__file__).parent
ASSETS_DIR = PACKAGE_ROOT / "assets"
CHARSET_PATH = ASSETS_DIR / "valid_chars.txt"

# Model cache configuration
CACHE_DIR = Path.home() / ".monocr" / "models"
MODEL_FILENAME = "monocr.onnx"
DEFAULT_MODEL_PATH = CACHE_DIR / MODEL_FILENAME

# Model download configuration.
#
# HF_REVISION is not optional. Without it hf_hub_download resolves `main`, so
# every re-upload to the model repository silently changes what an installed
# copy of this package downloads — including changing the class count out from
# under the bundled charset. Bump this deliberately, together with the charset
# below and a release.
HF_REPO_ID = "janakhpon/monocr"
HF_REVISION = "a51be11"
HF_FILENAME = "onnx/monocr.onnx"

# The charset published beside the weights at that same revision. It is
# downloaded rather than trusted from the package, because the pair is what has
# to agree: a charset from one revision against a graph from another decodes
# every index to the wrong character. valid_chars.txt is the offline fallback
# and must match this revision.
HF_CHARSET_FILENAME = "onnx/charset.txt"

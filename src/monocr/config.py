from pathlib import Path

# Image processing constants
TARGET_WIDTH = 1024
TARGET_HEIGHT = 64
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
MODEL_FILENAME = "monocr.ckpt"
DEFAULT_MODEL_PATH = CACHE_DIR / MODEL_FILENAME

# Model download configuration
MODEL_URL = "https://github.com/janakhpon/monocr/releases/download/v0.1.15/monocr.ckpt"
MODEL_SHA256 = "485ebc4a04e450d2970621271839114048d13dd11601e77f76852e42f41c1184"

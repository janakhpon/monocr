import os
import onnxruntime as ort
import numpy as np
import logging
from PIL import Image, ImageOps, UnidentifiedImageError
from typing import List, Optional, Union, Dict
from pathlib import Path

from .segmenter import LineSegmenter
from .config import (
    IMAGE_NORM_MEAN, IMAGE_NORM_STD,
    TARGET_WIDTH, TARGET_HEIGHT,
    CHARSET_PATH
)
from .exceptions import (
    ModelNotFoundError, CharsetNotFoundError, ImageLoadError
)
from .download import get_cached_model_path

logger = logging.getLogger(__name__)

# The model is trained on dark text on a light background, and this package never
# checked which it was given.
#
# Measured 2026-08-27 over 300 labelled crops from mon_OCR's
# `data/real/digits/val`, same graph, only the polarity of the input changed:
#
#     upright, with this probe      CER 0.0000   300/300 exact
#     inverted, with this probe     CER 0.0000   300/300 exact
#     upright, without it           CER 0.0036   296/300
#     inverted, without it          CER 0.0342   288/300   <- 9.5x worse
#
# Degradation rather than the total failure it might sound like, and cheap to
# close: four corner patches. The caveat is that those crops are Myanmar digits
# on composited backgrounds, so the effect on full Mon text lines is unmeasured.
#
# A COPY of `mon_OCR/src/monocr/utils.py::to_normalized_grayscale`'s steps 1-3,
# deliberately not a shared module: these packages ship independently and a
# dependency across that boundary is coupling their own docs refuse. Step 4 of
# that function, background levelling, is NOT ported here -- it is what the
# 0.0036 upright row above costs, and it is a separate change.
_POLARITY_CORNER_FRACTION = 10
_POLARITY_CORNER_FLOOR = 3
_DARK_BACKGROUND_MEDIAN = 128


def normalize_polarity(img: Image.Image) -> Image.Image:
    """Return `img` as dark-text-on-light, inverting it if the background is dark.

    Corner-median rather than a global mean: document corners are almost always
    background, so their median survives a dense, text-heavy page where a global
    mean is dragged toward the ink. A page that is already dark-on-light comes
    back unchanged, which is what makes this safe on every input.
    """
    arr = np.asarray(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    ch = max(_POLARITY_CORNER_FLOOR, h // _POLARITY_CORNER_FRACTION)
    cw = max(_POLARITY_CORNER_FLOOR, w // _POLARITY_CORNER_FRACTION)
    corners = np.concatenate(
        [
            arr[:ch, :cw].ravel(),
            arr[:ch, w - cw :].ravel(),
            arr[h - ch :, :cw].ravel(),
            arr[h - ch :, w - cw :].ravel(),
        ]
    )
    if float(np.median(corners)) < _DARK_BACKGROUND_MEDIAN:
        return ImageOps.invert(img)
    return img


class MonOCR:
    """
    Mon OCR Inference Class.
    Supports single-line and multi-line (paragraph) Mon text recognition.
    Powered by ONNX Runtime for high-performance inference.
    """
    
    def __init__(self, model_path: Optional[str] = None, providers: Optional[List[str]] = None):
        """
        Initialize Mon OCR.
        
        Args:
            model_path: Path to the .onnx model file. If None, downloads the
                pinned v3.5 model — `janakhpon/monocr` at revision `d3d9d5e`,
                160x1024 input, 277 classes against the 276-character charset.
                See config.HF_REVISION; the pin is what stops `main` moving
                under an installed copy.
            providers: ONNX Runtime execution providers (e.g. ['CUDAExecutionProvider', 'CPUExecutionProvider']).
        """
        self.session = None
        self.charset = None
        self.providers = providers or ['CPUExecutionProvider']
        
        if model_path is None:
            # check for model in package (dev env)
            local_model = Path(__file__).parent / "models" / "monocr.onnx"
            if local_model.exists():
                model_path = str(local_model)
                logger.info(f"found local model in package: {model_path}")
            else:
                # use default model - download if not cached
                try:
                    from .config import (
                        HF_CHARSET_FILENAME,
                        HF_FILENAME,
                        HF_REPO_ID,
                        HF_REVISION,
                    )
                    model_path = str(get_cached_model_path(
                        repo_id=HF_REPO_ID,
                        filename=HF_FILENAME,
                        revision=HF_REVISION,
                    ))
                    # The charset that was published beside these weights. The
                    # pair is what has to agree; a bundled charset from another
                    # revision decodes every index to the wrong character.
                    try:
                        self._downloaded_charset_path = str(get_cached_model_path(
                            repo_id=HF_REPO_ID,
                            filename=HF_CHARSET_FILENAME,
                            revision=HF_REVISION,
                        ))
                    except Exception as e:
                        logger.warning(
                            f"could not fetch the charset beside the model ({e}); "
                            f"falling back to the bundled copy"
                        )
                    logger.info(f"using cached model at {model_path}")
                except Exception as e:
                    logger.error(f"cannot get model, error: {e}")
                    raise ModelNotFoundError(f"cannot download or find model: {e}")
        
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load a production ONNX model from disk."""
        if not os.path.exists(model_path):
            raise ModelNotFoundError(f"Model file not found: {model_path}")
            
        logger.info(f"Loading ONNX model from {model_path}")
        try:
            self.session = ort.InferenceSession(model_path, providers=self.providers)
        except Exception as e:
            raise ModelNotFoundError(f"Failed to initialize ONNX session: {e}")
        
        # Prefer the charset published beside these weights; fall back to the
        # bundled copy, which must match the pinned revision.
        for source in (getattr(self, "_downloaded_charset_path", None), CHARSET_PATH):
            if source and os.path.exists(source):
                try:
                    with open(source, "r", encoding="utf-8") as f:
                        # strip("\n\r"), never strip(). The charset's first
                        # character is U+0020, and a bare .strip() removes it —
                        # which shifts every index by one and silently decodes
                        # all 315 classes to the wrong character. That shipped.
                        self.charset = f.read().strip("\n\r")
                    logger.info(f"loaded charset from {source}")
                    break
                except Exception as e:
                    logger.error(f"cannot read charset file {source}: {e}")

        if self.charset is None:
            raise CharsetNotFoundError(f"charset file not found at {CHARSET_PATH}")

        self._check_contract()
        logger.debug("model loaded and ready.")

    def _check_contract(self) -> None:
        """Refuse a model and charset that cannot belong together.

        CTC reserves index 0 for the blank, so a graph emitting N classes needs
        exactly N-1 characters. When they disagree every decode is wrong and
        nothing raises — the output is fluent-looking text in the wrong
        alphabet. This package shipped that state: a 914-character file against
        a 316-class model, every one of 315 indices mapping to the wrong
        character.
        """
        output = self.session.get_outputs()[0]
        num_classes = output.shape[-1]
        if isinstance(num_classes, int) and num_classes != len(self.charset) + 1:
            raise CharsetNotFoundError(
                f"charset/model mismatch: the model emits {num_classes} classes, "
                f"which needs a {num_classes - 1}-character charset, but the "
                f"charset loaded has {len(self.charset)}. Refusing to decode — "
                f"every index would map to the wrong character."
            )

        # Geometry comes off the graph, not from a constant. A model exported at
        # a different input height silently produces garbage otherwise.
        model_height = self.session.get_inputs()[0].shape[2]
        if isinstance(model_height, int):
            self.input_height = model_height
        else:
            self.input_height = TARGET_HEIGHT

    def predict_line(self, image: Union[str, Image.Image, Path]) -> str:
        """Read the whole image as a single text line, with no segmentation.

        Call this when you already have a line crop. It is the only path that
        cannot split one line into several.
        """
        return self._predict_single_line(self._load(image))

    def predict_page(self, image: Union[str, Image.Image, Path]) -> str:
        """Segment the image into lines, read each, and join them with newlines."""
        img = self._load(image)
        results = []
        for line_img in self._lines_of(img):
            text = self._predict_single_line(line_img)
            if text.strip():
                results.append(text)
        return "\n".join(results)

    def predict(self, image: Union[str, Image.Image, Path]) -> str:
        """Read text from an image. An alias for `predict_page`.

        Until 2.3.0 this dispatched on ``img.height > 100`` — a constant that
        the current model contradicts, since v3.5 takes a 160-pixel input, so a
        line crop at native resolution was tall enough to be treated as a page.
        There is no height that separates the two cases, so the choice is now
        the caller's: `predict_line` for a crop, `predict_page` for a page.
        """
        return self.predict_page(image)

    def predict_with_confidence(self, image: Union[str, Image.Image, Path]) -> Dict[str, Union[str, float]]:
        """Predict text and return alongside a confidence score."""
        img = self._load(image)

        all_text = []
        confs = []

        for line_img in self._lines_of(img):
            text, conf = self._predict_single_line(line_img, return_confidence=True)
            if text.strip():
                all_text.append(text)
                confs.append(conf)

        return {
            'text': "\n".join(all_text),
            'confidence': sum(confs)/len(confs) if confs else 0.0
        }

    def _load(self, image: Union[str, Image.Image, Path]) -> Image.Image:
        """Check the session is up and turn the argument into a grayscale image."""
        if self.session is None:
            raise RuntimeError("Model used before loading. Call load_model() first.")
        try:
            return self._prepare_image(image)
        except ImageLoadError:
            raise
        except Exception as e:
            logger.error(f"Prediction failed during image preparation: {e}")
            raise ImageLoadError(str(e))

    def _lines_of(self, img: Image.Image) -> List[Image.Image]:
        """The line crops to read, falling back to the whole image.

        An empty segmentation used to return "" from predict, which reports
        success for a failure — the caller cannot tell a blank page from a
        segmenter that found nothing in a page full of text.
        """
        lines = self._segment_lines(img)
        if not lines:
            logger.debug("segmentation found no lines; reading the image whole")
            return [img]
        return lines

    # API Aliases and Batch Methods
    def read_text(self, image: Union[str, Image.Image, Path]) -> str:
        return self.predict(image)

    def read_from_folder(self, folder_path: str, extensions: Optional[List[str]] = None) -> Dict[str, str]:
        import glob
        if extensions is None:
            extensions = ['*.png', '*.jpg', '*.jpeg']
        
        results = {}
        for ext in extensions:
            for img_path in glob.glob(os.path.join(folder_path, ext)):
                try:
                    results[os.path.basename(img_path)] = self.predict(img_path)
                except Exception as e:
                    logger.warning(f"Failed to process {img_path}: {e}")
                    results[os.path.basename(img_path)] = ""
        return results

    def predict_batch(self, images: List[Union[str, Image.Image, Path]]) -> List[str]:
        return [self.predict(img) for img in images]

    def _prepare_image(self, image: Union[str, Image.Image, Path]) -> Image.Image:
        """Standardize image to grayscale."""
        if isinstance(image, (str, Path)):
            try:
                image = Image.open(str(image))
            except (FileNotFoundError, UnidentifiedImageError) as e:
                raise ImageLoadError(f"Could not open image file: {e}")
        return normalize_polarity(image.convert("L"))

    def _segment_lines(self, image: Image.Image) -> List[Image.Image]:
        """Split multi-line images using robust LineSegmenter."""
        if not hasattr(self, 'segmenter'):
            self.segmenter = LineSegmenter()
            
        segments = self.segmenter.segment(image)
        return [crop for crop, bbox in segments]

    def _predict_single_line(self, image: Image.Image, return_confidence=False) -> Union[str, tuple]:
        """Core ONNX inference for a single line."""
        target_w = TARGET_WIDTH
        target_h = getattr(self, "input_height", TARGET_HEIGHT)
        
        # Aspect-ratio preserving resize to the height the graph declares.
        # `target_h` is read off the ONNX input shape in __init__, so this is
        # 160 at the pinned revision and follows the model if it is re-exported.
        # It said "v2.0 alignment: height to 128" until 2026-08-27: 128 was the
        # v2 height, and naming it here invited someone to hardcode it back.
        w, h = image.size
        ratio = w / h
        new_w = int(target_h * ratio)
        if new_w > target_w:
            new_w = target_w
            
        pil_img = image.resize((new_w, target_h), Image.Resampling.BILINEAR)
        
        # Pad width to 1024
        new_img = Image.new("L", (target_w, target_h), 255)
        new_img.paste(pil_img, (0, 0))
        
        # Normalize to [-1.0, 1.0]
        img_arr = np.array(new_img).astype(np.float32)
        img_norm = (img_arr - IMAGE_NORM_MEAN) / IMAGE_NORM_STD
        
        # Add channel and batch dimensions: [1, 1, 160, 1024] at the pinned revision
        tensor = np.expand_dims(img_norm, axis=(0, 1))
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        preds = self.session.run(None, {input_name: tensor})[0]
        
        # Decoding
        if return_confidence:
            probs = self._softmax(preds, axis=2)[0]
            indices = np.argmax(probs, axis=1)
            text = self._decode(indices)
            conf = np.mean(np.max(probs, axis=1))
            return text, conf
        else:
            # Efficient greedy decoding directly from logits
            indices = np.argmax(preds[0], axis=1)
            return self._decode(indices)

    def _softmax(self, x, axis=None):
        """Native numpy softmax implementation."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def _decode(self, indices: np.ndarray) -> str:
        """Greedy CTC decoding."""
        text = []
        prev_idx = 0
        for idx in indices:
            if idx != 0 and idx != prev_idx:
                if 0 < idx <= len(self.charset):
                    text.append(self.charset[idx-1])
            prev_idx = idx
        return "".join(text)
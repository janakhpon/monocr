import os
import onnxruntime as ort
import numpy as np
import logging
from PIL import Image, UnidentifiedImageError
from typing import List, Optional, Union, Dict
from pathlib import Path

from .segmenter import LineSegmenter
from .config import (
    TARGET_WIDTH, TARGET_HEIGHT, 
    CHARSET_PATH
)
from .exceptions import (
    ModelNotFoundError, CharsetNotFoundError, ImageLoadError
)
from .download import get_cached_model_path

logger = logging.getLogger(__name__)

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
            model_path: Path to the .onnx model file. If None, downloads default v2.0 model.
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
                    from .config import HF_REPO_ID, HF_FILENAME
                    model_path = str(get_cached_model_path(
                        repo_id=HF_REPO_ID,
                        filename=HF_FILENAME
                    ))
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
        
        # Load charset from sidecar file (standard for v2.0)
        if os.path.exists(CHARSET_PATH):
            try:
                with open(CHARSET_PATH, "r", encoding="utf-8") as f:
                    self.charset = f.read().strip()
                logger.info(f"loaded charset from {CHARSET_PATH}")
            except Exception as e:
                logger.error(f"cannot read charset file: {e}")
        
        if self.charset is None:
            raise CharsetNotFoundError(f"charset file not found at {CHARSET_PATH}")

        logger.debug("model loaded and ready.")

    def predict(self, image: Union[str, Image.Image, Path]) -> str:
        """Extract text from an image. Handles single and multi-line images."""
        if self.session is None:
             raise RuntimeError("Model used before loading. Call load_model() first.")

        try:
            img = self._prepare_image(image)
        except Exception as e:
            logger.error(f"Prediction failed during image preparation: {e}")
            raise ImageLoadError(str(e))
        
        # Simple vertical check: if image is tall, try segmentation
        if img.height > 100:
            lines = self._segment_lines(img)
        else:
            lines = [img]
        
        results = []
        for line_img in lines:
            text = self._predict_single_line(line_img)
            if text.strip():
                results.append(text)
                
        return "\n".join(results)

    def predict_with_confidence(self, image: Union[str, Image.Image, Path]) -> Dict[str, Union[str, float]]:
        """Predict text and return alongside a confidence score."""
        if self.session is None:
             raise RuntimeError("Model used before loading.")

        try:
            img = self._prepare_image(image)
        except Exception as e:
             raise ImageLoadError(str(e))

        if img.height > 100:
            lines = self._segment_lines(img)
        else:
            lines = [img]
        
        all_text = []
        confs = []
        
        for line_img in lines:
            text, conf = self._predict_single_line(line_img, return_confidence=True)
            if text.strip():
                all_text.append(text)
                confs.append(conf)
                
        return {
            'text': "\n".join(all_text), 
            'confidence': sum(confs)/len(confs) if confs else 0.0
        }

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
        return image.convert("L")

    def _segment_lines(self, image: Image.Image) -> List[Image.Image]:
        """Split multi-line images using robust LineSegmenter."""
        if not hasattr(self, 'segmenter'):
            self.segmenter = LineSegmenter()
            
        segments = self.segmenter.segment(image)
        return [crop for crop, bbox in segments]

    def _predict_single_line(self, image: Image.Image, return_confidence=False) -> Union[str, tuple]:
        """Core ONNX inference for a single line."""
        target_w, target_h = TARGET_WIDTH, TARGET_HEIGHT
        
        # Aspect-ratio preserving resize (v2.0 alignment: height to 128)
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
        img_norm = (img_arr / 127.5) - 1.0
        
        # Add channel and batch dimensions: [1, 1, 128, 1024]
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
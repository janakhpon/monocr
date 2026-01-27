#!/usr/bin/env python3
import os
import torch
import numpy as np
import logging
from PIL import Image, UnidentifiedImageError
from typing import List, Optional, Union, Dict
from pathlib import Path

from .model import MonOCRModel
from .segmenter import LineSegmenter
from .config import (
    TARGET_WIDTH, TARGET_HEIGHT, 
    IMAGE_NORM_MEAN, IMAGE_NORM_STD,
    PROJECTION_THRESHOLD, MIN_LINE_GAP, BINARY_THRESHOLD,
    CHARSET_PATH, DEFAULT_MODEL_PATH
)
from .exceptions import (
    ModelNotFoundError, CharsetNotFoundError, ImageLoadError
)

logger = logging.getLogger(__name__)

class MonOCR:
    """
    Mon OCR Inference Class.
    Supports single-line and multi-line (paragraph) Mon text recognition.
    """
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = "crnn", device: str = None):
        """
        Initialize Mon OCR.
        
        Args:
            model_path: Path to the .pt model file. If None, tries to load bundled default model.
            model_type: Type of model (defaults to 'crnn').
            device: Computing device ('cuda', 'cpu').
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model_type = model_type.lower()
        self.model = None
        self.charset = None
        
        if model_path is None:
            # Fallback to default bundled model
            if os.path.exists(DEFAULT_MODEL_PATH):
                logger.info(f"No model path provided. Loading default model from {DEFAULT_MODEL_PATH}")
                model_path = str(DEFAULT_MODEL_PATH)
            else:
                logger.warning(f"Default model not found at {DEFAULT_MODEL_PATH}. Initialized without model.")
        
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load a trained model from disk."""
        if not os.path.exists(model_path):
            raise ModelNotFoundError(f"Model file not found: {model_path}")
            
        logger.info(f"Loading model from {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load checkpoint: {e}")
        
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
            self.charset = checkpoint.get('charset') or checkpoint.get('hyper_parameters', {}).get('charset')
        else:
            state_dict = checkpoint
            self.charset = None

        if self.charset is None:
            logger.warning("Charset not found in checkpoint. Attempting fallback.")
            if os.path.exists(CHARSET_PATH):
                try:
                    with open(CHARSET_PATH, "r", encoding="utf-8") as f:
                        self.charset = f.read().strip()
                    logger.info(f"Loaded charset from {CHARSET_PATH}")
                except Exception as e:
                    logger.error(f"Failed to read charset file: {e}")
            
        if self.charset is None:
            raise CharsetNotFoundError("Model checkpoint is missing charset information and valid_chars.txt not found.")

        # Versatile size handling
        num_classes = len(self.charset) + 1
        if 'fc.weight' in state_dict:
            ckpt_classes = state_dict['fc.weight'].size(0)
            if ckpt_classes != num_classes:
                logger.warning(f"Model checkpoint has {ckpt_classes} classes, but charset has {len(self.charset)} (+1={num_classes}).")
                if ckpt_classes < num_classes:
                     logger.warning(f"Adjusting charset to match checkpoint size. {num_classes - ckpt_classes} characters will be ignored.")
                     self.charset = self.charset[:ckpt_classes-1]
                     num_classes = ckpt_classes
                else:
                     logger.warning("Checkpoint has MORE classes than charset. Unknown characters will be ignored during decoding.")
                     num_classes = ckpt_classes

        self.model = MonOCRModel(num_classes=num_classes)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        logger.debug("Model loaded and ready.")

    def predict(self, image: Union[str, Image.Image, Path]) -> str:
        """Extract text from an image. Handles single and multi-line images."""
        if self.model is None:
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
        if self.model is None:
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
            
        # Segmenter returns list of (crop, bbox)
        segments = self.segmenter.segment(image)
        return [crop for crop, bbox in segments]

    def _predict_single_line(self, image: Image.Image, return_confidence=False) -> Union[str, tuple]:
        """Core CRNN inference for a single line."""
        target_w, target_h = TARGET_WIDTH, TARGET_HEIGHT
        
        # Aspect-ratio preserving resize
        w, h = image.size
        ratio = w / h
        new_w = int(target_h * ratio)
        
        # Resize
        if new_w > target_w:
            new_w = target_w
        pil_img = image.resize((new_w, target_h), Image.Resampling.BILINEAR)
        
        # Create canvas of target size (fixed width) and paste
        # Training code uses 255 (white) background
        new_img = Image.new("L", (target_w, target_h), 255)
        new_img.paste(pil_img, (0, 0))
        pil_img = new_img
        
        # Normalize to [-1, 1] as per training logic (utils.resize_and_pad)
        # canvas is 0..255
        img_arr = np.array(pil_img).astype(np.float32)
        img_norm = img_arr / 127.5 - 1.0

        tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.model(tensor)
            probs = preds.softmax(2).squeeze(0)
        
        best_path = probs.argmax(1)
        text = self._decode(best_path)
        
        if return_confidence:
            conf = probs.max(1).values.mean().item()
            return text, conf
        
        return text

    def _decode(self, indices: torch.Tensor) -> str:
        """Greedy CTC decoding."""
        text = []
        prev_idx = 0
        for idx in indices:
            val = idx.item()
            if val != 0 and val != prev_idx:
                if 0 < val <= len(self.charset):
                    text.append(self.charset[val-1])
            prev_idx = val
        return "".join(text)
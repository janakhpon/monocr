#!/usr/bin/env python3
import os
import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Union, Dict
from .model import MonOCRModel

# Optional TrOCR
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False

class MonOCR:
    """
    Mon OCR Inference Class for CRNN and TrOCR models.
    Supports both single image and batch processing.
    """
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = "crnn", device: str = None):
        """
        Initialize Mon OCR.
        
        Args:
            model_path: Path to the model file. If None, model must be loaded manually.
            model_type: Type of model ('crnn' or 'trocr').
            device: Device to use for inference ('cuda', 'cpu', etc.).
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type.lower()
        self.model = None
        self.processor = None
        self.charset = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the .pt model file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        if self.model_type == "crnn":
            self._load_crnn(model_path)
        elif self.model_type == "trocr":
            self._load_trocr(model_path)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _load_crnn(self, model_path: str):
        """Internal method to load CRNN architecture."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Resolve state_dict and charset
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
            self.charset = checkpoint.get('charset')
        else:
            state_dict = checkpoint
            self.charset = None

        if self.charset is None:
            # Legacy/Fallback support
            self.model = MonOCRModel(num_classes=256)
            try:
                self.model.load_state_dict(state_dict)
                self.model.to(self.device).eval()
                return
            except Exception as e:
                raise ValueError(f"Unable to load legacy model: {e}")

        # Initialize and load modern model
        self.model = MonOCRModel(num_classes=len(self.charset) + 1)
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            self.model.load_state_dict(state_dict, strict=False)
            
        self.model.to(self.device).eval()

    def _load_trocr(self, model_path: str):
        """Internal method to load TrOCR architecture."""
        if not TROCR_AVAILABLE:
            raise ImportError("TrOCR requires 'transformers'. Install with: pip install transformers")
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model.to(self.device).eval()

    def predict(self, image: Union[str, Image.Image]) -> str:
        """
        Predict text from an image.
        
        Args:
            image: Path to image or PIL Image object.
            
        Returns:
            Extracted text as a string.
        """
        img = self._prepare_image(image)
            
        if self.model_type == "crnn":
            return self._predict_crnn(img)
        elif self.model_type == "trocr":
            return self._predict_trocr(img)
        return ""

    def predict_with_confidence(self, image: Union[str, Image.Image]) -> Dict[str, Union[str, float]]:
        """
        Predict text and calculate a heuristic confidence score.
        """
        img = self._prepare_image(image)
        text = self.predict(img)
        
        # Simple heuristic confidence based on image properties and text
        if not text:
            confidence = 0.0
        else:
            w, h = img.size
            confidence = min(1.0, (len(text) * 100) / (w * h))
            
        return {'text': text, 'confidence': confidence}

    def read_text(self, image: Union[str, Image.Image]) -> str:
        """Simplified alias for predict()"""
        return self.predict(image)

    def read_from_folder(self, folder_path: str, extensions: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Process all images in a folder and return a dictionary of results.
        """
        import glob
        if extensions is None:
            extensions = ['*.png', '*.jpg', '*.jpeg']
        
        results = {}
        for ext in extensions:
            for img_path in glob.glob(os.path.join(folder_path, ext)):
                results[os.path.basename(img_path)] = self.predict(img_path)
        return results

    def _prepare_image(self, image: Union[str, Image.Image]) -> Image.Image:
        """Internal helper to ensure image is a grayscale PIL Image."""
        if isinstance(image, str):
            return Image.open(image).convert("L")
        return image.convert("L")

    def _predict_crnn(self, image: Image.Image) -> str:
        """Inference logic for CRNN architecture."""
        if self.model is None:
            raise ValueError("Model not loaded")
            
        w, h = image.size
        target_h = 64
        target_w = int(w * (target_h / h))
        
        image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
        img_arr = np.array(image).astype(np.float32) / 127.5 - 1.0
        
        tensor = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.model(tensor)
            preds_log = preds.log_softmax(2)
            
        return self._decode(preds_log)

    def _predict_trocr(self, image: Image.Image) -> str:
        """Inference logic for TrOCR architecture."""
        pixel_values = self.processor(image.convert("RGB"), return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            ids = self.model.generate(pixel_values)
            return self.processor.batch_decode(ids, skip_special_tokens=True)[0]

    def _decode(self, preds):
        """Greedy CTC decoding for CRNN."""
        pred_indices = preds.argmax(dim=2).squeeze(0)
        
        text = []
        prev_idx = 0
        idx2char = {i + 1: c for i, c in enumerate(self.charset)} if self.charset else {}
        
        for idx in pred_indices:
            idx = idx.item()
            if idx != 0 and idx != prev_idx:
                text.append(idx2char.get(idx, f"[{idx}]" if not self.charset else ""))
            prev_idx = idx
            
        return "".join(text)

    def predict_batch(self, images: List[Union[str, Image.Image]]) -> List[str]:
        """Process a list of images."""
        return [self.predict(img) for img in images]
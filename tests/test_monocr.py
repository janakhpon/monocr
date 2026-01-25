#!/usr/bin/env python3
"""
Tests for MonOCR 0.1.3
Tests the consolidated CRNN-based OCR system
"""

import unittest
from monocr import MonOCR, read_text
from monocr.exceptions import MonOCRError, ModelNotFoundError
from PIL import Image
import numpy as np
import tempfile
import os


class TestMonOCR(unittest.TestCase):
    """Test core MonOCR functionality"""
    
    def test_initialization_defaults(self):
        """Test that MonOCR initializes with correct defaults"""
        ocr = MonOCR()
        self.assertEqual(ocr.model_type, "crnn")
        self.assertIn(ocr.device, ["cuda", "cpu"])
        
    def test_initialization_with_model_type(self):
        """Test initialization with explicit model type"""
        ocr = MonOCR(model_type="crnn")
        self.assertEqual(ocr.model_type, "crnn")
    
    def test_image_preparation(self):
        """Test internal image preparation helper"""
        ocr = MonOCR()
        
        # Test with PIL Image
        img = Image.fromarray(np.zeros((100, 200), dtype=np.uint8))
        prepared = ocr._prepare_image(img)
        self.assertEqual(prepared.mode, "L")
        
        # Test with image path
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name)
            prepared = ocr._prepare_image(tmp.name)
            self.assertEqual(prepared.mode, "L")
            os.unlink(tmp.name)
    
    def test_api_methods_exist(self):
        """Verify all expected API methods are present"""
        ocr = MonOCR()
        
        # Core methods
        self.assertTrue(hasattr(ocr, 'predict'))
        self.assertTrue(hasattr(ocr, 'predict_with_confidence'))
        
        # Convenience aliases
        self.assertTrue(hasattr(ocr, 'read_text'))
        self.assertTrue(hasattr(ocr, 'read_from_folder'))
        self.assertTrue(hasattr(ocr, 'predict_batch'))

    def test_predict_without_model_raises_error(self):
        """Test that predicting without a model raises RuntimeError"""
        ocr = MonOCR()
        ocr.model = None # Force model to None to simulate unloaded state
        with self.assertRaises(RuntimeError):
            ocr.predict("dummy_path")

    def test_invalid_model_path(self):
        """Test that invalid model path raises ModelNotFoundError"""
        with self.assertRaises(ModelNotFoundError):
            MonOCR(model_path="non_existent_model.pt")


class TestPackageAPI(unittest.TestCase):
    """Test package-level convenience functions"""
    
    def test_read_text_function_exists(self):
        """Verify read_text is available at package level"""
        self.assertTrue(callable(read_text))


class TestArchitecture(unittest.TestCase):
    """Test CRNN architecture components"""
    
    def test_model_import(self):
        """Verify model components can be imported"""
        from monocr.model import MonOCRModel, ResNetFeatureExtractor
        self.assertTrue(callable(MonOCRModel))
        self.assertTrue(callable(ResNetFeatureExtractor))


if __name__ == "__main__":
    unittest.main()

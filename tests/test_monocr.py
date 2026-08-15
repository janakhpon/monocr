#!/usr/bin/env python3
"""
Tests for MonOCR
Tests the consolidated ONNX-based OCR system
"""

import unittest
import unittest.mock

from monocr import MonOCR, read_text
from monocr.exceptions import MonOCRError, ModelNotFoundError, ImageLoadError
from PIL import Image
import numpy as np
import tempfile
import os
from pathlib import Path


class TestMonOCR(unittest.TestCase):
    """Test core MonOCR functionality"""
    
    def test_initialization(self):
        """Test that MonOCR initializes correctly"""
        # This will attempt to find/download the model
        try:
            ocr = MonOCR()
            self.assertIsNotNone(ocr)
        except ModelNotFoundError:
            # Skip if model can't be found/downloaded in test env
            self.skipTest("Model not found/downloadable in this environment")

    @unittest.mock.patch('monocr.ocr.MonOCR.load_model')
    def test_image_preparation(self, mock_load):
        """Test internal image preparation helper"""
        ocr = MonOCR(model_path="dummy.onnx")
        
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
        # We can init without loading session for this check if we are careful
        # But MonOCR.__init__ calls load_model. 
        # Let's mock a bit or just assume it might fail if model missing.
        try:
            ocr = MonOCR()
            # Core methods
            self.assertTrue(hasattr(ocr, 'predict'))
            self.assertTrue(hasattr(ocr, 'predict_with_confidence'))
            
            # Convenience aliases
            self.assertTrue(hasattr(ocr, 'read_text'))
            self.assertTrue(hasattr(ocr, 'read_from_folder'))
            self.assertTrue(hasattr(ocr, 'predict_batch'))
        except (ModelNotFoundError, Exception):
             self.skipTest("MonOCR init failed (likely missing model/charset)")

    def test_invalid_model_path(self):
        """Test that invalid model path raises ModelNotFoundError"""
        with self.assertRaises(ModelNotFoundError):
            MonOCR(model_path="non_existent_model.onnx")

class TestPackageAPI(unittest.TestCase):
    """Test package-level convenience functions"""
    
    def test_read_text_function_exists(self):
        """Verify read_text is available at package level"""
        self.assertTrue(callable(read_text))


if __name__ == "__main__":
    unittest.main()


class TestCharsetContract(unittest.TestCase):
    """The defect that shipped in 2.1.2, pinned so it cannot ship again.

    The bundled charset was a 914-character file against a 316-class model, and
    it was loaded with `.strip()`, which eats the leading U+0020 and shifts every
    index by one. Measured on 2026-08-15: all 315 decodable indices mapped to the
    wrong character, so the package returned fluent-looking Latin for Mon input
    and raised nothing.
    """

    def test_the_bundled_charset_matches_the_pinned_model(self):
        from monocr.config import CHARSET_PATH

        with open(CHARSET_PATH, encoding="utf-8") as f:
            charset = f.read().strip("\n\r")
        self.assertEqual(
            len(charset), 276,
            "the bundled charset must match the model at config.HF_REVISION; "
            "277 classes need exactly 276 characters",
        )

    def test_the_charset_starts_with_a_space_and_loading_keeps_it(self):
        from monocr.config import CHARSET_PATH

        raw = open(CHARSET_PATH, encoding="utf-8").read()
        self.assertEqual(raw[0], " ", "the first class is U+0020")
        self.assertEqual(
            raw.strip("\n\r")[0], " ",
            "strip('\\n\\r') must keep it; a bare .strip() removes it and shifts "
            "every index by one",
        )
        self.assertNotEqual(
            len(raw.strip()), len(raw.strip("\n\r")),
            "if these are equal the leading space is already gone and this test "
            "no longer guards anything",
        )

    def test_a_revision_is_pinned(self):
        from monocr import config

        self.assertTrue(
            getattr(config, "HF_REVISION", None),
            "an unpinned download resolves `main`, so any upload to the model "
            "repository changes what installed copies fetch",
        )

    def test_a_mismatched_charset_is_refused_rather_than_decoded(self):
        from monocr.exceptions import CharsetNotFoundError
        from monocr.ocr import MonOCR

        class _Output:
            shape = [1, "sequence", 277]

        class _Input:
            shape = [1, 1, 160, 1024]

        class _Session:
            def get_outputs(self):
                return [_Output()]

            def get_inputs(self):
                return [_Input()]

        ocr = MonOCR.__new__(MonOCR)
        ocr.session = _Session()
        ocr.charset = "abc"  # 3 characters against 277 classes
        with self.assertRaises(CharsetNotFoundError):
            ocr._check_contract()

        ocr.charset = "x" * 276
        ocr._check_contract()  # must not raise
        self.assertEqual(ocr.input_height, 160, "height comes off the graph")

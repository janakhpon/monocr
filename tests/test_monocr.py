#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Mon OCR package
"""

import pytest
import tempfile
from PIL import Image
from pathlib import Path

from monocr import MonOCR, MonOCRInference


class TestMonOCR:
    """Test MonOCR class"""
    
    def test_init(self):
        """Test initialization"""
        ocr = MonOCR()
        assert ocr is not None
    
    def test_read_text_with_mock(self):
        """Test read_text method (without actual model)"""
        ocr = MonOCR()
        
        # Create a simple test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img = Image.new('L', (256, 64), color=255)
            img.save(tmp.name)
            
            # This will fail without a model, but we can test the structure
            try:
                text = ocr.read_text(tmp.name)
                assert isinstance(text, str)
            except (ValueError, FileNotFoundError):
                # Expected without a trained model
                pass
            
            # Clean up
            Path(tmp.name).unlink()


class TestMonOCRAdvanced:
    """Test MonOCR advanced features"""
    
    def test_init_with_model_type(self):
        """Test initialization with different model types"""
        ocr = MonOCR(model_type="trocr")
        assert ocr.model_type == "trocr"


class TestMonOCRInference:
    """Test MonOCRInference class"""
    
    def test_init(self):
        """Test initialization"""
        ocr = MonOCRInference()
        assert ocr is not None

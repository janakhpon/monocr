"""What actually reaches the model.

The tensor is the contract between this package and the graph: an image scaled
to the graph's own input height, left-aligned on a white canvas, normalised to
[-1, 1]. Every one of those four is a place where a wrong sign or a wrong fill
colour produces plausible-looking output rather than an error.
"""

import numpy as np
from PIL import Image

from monocr.config import TARGET_HEIGHT, TARGET_WIDTH


def tensor_for(ocr, image):
    """Run one line through and hand back what the session was fed."""
    ocr._predict_single_line(image)
    return ocr.fake_session.last_input


def test_the_tensor_has_a_batch_and_a_channel_axis(make_ocr):
    ocr = make_ocr()
    tensor = tensor_for(ocr, Image.new("L", (200, 100), 255))
    assert tensor.shape == (1, 1, 160, TARGET_WIDTH)
    assert tensor.dtype == np.float32


def test_white_becomes_plus_one_and_black_becomes_minus_one(make_ocr):
    ocr = make_ocr()
    white = tensor_for(ocr, Image.new("L", (200, 100), 255))
    assert np.allclose(white, 1.0)

    black = tensor_for(ocr, Image.new("L", (1024, 160), 0))
    assert np.allclose(black, -1.0)


def test_the_padding_is_white_not_black(make_ocr):
    """Black padding reads as ink. The model would see a solid bar after the
    text and has no training example that looks like that."""
    ocr = make_ocr()
    tensor = tensor_for(ocr, Image.new("L", (100, 100), 0))

    scaled_w = int(160 * (100 / 100))
    assert np.allclose(tensor[0, 0, :, :scaled_w], -1.0), "the image itself is black"
    assert np.allclose(tensor[0, 0, :, scaled_w:], 1.0), "the padding must be white"


def test_the_height_comes_off_the_graph_not_the_constant(make_ocr):
    """A model exported at another height silently produces garbage otherwise."""
    ocr = make_ocr(height=96)
    assert ocr.input_height == 96
    assert tensor_for(ocr, Image.new("L", (200, 100), 255)).shape[2] == 96


def test_a_dynamic_height_axis_falls_back_to_the_constant(make_ocr):
    """onnxruntime reports a dynamic axis as a string, which is not a height."""
    ocr = make_ocr(height="height")
    assert ocr.input_height == TARGET_HEIGHT


def test_the_aspect_ratio_is_preserved_within_the_window(make_ocr):
    """A 4:1 line at height 160 must occupy 640 columns, not be stretched to fill."""
    ocr = make_ocr()
    tensor = tensor_for(ocr, Image.new("L", (400, 100), 0))

    ink_columns = np.where((tensor[0, 0] < 0).any(axis=0))[0]
    assert ink_columns.min() == 0
    assert ink_columns.max() == 639, "160 * (400/100) = 640 columns of image"


def test_a_line_too_wide_for_the_window_is_clamped_not_overflowed(make_ocr):
    """Squeezing is wrong, but writing past the canvas would raise. The clamp is
    what keeps a 3000px line producing a tensor at all; monocr-onnx tiles
    instead, and that difference is recorded in its parity notes."""
    ocr = make_ocr()
    tensor = tensor_for(ocr, Image.new("L", (3000, 100), 0))
    assert tensor.shape == (1, 1, 160, TARGET_WIDTH)
    assert (tensor[0, 0] < 0).all(), "the whole window is covered by the squeezed line"


def test_a_line_narrower_than_the_window_is_not_stretched(make_ocr):
    ocr = make_ocr()
    tensor = tensor_for(ocr, Image.new("L", (10, 100), 0))
    ink_columns = np.where((tensor[0, 0] < 0).any(axis=0))[0]
    assert ink_columns.max() == 15, "160 * (10/100) = 16 columns"

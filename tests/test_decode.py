"""Greedy CTC decoding.

`_decode` had no tests at all, and it is exactly where the 2.1.2 defect lived:
the charset it indexes into was the wrong file, loaded with a `.strip()` that
ate the leading U+0020, so all 315 decodable indices returned the wrong
character and nothing raised. The output was fluent-looking Latin for Mon input.

Index 0 is the CTC blank and is never a character; index i means charset[i-1].
"""

import numpy as np
import pytest


def decoder(make_ocr, charset):
    """A MonOCR whose charset is exactly `charset`, wired to a fake graph."""
    return make_ocr(charset=charset, num_classes=len(charset) + 1)


def test_blanks_are_dropped(make_ocr):
    ocr = decoder(make_ocr, " abc")
    assert ocr._decode(np.array([0, 0, 0])) == ""


def test_an_empty_sequence_decodes_to_nothing(make_ocr):
    ocr = decoder(make_ocr, " abc")
    assert ocr._decode(np.array([], dtype=int)) == ""


def test_indices_map_one_behind_the_charset(make_ocr):
    """1 is the first character, not the second — the off-by-one that shipped."""
    ocr = decoder(make_ocr, " abc")
    assert ocr._decode(np.array([1])) == " "
    assert ocr._decode(np.array([2])) == "a"
    assert ocr._decode(np.array([4])) == "c"


def test_every_index_in_range_maps_to_its_own_character(make_ocr):
    """The whole alphabet, not a sample.

    A charset off by one position, or off by one file, still decodes *something*
    for every index. Only checking all of them catches it.
    """
    charset = " abcdefghij"
    ocr = decoder(make_ocr, charset)
    indices = np.arange(1, len(charset) + 1)
    # Interleave blanks so no two adjacent indices collapse into one another.
    spaced = np.empty(len(indices) * 2, dtype=int)
    spaced[0::2] = indices
    spaced[1::2] = 0
    assert ocr._decode(spaced) == charset


def test_adjacent_repeats_collapse(make_ocr):
    ocr = decoder(make_ocr, " abc")
    assert ocr._decode(np.array([2, 2, 2])) == "a"


def test_a_repeat_split_by_a_blank_is_two_characters(make_ocr):
    """The reason CTC has a blank at all. Without this, "aa" can never be read."""
    ocr = decoder(make_ocr, " abc")
    assert ocr._decode(np.array([2, 0, 2])) == "aa"


def test_a_run_broken_by_another_character_is_not_collapsed(make_ocr):
    ocr = decoder(make_ocr, " abc")
    assert ocr._decode(np.array([2, 3, 2])) == "aba"


def test_an_index_past_the_charset_is_skipped_not_wrapped(make_ocr):
    """A graph and charset that disagree are refused at load, but if one ever
    slipped through, silently indexing from the end would be worse than a gap."""
    ocr = decoder(make_ocr, " abc")
    assert ocr._decode(np.array([2, 99, 3])) == "ab"


def test_a_negative_index_is_skipped(make_ocr):
    """argmax cannot produce one, but `charset[-1]` would return the last
    character rather than fail, so the guard is worth pinning."""
    ocr = decoder(make_ocr, " abc")
    assert ocr._decode(np.array([-1, 2])) == "a"


def test_the_bundled_charset_decodes_its_own_last_index(make_ocr, bundled_charset):
    """Against the real 276-character charset, not a toy one."""
    ocr = make_ocr()
    assert len(bundled_charset) == 276
    assert ocr._decode(np.array([276])) == bundled_charset[-1]
    assert ocr._decode(np.array([1])) == " "
    assert ocr._decode(np.array([277])) == "", "277 is out of range for 276 characters"


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_decoding_does_not_depend_on_the_index_dtype(make_ocr, dtype):
    """argmax returns int64 on most platforms and int32 on some."""
    ocr = decoder(make_ocr, " abc")
    assert ocr._decode(np.array([2, 0, 3], dtype=dtype)) == "ab"

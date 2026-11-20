# Copyright (c) 2026 moonlab
# SPDX-License-Identifier: MIT
"""Contract tests for the binary mask produced by the SAM 3 engine."""

import numpy as np

from sam3_ros.config import EngineConfig
from sam3_ros.sam3_engine import Sam3Engine


def _engine() -> Sam3Engine:
    # _extract_result is pure NumPy; no model initialization is required.
    return Sam3Engine(EngineConfig())


def test_mask_is_same_size_mono8_binary():
    height, width = 4, 6
    masks = np.zeros((2, 1, height, width), dtype=bool)
    masks[0, 0, 1:3, 2:4] = True
    masks[1, 0, 0, 0] = True
    state = {"masks": masks}

    result = _engine()._extract_result(state, width, height)

    assert result.mask.shape == (height, width)
    assert result.mask.dtype == np.uint8
    assert set(np.unique(result.mask)).issubset({0, 1})
    assert result.num_instances == 2
    # Union of both instance masks.
    assert result.mask[1, 2] == 1
    assert result.mask[0, 0] == 1
    assert result.mask[3, 5] == 0


def test_mask_is_resized_to_the_input_if_needed():
    height, width = 5, 9
    state = {"masks": np.ones((1, 1, 3, 3), dtype=bool)}

    result = _engine()._extract_result(state, width, height)

    assert result.mask.shape == (height, width)
    assert result.mask.dtype == np.uint8
    assert result.mask.all()


def test_no_instances_still_return_full_size_mask():
    height, width = 3, 7
    state = {"masks": np.zeros((0, 1, height, width), dtype=bool)}

    result = _engine()._extract_result(state, width, height)

    assert result.mask.shape == (height, width)
    assert result.mask.dtype == np.uint8
    assert not result.mask.any()
    assert result.num_instances == 0


def test_missing_masks_return_background_only():
    height, width = 2, 4

    result = _engine()._extract_result({}, width, height)

    assert result.mask.shape == (height, width)
    assert result.mask.dtype == np.uint8
    assert not result.mask.any()
    assert result.num_instances == 0

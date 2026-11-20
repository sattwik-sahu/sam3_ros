# Copyright (c) 2026 moonlab
# SPDX-License-Identifier: MIT
"""ROS image <-> NumPy helpers used by both SAM 3 nodes."""

from __future__ import annotations

import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class ImageConversionError(RuntimeError):
    """Raised when an incoming image cannot be converted to RGB."""


class ImageConverter:
    """Small wrapper around :class:`cv_bridge.CvBridge`.

    Keeping the bridge in one place makes it trivial to swap the conversion
    backend or to add encodings later.  Incoming images are normalised to
    contiguous ``uint8`` RGB because that is what SAM 3 (via Pillow) expects.
    """

    #: Encodings accepted by :meth:`to_rgb`.
    SUPPORTED_ENCODINGS = frozenset(
        {
            "rgb8",
            "bgr8",
            "rgba8",
            "bgra8",
            "mono8",
            "mono16",
            "8UC3",
            "8UC4",
            "8UC1",
            "bayer_rggb8",
            "bayer_bggr8",
            "bayer_gbrg8",
            "bayer_grbg8",
        }
    )

    def __init__(self) -> None:
        self._bridge = CvBridge()

    def to_rgb(self, msg: Image) -> np.ndarray:
        """Convert a ROS image message to a contiguous ``(H, W, 3)`` RGB array."""
        encoding = msg.encoding.lower()
        if encoding not in self.SUPPORTED_ENCODINGS:
            raise ImageConversionError(
                f"Unsupported image encoding {msg.encoding!r}; supported: "
                f"{sorted(self.SUPPORTED_ENCODINGS)}"
            )
        try:
            rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as exc:  # cv_bridge raises several exception types
            raise ImageConversionError(
                f"Failed to convert image ({msg.encoding}, "
                f"{msg.width}x{msg.height}): {exc}"
            ) from exc
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    def to_mono8(self, mask: np.ndarray, header) -> Image:
        """Convert a ``{0, 1}`` mask to a ``mono8`` image with ``header``."""
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        msg = self._bridge.cv2_to_imgmsg(np.ascontiguousarray(mask), encoding="mono8")
        msg.header = header
        return msg

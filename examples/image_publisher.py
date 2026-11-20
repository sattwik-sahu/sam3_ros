#!/usr/bin/env python3
# Copyright (c) 2026 moonlab
# SPDX-License-Identifier: MIT
"""Publish a still image on a ROS 2 topic to smoke-test the streaming node.

Usage:
    python examples/image_publisher.py \
        --image frame.jpg --topic /camera/color/image_raw --rate-hz 5

Requires a sourced ROS 2 environment (``pixi run -e sam3 python ...``).
"""

from __future__ import annotations

import argparse
import sys

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class StillImagePublisher(Node):
    def __init__(self, image_rgb, topic: str, rate_hz: float) -> None:
        super().__init__("sam3_image_publisher")
        self._message = CvBridge().cv2_to_imgmsg(image_rgb, encoding="rgb8")
        self._publisher = self.create_publisher(Image, topic, 10)
        period = 1.0 / rate_hz if rate_hz > 0.0 else 0.0
        self._timer = self.create_timer(period, self._publish) if period else None
        if self._timer is None:
            self._publish()
        self.get_logger().info(f"Publishing {topic} at {rate_hz:.1f} Hz")

    def _publish(self) -> None:
        self._message.header.stamp = self.get_clock().now().to_msg()
        self._message.header.frame_id = "sam3_test"
        self._publisher.publish(self._message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="input image path")
    parser.add_argument(
        "--topic", default="/camera/color/image_raw", help="output image topic"
    )
    parser.add_argument(
        "--rate-hz", type=float, default=5.0, help="publish rate; 0 = once"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"Could not read image: {args.image}", file=sys.stderr)
        return 1
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    rclpy.init()
    node = StillImagePublisher(rgb, args.topic, args.rate_hz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

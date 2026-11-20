#!/usr/bin/env python3
# Copyright (c) 2026 moonlab
# SPDX-License-Identifier: MIT
"""Call the SAM 3 SegmentImage service with a single image file.

Usage:
    python examples/segment_client.py \
        --image frame.jpg --prompt "red mug" --service /segment

Requires a sourced ROS 2 environment (``pixi run -e sam3 python ...``).
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import rclpy
from cv_bridge import CvBridge

from sam3_msgs.srv import SegmentImage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="input image path")
    parser.add_argument("--prompt", required=True, help="text concept to segment")
    parser.add_argument("--service", default="/segment", help="service name")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=-1.0,
        help="per-request threshold in [0, 1]; negative uses the node default",
    )
    parser.add_argument(
        "--output",
        default="",
        help="optional path to write the mask as an 8-bit PNG",
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="service wait/call timeout"
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
    node = rclpy.create_node("sam3_segment_client")
    bridge = CvBridge()
    client = node.create_client(SegmentImage, args.service)
    try:
        if not client.wait_for_service(timeout_sec=args.timeout):
            print(f"Service {args.service} is not available", file=sys.stderr)
            return 1

        request = SegmentImage.Request()
        request.image = bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
        request.text_prompt = args.prompt
        request.confidence_threshold = args.confidence_threshold

        started = time.perf_counter()
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=args.timeout)
        if not future.done() or future.result() is None:
            print("Service call did not complete", file=sys.stderr)
            return 1
        response: SegmentImage.Response = future.result()
        round_trip_ms = (time.perf_counter() - started) * 1e3

        if not response.success:
            print(f"Inference failed: {response.message}", file=sys.stderr)
            return 1

        mask = bridge.imgmsg_to_cv2(response.mask, desired_encoding="mono8")
        foreground = int((mask > 0).sum())
        total = int(mask.size)
        print(f"prompt:              {args.prompt!r}")
        print(f"foreground pixels:   {foreground}/{total} "
              f"({100.0 * foreground / max(total, 1):.2f}%)")
        print(f"round trip:          {round_trip_ms:.1f} ms")
        if args.output:
            cv2.imwrite(args.output, mask * 255)
            print(f"mask written to      {args.output}")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())

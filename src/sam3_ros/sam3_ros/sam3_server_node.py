# Copyright (c) 2026 moonlab
# SPDX-License-Identifier: MIT
"""Request/response SAM 3 segmentation node.

The node exposes a single ``sam3_msgs/srv/SegmentImage`` service.  A request
carries the RGB image, the text prompt and an optional per-call confidence
threshold; the response carries a ``mono8`` binary segmentation mask
(0 = background, 1 = foreground) with exactly the input image dimensions.

The model is loaded once at startup and every parameter is declared with a
descriptor, so ``ros2 param describe /sam3_server`` documents the node live.
"""

from __future__ import annotations

import threading

import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from sam3_msgs.srv import SegmentImage

from sam3_ros import parameters
from sam3_ros.image_utils import ImageConversionError, ImageConverter
from sam3_ros.sam3_engine import Sam3Engine


class Sam3ServerNode(Node):
    """ROS 2 service node exposing text-prompted SAM 3 segmentation."""

    def __init__(self) -> None:
        super().__init__("sam3_server")
        parameters.declare_node_parameters(self, streaming=False)

        self._engine_config = parameters.load_engine_config(self)
        self._server_config = parameters.load_server_config(self)
        self._converter = ImageConverter()

        self.get_logger().info(
            "Starting SAM3 server: "
            f"device={self._engine_config.model.device}, "
            f"resolution={self._engine_config.inference.resolution}, "
            f"precision={self._engine_config.inference.precision}, "
            f"threshold={self._engine_config.inference.confidence_threshold}"
        )
        self._engine = Sam3Engine(self._engine_config, logger=self.get_logger())
        self._engine.initialize()

        self._stats_lock = threading.Lock()
        self._requests = 0
        self._failures = 0
        self._total_latency_ms = 0.0

        self._service = self.create_service(
            SegmentImage,
            self._server_config.service_name,
            self._handle_request,
        )
        self.get_logger().info(
            f"Serving '{self.get_fully_qualified_name()}' on service "
            f"{self._service.srv_name} (SAM3 on {self._engine.device})"
        )

        self._stats_timer = None
        if self._server_config.stats_log_period_s > 0.0:
            self._stats_timer = self.create_timer(
                self._server_config.stats_log_period_s, self._log_stats
            )

    # --------------------------------------------------------------- service
    def _handle_request(
        self, request: SegmentImage.Request, response: SegmentImage.Response
    ) -> SegmentImage.Response:
        prompt = request.text_prompt.strip()
        if not prompt:
            response.success = False
            response.message = "text_prompt must not be empty"
            return response

        try:
            rgb = self._converter.to_rgb(request.image)
        except ImageConversionError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=5.0)
            response.success = False
            response.message = str(exc)
            self._record_failure()
            return response

        threshold = (
            None
            if request.confidence_threshold < 0.0
            else float(request.confidence_threshold)
        )
        try:
            result = self._engine.segment(
                rgb, prompt, confidence_threshold=threshold
            )
        except Exception as exc:  # never crash the service on inference errors
            self.get_logger().error(
                f"SAM3 inference failed for prompt {prompt!r}: {exc}",
                throttle_duration_sec=5.0,
            )
            response.success = False
            response.message = f"inference failed: {exc}"
            self._record_failure()
            return response

        response.mask = self._converter.to_mono8(result.mask, request.image.header)
        response.success = True
        response.message = ""
        self._record_success(result.inference_ms)
        self.get_logger().debug(
            f"prompt={prompt!r} instances={result.num_instances} "
            f"latency={result.inference_ms:.1f} ms"
        )
        return response

    # ----------------------------------------------------------------- stats
    def _record_success(self, latency_ms: float) -> None:
        with self._stats_lock:
            self._requests += 1
            self._total_latency_ms += latency_ms

    def _record_failure(self) -> None:
        with self._stats_lock:
            self._failures += 1

    def _log_stats(self) -> None:
        with self._stats_lock:
            requests = self._requests
            failures = self._failures
            total_latency = self._total_latency_ms
            self._requests = 0
            self._failures = 0
            self._total_latency_ms = 0.0
        if requests == 0 and failures == 0:
            return
        average = total_latency / requests if requests else 0.0
        self.get_logger().info(
            f"service stats: requests={requests} failures={failures} "
            f"avg_inference={average:.1f} ms"
        )

    # -------------------------------------------------------------- lifecycle
    def destroy_node(self) -> bool:
        if self._stats_timer is not None:
            self.destroy_timer(self._stats_timer)
            self._stats_timer = None
        if getattr(self, "_engine", None) is not None:
            self._engine.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    """Entry point for ``ros2 run sam3_ros sam3_server``."""
    rclpy.init(args=args)
    node: Sam3ServerNode | None = None
    executor = MultiThreadedExecutor(num_threads=2)
    try:
        node = Sam3ServerNode()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:  # startup failures should exit non-zero, clearly
        if node is not None:
            node.get_logger().fatal(f"sam3_server failed: {exc}")
        else:
            print(f"[sam3_server] fatal startup error: {exc}", flush=True)
        raise
    finally:
        if node is not None:
            executor.remove_node(node)
            node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

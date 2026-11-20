# Copyright (c) 2026 moonlab
# SPDX-License-Identifier: MIT
"""Streaming SAM 3 segmentation node.

Subscribes to an RGB image topic, runs SAM 3 with a prompt fixed at startup and
publishes a ``mono8`` binary segmentation mask (0 = background, 1 = foreground)
with exactly the input image dimensions.

Design notes
------------
* The prompt embedding is computed once during warmup and reused for every
  frame, so the text encoder never runs in the steady state.
* The ROS callback only enqueues the image message.  A dedicated worker thread
  performs conversion + inference, and a queue of depth ``stream.queue_size``
  guarantees that the freshest frame wins instead of building latency.
* ``stream.max_rate_hz`` throttles GPU usage without dropping the subscription.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import replace

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from sam3_ros import parameters
from sam3_ros.image_utils import ImageConversionError, ImageConverter
from sam3_ros.sam3_engine import Sam3Engine


class Sam3StreamNode(Node):
    """ROS 2 node that continuously segments a video stream with one prompt."""

    def __init__(self) -> None:
        super().__init__("sam3_stream")
        parameters.declare_node_parameters(self, streaming=True)

        self._engine_config = parameters.load_engine_config(self)
        self._stream_config = parameters.load_stream_config(self)
        self._converter = ImageConverter()

        # Warm up with the streaming prompt so its text embedding is cached.
        if (
            self._engine_config.inference.warmup
            and self._engine_config.inference.warmup_iterations > 0
        ):
            self._engine_config = replace(
                self._engine_config,
                inference=replace(
                    self._engine_config.inference,
                    warmup_prompt=self._stream_config.text_prompt,
                ),
            )

        cfg = self._stream_config
        self.get_logger().info(
            "Starting SAM3 stream: "
            f"prompt={cfg.text_prompt!r}, input={cfg.image_topic}, "
            f"mask={cfg.mask_topic}, device={self._engine_config.model.device}, "
            f"threshold={self._engine_config.inference.confidence_threshold}"
        )

        self._engine = Sam3Engine(self._engine_config, logger=self.get_logger())
        self._engine.initialize()
        self._engine.precompute_text(cfg.text_prompt)

        # Output publisher --------------------------------------------------
        output_qos = self._make_qos(
            cfg.output_qos_reliability, cfg.output_qos_depth
        )
        self._mask_publisher = self.create_publisher(
            Image, cfg.mask_topic, output_qos
        )

        # Worker / queue ----------------------------------------------------
        self._frame_queue: queue.Queue[Image | None] = queue.Queue(
            maxsize=cfg.queue_size
        )
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop, name="sam3-inference", daemon=True
        )
        self._worker.start()

        # Stats -------------------------------------------------------------
        self._stats_lock = threading.Lock()
        self._submitted = 0
        self._dropped = 0
        self._processed = 0
        self._errors = 0
        self._total_latency_ms = 0.0
        self._last_submit_time = 0.0

        self._stats_timer = None
        if cfg.stats_log_period_s > 0.0:
            self._stats_timer = self.create_timer(
                cfg.stats_log_period_s, self._log_stats
            )

        input_qos = self._make_qos(
            cfg.input_qos_reliability, cfg.input_qos_depth
        )
        self._subscription = self.create_subscription(
            Image, cfg.image_topic, self._on_image, input_qos
        )
        self.get_logger().info(
            f"SAM3 stream ready on {self._engine.device}; "
            f"publishing {cfg.mask_topic}"
        )

    # ------------------------------------------------------------------ QoS
    @staticmethod
    def _make_qos(reliability: str, depth: int) -> QoSProfile:
        policy = (
            ReliabilityPolicy.BEST_EFFORT
            if reliability == "best_effort"
            else ReliabilityPolicy.RELIABLE
        )
        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=depth, reliability=policy
        )

    # ------------------------------------------------------------- callbacks
    def _on_image(self, msg: Image) -> None:
        if self._stop_event.is_set():
            return

        cfg = self._stream_config
        now = time.monotonic()
        if cfg.max_rate_hz > 0.0:
            min_period = 1.0 / cfg.max_rate_hz
            if now - self._last_submit_time < min_period:
                self._count_drop()
                return
        self._last_submit_time = now

        try:
            self._frame_queue.put_nowait(msg)
        except queue.Full:
            if not cfg.drop_stale_frames:
                self._count_drop()
                return
            try:
                self._frame_queue.get_nowait()
                self._count_drop()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(msg)
            except queue.Full:  # pragma: no cover - race with the worker
                self._count_drop()

    def _count_drop(self) -> None:
        with self._stats_lock:
            self._dropped += 1

    # ---------------------------------------------------------------- worker
    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                message = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if message is None:
                break
            try:
                self._process(message)
            except ImageConversionError as exc:
                self.get_logger().error(str(exc), throttle_duration_sec=5.0)
                with self._stats_lock:
                    self._errors += 1
            except Exception as exc:  # keep the stream alive on model errors
                self.get_logger().error(
                    f"SAM3 stream inference failed: {exc}",
                    throttle_duration_sec=5.0,
                )
                with self._stats_lock:
                    self._errors += 1
            finally:
                self._frame_queue.task_done()

    def _process(self, message: Image) -> None:
        cfg = self._stream_config
        rgb = self._converter.to_rgb(message)
        result = self._engine.segment(
            rgb, cfg.text_prompt, use_text_cache=True
        )
        header = message.header

        self._mask_publisher.publish(
            self._converter.to_mono8(result.mask, header)
        )

        with self._stats_lock:
            self._processed += 1
            self._total_latency_ms += result.inference_ms
        self.get_logger().debug(
            f"instances={result.num_instances} "
            f"latency={result.inference_ms:.1f} ms"
        )

    # ----------------------------------------------------------------- stats
    def _log_stats(self) -> None:
        with self._stats_lock:
            processed = self._processed
            dropped = self._dropped
            errors = self._errors
            total_latency = self._total_latency_ms
            self._processed = 0
            self._dropped = 0
            self._errors = 0
            self._total_latency_ms = 0.0
        if processed == 0 and dropped == 0 and errors == 0:
            return
        average = total_latency / processed if processed else 0.0
        fps = processed / self._stream_config.stats_log_period_s
        self.get_logger().info(
            f"stream stats: processed={processed} dropped={dropped} "
            f"errors={errors} avg_inference={average:.1f} ms "
            f"fps={fps:.2f} queue={self._frame_queue.qsize()}"
        )

    # -------------------------------------------------------------- lifecycle
    def destroy_node(self) -> bool:
        self._stop_event.set()
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass
        if getattr(self, "_worker", None) is not None:
            self._worker.join(timeout=10.0)
        if self._stats_timer is not None:
            self.destroy_timer(self._stats_timer)
            self._stats_timer = None
        if getattr(self, "_engine", None) is not None:
            self._engine.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    """Entry point for ``ros2 run sam3_ros sam3_stream``."""
    rclpy.init(args=args)
    node: Sam3StreamNode | None = None
    try:
        node = Sam3StreamNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:  # surface startup errors as a non-zero exit
        if node is not None:
            node.get_logger().fatal(f"sam3_stream failed: {exc}")
        else:
            print(f"[sam3_stream] fatal startup error: {exc}", flush=True)
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

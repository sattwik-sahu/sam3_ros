# Copyright (c) 2026 moonlab
# SPDX-License-Identifier: MIT
"""ROS 2 parameter declarations and validation for the SAM 3 nodes.

Every parameter is declared with a :class:`rcl_interfaces.msg.ParameterDescriptor`,
which means ``ros2 param describe <node> <name>`` documents the full set at
runtime.  The parameter names use dots so that the nested YAML files in
``config/`` map 1:1 onto them::

    model:
      device: cuda
    inference:
      precision: bf16

becomes ``model.device`` and ``inference.precision``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node

from sam3_ros.config import (
    EngineConfig,
    InferenceConfig,
    ModelConfig,
    ServerConfig,
    StreamConfig,
)


@dataclass(frozen=True)
class ParamSpec:
    """Default value, Python type and human-readable documentation."""

    default: Any
    ptype: type
    description: str


MODEL_PARAMS: dict[str, ParamSpec] = {
    "model.checkpoint_path": ParamSpec(
        "",
        str,
        "Absolute path to a SAM3 checkpoint (.pt). When set, the checkpoint is "
        "loaded directly and no Hugging Face lookup is performed. A directory "
        "containing the file named by model.hf_filename is also accepted. "
        "Empty means: resolve via model.load_from_hf.",
    ),
    "model.device": ParamSpec(
        "cuda",
        str,
        "Torch device for the model, e.g. 'cuda', 'cuda:1' or 'cpu'. Falls back "
        "to CPU when CUDA is unavailable and model.allow_cpu_fallback is true.",
    ),
    "model.eval_mode": ParamSpec(
        True,
        bool,
        "Put the model in evaluation mode (inference).",
    ),
    "model.load_from_hf": ParamSpec(
        True,
        bool,
        "Download/cache the checkpoint from Hugging Face when "
        "model.checkpoint_path is empty. Uses the standard HF cache "
        "(HF_HOME), which is a persistent volume in the Docker deployment.",
    ),
    "model.hf_repo_id": ParamSpec(
        "facebook/sam3",
        str,
        "Hugging Face repository that holds the gated SAM3 weights.",
    ),
    "model.hf_filename": ParamSpec(
        "sam3.pt",
        str,
        "Checkpoint filename inside model.hf_repo_id.",
    ),
    "model.hf_revision": ParamSpec(
        "",
        str,
        "Optional Hugging Face revision (branch, tag or commit) for the "
        "checkpoint. Empty means the repository default branch.",
    ),
    "model.local_files_only": ParamSpec(
        False,
        bool,
        "Never touch the network: resolve the checkpoint from the local HF "
        "cache only. Useful for fully offline deployments.",
    ),
    "model.bpe_path": ParamSpec(
        "",
        str,
        "Optional path to the SAM3 BPE vocabulary "
        "(bpe_simple_vocab_16e6.txt.gz). Empty uses the copy shipped with the "
        "sam3 Python package.",
    ),
    "model.compile": ParamSpec(
        False,
        bool,
        "Compile the model with torch.compile. First inference is much slower "
        "while the graph is captured, subsequent ones can be faster. Enable "
        "only after benchmarking your GPU.",
    ),
    "model.enable_segmentation": ParamSpec(
        True,
        bool,
        "Build the segmentation head. Must stay true for mask output.",
    ),
    "model.enable_inst_interactivity": ParamSpec(
        False,
        bool,
        "Also build the interactive (point/box) predictor. Not required for "
        "text-prompted segmentation and adds GPU memory.",
    ),
    "model.allow_cpu_fallback": ParamSpec(
        True,
        bool,
        "Fall back to CPU with a warning when a CUDA device was requested but "
        "is not available. Set false to fail fast on GPU configuration errors.",
    ),
}

INFERENCE_PARAMS: dict[str, ParamSpec] = {
    "inference.resolution": ParamSpec(
        1008,
        int,
        "Square input resolution fed to the SAM3 image encoder. 1008 is the "
        "value the released weights were trained with; changing it is "
        "experimental.",
    ),
    "inference.confidence_threshold": ParamSpec(
        0.5,
        float,
        "Default score threshold in [0, 1]. Detections below it are discarded. "
        "Service requests may override it per call.",
    ),
    "inference.precision": ParamSpec(
        "bf16",
        str,
        "Autocast precision: 'bf16' (recommended on Ampere+), 'fp16' or "
        "'fp32' to disable autocast. Weights stay in fp32; only activations "
        "are cast.",
    ),
    "inference.tf32": ParamSpec(
        True,
        bool,
        "Allow TF32 for matmul and cuDNN on Ampere+ GPUs. Large speedup with "
        "negligible segmentation quality impact.",
    ),
    "inference.cudnn_benchmark": ParamSpec(
        True,
        bool,
        "Enable cuDNN autotuning. Beneficial when the input resolution is "
        "fixed across frames (the typical streaming case).",
    ),
    "inference.cache_text_embeddings": ParamSpec(
        True,
        bool,
        "Cache text-prompt embeddings. Streaming with a fixed prompt then "
        "skips the text encoder on every frame; the server caches recently "
        "used prompts in an LRU of inference.text_cache_size.",
    ),
    "inference.text_cache_size": ParamSpec(
        8,
        int,
        "Maximum number of cached text-prompt embeddings (0 disables the LRU "
        "while keeping single-prompt caching for streaming).",
    ),
    "inference.warmup": ParamSpec(
        True,
        bool,
        "Run synthetic inferences after loading so CUDA kernels and cuDNN "
        "plans are ready before the first real frame.",
    ),
    "inference.warmup_iterations": ParamSpec(
        1,
        int,
        "Number of warmup inferences.",
    ),
    "inference.warmup_prompt": ParamSpec(
        "object",
        str,
        "Text prompt used for warmup on the server node. The streaming node "
        "always warms up with its configured stream.text_prompt so that "
        "prompt embedding is cached too.",
    ),
}

SERVER_PARAMS: dict[str, ParamSpec] = {
    "server.service_name": ParamSpec(
        "segment",
        str,
        "Name of the SegmentImage service. A name starting with '~/' is "
        "private to the node; a plain name is resolved in the node namespace.",
    ),
    "server.stats_log_period_s": ParamSpec(
        30.0,
        float,
        "Period for the aggregate throughput/latency log line. 0 disables it.",
    ),
}

STREAM_PARAMS: dict[str, ParamSpec] = {
    "stream.text_prompt": ParamSpec(
        "object",
        str,
        "Fixed text prompt used for every frame. It is encoded once at "
        "startup and its embedding is reused for all frames.",
    ),
    "stream.image_topic": ParamSpec(
        "/camera/color/image_raw",
        str,
        "Input sensor_msgs/Image topic. rgb8, bgr8, rgba8, bgra8 and mono8 "
        "encodings are converted to RGB automatically.",
    ),
    "stream.mask_topic": ParamSpec(
        "/sam3/mask",
        str,
        "Output binary segmentation mask topic. sensor_msgs/Image with "
        "encoding mono8, values {0 = background, 1 = foreground}, exactly the "
        "same width/height as the input image.",
    ),
    "stream.max_rate_hz": ParamSpec(
        0.0,
        float,
        "Maximum processing rate. Frames arriving faster are dropped before "
        "they reach the GPU. 0 means as fast as the model allows.",
    ),
    "stream.queue_size": ParamSpec(
        1,
        int,
        "Capacity of the frame queue between the ROS callback and the GPU "
        "worker. 1 always processes the freshest frame.",
    ),
    "stream.drop_stale_frames": ParamSpec(
        True,
        bool,
        "Drop the oldest queued frame when the queue is full. Disable to "
        "process every frame (adds latency under load).",
    ),
    "stream.input_qos_reliability": ParamSpec(
        "best_effort",
        str,
        "Input QoS reliability: 'best_effort' (recommended for cameras) or "
        "'reliable'.",
    ),
    "stream.input_qos_depth": ParamSpec(
        5,
        int,
        "Input QoS history depth.",
    ),
    "stream.output_qos_reliability": ParamSpec(
        "reliable",
        str,
        "Output QoS reliability: 'reliable' (default, lossless in-process and "
        "LAN use) or 'best_effort' for lossy low-latency links.",
    ),
    "stream.output_qos_depth": ParamSpec(
        1,
        int,
        "Output QoS history depth. 1 keeps only the latest mask.",
    ),
    "stream.stats_log_period_s": ParamSpec(
        30.0,
        float,
        "Period for the throughput/drop/latency log line. 0 disables it.",
    ),
}

_TYPE_MAP: dict[type, int] = {
    bool: ParameterType.PARAMETER_BOOL,
    int: ParameterType.PARAMETER_INTEGER,
    float: ParameterType.PARAMETER_DOUBLE,
    str: ParameterType.PARAMETER_STRING,
}


def _declare(node: Node, specs: Iterable[tuple[str, ParamSpec]]) -> None:
    for name, spec in specs:
        node.declare_parameter(
            name,
            spec.default,
            ParameterDescriptor(
                name=name,
                type=_TYPE_MAP[spec.ptype],
                description=spec.description,
                read_only=False,
                dynamic_typing=False,
            ),
        )


def declare_node_parameters(node: Node, *, streaming: bool) -> None:
    """Declare every parameter used by the requested node flavour.

    Args:
        node: Node whose parameter server receives the declarations.
        streaming: ``True`` for the streaming node, ``False`` for the server.
    """
    specs: list[tuple[str, ParamSpec]] = []
    specs.extend(MODEL_PARAMS.items())
    specs.extend(INFERENCE_PARAMS.items())
    specs.extend(STREAM_PARAMS.items() if streaming else SERVER_PARAMS.items())
    _declare(node, specs)


def _value(node: Node, name: str) -> Any:
    return node.get_parameter(name).value


def load_model_config(node: Node) -> ModelConfig:
    """Build and validate a :class:`ModelConfig` from the parameter server."""
    cfg = ModelConfig(
        checkpoint_path=_value(node, "model.checkpoint_path"),
        device=_value(node, "model.device"),
        eval_mode=_value(node, "model.eval_mode"),
        load_from_hf=_value(node, "model.load_from_hf"),
        hf_repo_id=_value(node, "model.hf_repo_id"),
        hf_filename=_value(node, "model.hf_filename"),
        hf_revision=_value(node, "model.hf_revision"),
        local_files_only=_value(node, "model.local_files_only"),
        bpe_path=_value(node, "model.bpe_path"),
        compile=_value(node, "model.compile"),
        enable_segmentation=_value(node, "model.enable_segmentation"),
        enable_inst_interactivity=_value(node, "model.enable_inst_interactivity"),
        allow_cpu_fallback=_value(node, "model.allow_cpu_fallback"),
    )
    if not cfg.device:
        raise ValueError("model.device must not be empty")
    if not cfg.checkpoint_path and not cfg.load_from_hf:
        raise ValueError(
            "Set model.checkpoint_path or enable model.load_from_hf; otherwise "
            "the model would be initialized with random weights."
        )
    if not cfg.enable_segmentation:
        raise ValueError("model.enable_segmentation must be true to publish masks")
    if not cfg.hf_filename:
        raise ValueError("model.hf_filename must not be empty")
    return cfg


def load_inference_config(node: Node) -> InferenceConfig:
    """Build and validate an :class:`InferenceConfig` from the parameter server."""
    cfg = InferenceConfig(
        resolution=_value(node, "inference.resolution"),
        confidence_threshold=_value(node, "inference.confidence_threshold"),
        precision=_value(node, "inference.precision").lower(),
        tf32=_value(node, "inference.tf32"),
        cudnn_benchmark=_value(node, "inference.cudnn_benchmark"),
        cache_text_embeddings=_value(node, "inference.cache_text_embeddings"),
        text_cache_size=_value(node, "inference.text_cache_size"),
        warmup=_value(node, "inference.warmup"),
        warmup_iterations=_value(node, "inference.warmup_iterations"),
        warmup_prompt=_value(node, "inference.warmup_prompt"),
    )
    if cfg.resolution <= 0:
        raise ValueError("inference.resolution must be positive")
    if not 0.0 <= cfg.confidence_threshold <= 1.0:
        raise ValueError("inference.confidence_threshold must be in [0, 1]")
    if cfg.precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError(
            "inference.precision must be one of 'bf16', 'fp16' or 'fp32', "
            f"got {cfg.precision!r}"
        )
    if cfg.text_cache_size < 0:
        raise ValueError("inference.text_cache_size must be >= 0")
    if cfg.warmup_iterations < 0:
        raise ValueError("inference.warmup_iterations must be >= 0")
    return cfg


def load_engine_config(node: Node) -> EngineConfig:
    """Build the aggregate engine configuration."""
    return EngineConfig(
        model=load_model_config(node),
        inference=load_inference_config(node),
    )


def load_server_config(node: Node) -> ServerConfig:
    """Build and validate the server-node configuration."""
    cfg = ServerConfig(
        service_name=_value(node, "server.service_name"),
        stats_log_period_s=_value(node, "server.stats_log_period_s"),
    )
    if not cfg.service_name:
        raise ValueError("server.service_name must not be empty")
    if cfg.stats_log_period_s < 0.0:
        raise ValueError("server.stats_log_period_s must be >= 0")
    return cfg


def load_stream_config(node: Node) -> StreamConfig:
    """Build and validate the streaming-node configuration."""
    cfg = StreamConfig(
        text_prompt=_value(node, "stream.text_prompt"),
        image_topic=_value(node, "stream.image_topic"),
        mask_topic=_value(node, "stream.mask_topic"),
        max_rate_hz=_value(node, "stream.max_rate_hz"),
        queue_size=_value(node, "stream.queue_size"),
        drop_stale_frames=_value(node, "stream.drop_stale_frames"),
        input_qos_reliability=_value(node, "stream.input_qos_reliability"),
        input_qos_depth=_value(node, "stream.input_qos_depth"),
        output_qos_reliability=_value(node, "stream.output_qos_reliability"),
        output_qos_depth=_value(node, "stream.output_qos_depth"),
        stats_log_period_s=_value(node, "stream.stats_log_period_s"),
    )
    if not cfg.text_prompt.strip():
        raise ValueError("stream.text_prompt must not be empty")
    for name, topic in (
        ("stream.image_topic", cfg.image_topic),
        ("stream.mask_topic", cfg.mask_topic),
    ):
        if not topic:
            raise ValueError(f"{name} must not be empty")
    if cfg.max_rate_hz < 0.0:
        raise ValueError("stream.max_rate_hz must be >= 0")
    if cfg.queue_size < 1:
        raise ValueError("stream.queue_size must be >= 1")
    if cfg.input_qos_depth < 1 or cfg.output_qos_depth < 1:
        raise ValueError("QoS depths must be >= 1")
    if cfg.input_qos_reliability not in {"best_effort", "reliable"}:
        raise ValueError(
            "stream.input_qos_reliability must be 'best_effort' or 'reliable'"
        )
    if cfg.output_qos_reliability not in {"best_effort", "reliable"}:
        raise ValueError(
            "stream.output_qos_reliability must be 'best_effort' or 'reliable'"
        )
    if cfg.stats_log_period_s < 0.0:
        raise ValueError("stream.stats_log_period_s must be >= 0")
    return cfg

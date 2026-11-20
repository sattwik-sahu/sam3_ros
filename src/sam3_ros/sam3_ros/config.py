# Copyright (c) 2026 moonlab
# SPDX-License-Identifier: MIT
"""Typed configuration objects for the SAM 3 ROS 2 nodes.

ROS 2 parameters are declared with dotted names (``model.device``,
``inference.precision``, ...) which map directly onto the nested YAML layout in
``config/*.yaml``.  This module converts the flat parameter namespace into
frozen dataclasses so the rest of the code never has to touch the parameter
server directly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Everything needed to locate and instantiate the SAM 3 model."""

    checkpoint_path: str = ""
    device: str = "cuda"
    eval_mode: bool = True
    load_from_hf: bool = True
    hf_repo_id: str = "facebook/sam3"
    hf_filename: str = "sam3.pt"
    hf_revision: str = ""
    local_files_only: bool = False
    bpe_path: str = ""
    compile: bool = False
    enable_segmentation: bool = True
    enable_inst_interactivity: bool = False
    allow_cpu_fallback: bool = True


@dataclass(frozen=True)
class InferenceConfig:
    """Runtime behaviour of the SAM 3 inference engine."""

    resolution: int = 1008
    confidence_threshold: float = 0.5
    precision: str = "bf16"
    tf32: bool = True
    cudnn_benchmark: bool = True
    cache_text_embeddings: bool = True
    text_cache_size: int = 8
    warmup: bool = True
    warmup_iterations: int = 1
    warmup_prompt: str = "object"


@dataclass(frozen=True)
class EngineConfig:
    """Aggregate configuration consumed by :class:`sam3_ros.sam3_engine.Sam3Engine`."""

    model: ModelConfig = ModelConfig()
    inference: InferenceConfig = InferenceConfig()


@dataclass(frozen=True)
class ServerConfig:
    """Configuration specific to the request/response server node."""

    service_name: str = "segment"
    stats_log_period_s: float = 30.0


@dataclass(frozen=True)
class StreamConfig:
    """Configuration specific to the continuous streaming node."""

    text_prompt: str = "object"
    image_topic: str = "/camera/color/image_raw"
    mask_topic: str = "/sam3/mask"
    max_rate_hz: float = 0.0
    queue_size: int = 1
    drop_stale_frames: bool = True
    input_qos_reliability: str = "best_effort"
    input_qos_depth: int = 5
    output_qos_reliability: str = "reliable"
    output_qos_depth: int = 1
    stats_log_period_s: float = 30.0

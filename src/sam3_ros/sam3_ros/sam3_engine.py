# Copyright (c) 2026 moonlab
# SPDX-License-Identifier: MIT
"""Efficient SAM 3 inference engine shared by both ROS 2 nodes.

The engine is a thin, thread-safe wrapper around the official
``facebookresearch/sam3`` Python package.  It follows the implementation used by
`cap-x <https://github.com/capgym/cap-x>`_:

* a single model + processor are loaded once and reused;
* inference runs under ``torch.inference_mode`` and optional BF16/FP16 autocast;
* TF32 and cuDNN autotuning are enabled on Ampere+ GPUs;
* text-prompt embeddings are computed once and cached, so a streaming node with
  a fixed prompt never pays for the text encoder again;
* all GPU work is serialized with a re-entrant lock, making the engine safe for
  multi-threaded ROS executors and concurrent service calls.

The public API deliberately deals with plain NumPy arrays so it has no ROS
dependency and can be unit-tested or reused from a plain Python process.
"""

from __future__ import annotations

import contextlib
import gc
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sam3_ros.config import EngineConfig

# torch and sam3 are heavy imports.  They are imported lazily inside
# Sam3Engine.initialize() so that --help / parameter introspection stay fast,
# and so a missing GPU stack produces one clear error instead of a traceback at
# module import time.


@dataclass(frozen=True)
class SegmentResult:
    """Result of one text-prompted segmentation pass.

    Attributes:
        mask: ``uint8`` array of shape ``(H, W)`` aligned with the input image,
            with exactly the same width and height. Values are ``0``
            (background) and ``1`` (foreground). The mask is the union of every
            SAM 3 instance above the confidence threshold.
        num_instances: Number of SAM 3 instances that were merged into the mask.
        inference_ms: end-to-end latency of this call in milliseconds.
    """

    mask: np.ndarray
    num_instances: int
    inference_ms: float


class Sam3Engine:
    """Thread-safe SAM 3 image segmentation engine."""

    def __init__(self, config: EngineConfig, logger: Any | None = None) -> None:
        self._config = config
        self._logger = logger
        self._lock = threading.RLock()
        self._model: Any | None = None
        self._processor: Any | None = None
        self._device = "cpu"
        self._device_type = "cpu"
        self._autocast_dtype: Any | None = None
        self._torch: Any | None = None
        self._text_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._closed = False

    # ------------------------------------------------------------------ setup
    def initialize(self) -> None:
        """Load the model, configure torch and optionally run warmup."""
        if self._model is not None:
            return

        import torch

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self._torch = torch
        cfg = self._config
        self._configure_torch(torch, cfg)
        self._device = self._resolve_device(torch, cfg)
        self._device_type = "cuda" if self._device.startswith("cuda") else "cpu"
        self._autocast_dtype = self._resolve_autocast(torch, cfg)

        checkpoint = self._resolve_checkpoint()
        self._log(
            "info",
            "Loading SAM3 model "
            f"(device={self._device}, precision={cfg.inference.precision}, "
            f"checkpoint={checkpoint or 'random weights'})",
        )
        started = time.perf_counter()
        model = build_sam3_image_model(
            bpe_path=cfg.model.bpe_path or None,
            device=self._device,
            eval_mode=cfg.model.eval_mode,
            checkpoint_path=checkpoint,
            load_from_HF=False,
            enable_segmentation=cfg.model.enable_segmentation,
            enable_inst_interactivity=cfg.model.enable_inst_interactivity,
            compile=cfg.model.compile,
        )
        self._model = model
        self._processor = Sam3Processor(
            model,
            resolution=cfg.inference.resolution,
            device=self._device,
            confidence_threshold=cfg.inference.confidence_threshold,
        )
        load_s = time.perf_counter() - started
        self._log("info", f"SAM3 model ready in {load_s:.1f} s")

        if cfg.inference.warmup and cfg.inference.warmup_iterations > 0:
            self.warmup(cfg.inference.warmup_prompt, cfg.inference.warmup_iterations)

    def _configure_torch(self, torch: Any, cfg: EngineConfig) -> None:
        if cfg.inference.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = cfg.inference.cudnn_benchmark
        try:
            # Prefer fast fp32 matmuls on Ampere+ without changing model dtype.
            torch.set_float32_matmul_precision("high")
        except Exception:  # pragma: no cover - older torch builds
            pass

    def _resolve_device(self, torch: Any, cfg: EngineConfig) -> str:
        requested = cfg.model.device.strip()
        if requested.startswith("cuda"):
            if torch.cuda.is_available():
                index = 0
                if ":" in requested:
                    index = int(requested.split(":", 1)[1])
                torch.cuda.set_device(index)
                name = torch.cuda.get_device_name(index)
                total_gib = torch.cuda.get_device_properties(index).total_memory / 1024**3
                self._log("info", f"Using {requested} ({name}, {total_gib:.1f} GiB)")
                return requested
            message = (
                f"CUDA device {requested!r} requested but torch.cuda.is_available() "
                "is false"
            )
            if not cfg.model.allow_cpu_fallback:
                raise RuntimeError(message)
            self._log("warning", f"{message}; falling back to CPU (slow)")
            return "cpu"
        if requested == "cpu":
            self._log("info", "Running SAM3 on CPU")
            return "cpu"
        raise ValueError(f"Unsupported model.device {requested!r}")

    def _resolve_autocast(self, torch: Any, cfg: EngineConfig) -> Any | None:
        precision = cfg.inference.precision
        if self._device_type != "cuda":
            if precision != "fp32":
                self._log(
                    "info",
                    "Autocast is only enabled for CUDA; using fp32 on CPU",
                )
            return None
        if precision == "bf16":
            return torch.bfloat16
        if precision == "fp16":
            return torch.float16
        return None

    def _resolve_checkpoint(self) -> str | None:
        cfg = self._config.model
        if cfg.checkpoint_path:
            path = Path(os.path.expanduser(cfg.checkpoint_path))
            if path.is_dir():
                path = path / cfg.hf_filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"model.checkpoint_path does not exist: {path}"
                )
            return str(path)
        if not cfg.load_from_hf:
            return None

        from huggingface_hub import hf_hub_download

        kwargs: dict[str, Any] = {
            "repo_id": cfg.hf_repo_id,
            "filename": cfg.hf_filename,
            "local_files_only": cfg.local_files_only,
        }
        if cfg.hf_revision:
            kwargs["revision"] = cfg.hf_revision
        token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )
        if token:
            kwargs["token"] = token
        self._log(
            "info",
            f"Resolving {cfg.hf_filename} from Hugging Face repo "
            f"{cfg.hf_repo_id!r} "
            f"(cache: {os.environ.get('HF_HOME', '~/.cache/huggingface')})",
        )
        return hf_hub_download(**kwargs)

    # -------------------------------------------------------------- inference
    @contextlib.contextmanager
    def _autocast(self):
        if self._autocast_dtype is None or self._torch is None:
            yield
            return
        with self._torch.autocast(
            device_type=self._device_type, dtype=self._autocast_dtype
        ):
            yield

    def _forward_text_locked(self, prompt: str) -> dict[str, Any]:
        """Encode a text prompt and place the result in the LRU cache."""
        assert self._model is not None
        started = time.perf_counter()
        with self._torch.inference_mode(), self._autocast():
            outputs = self._model.backbone.forward_text(
                [prompt], device=self._device
            )
        if self._config.inference.cache_text_embeddings:
            limit = max(1, self._config.inference.text_cache_size)
            self._text_cache[prompt] = outputs
            self._text_cache.move_to_end(prompt)
            while len(self._text_cache) > limit:
                self._text_cache.popitem(last=False)
        self._log(
            "debug",
            f"Encoded text prompt {prompt!r} in "
            f"{(time.perf_counter() - started) * 1e3:.1f} ms",
        )
        return outputs

    def precompute_text(self, prompt: str) -> None:
        """Encode and cache ``prompt`` eagerly (used by the streaming node)."""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Text prompt must not be empty")
        with self._lock:
            self._ensure_initialized()
            if prompt not in self._text_cache:
                self._forward_text_locked(prompt)

    def _forward_grounding_locked(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run the detector/segmentation heads reusing image + text features.

        ``Sam3Processor.set_text_prompt`` does exactly this after recomputing the
        text embedding.  Calling the grounding step directly is the only public
        surface the upstream package exposes for reusing a cached prompt.
        """
        assert self._processor is not None and self._model is not None
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self._model._get_dummy_prompt()
        return self._processor._forward_grounding(state)

    def segment(
        self,
        image: np.ndarray | Image.Image,
        prompt: str,
        confidence_threshold: float | None = None,
        *,
        use_text_cache: bool = True,
    ) -> SegmentResult:
        """Segment all instances of ``prompt`` in an RGB image.

        Args:
            image: ``uint8`` RGB array of shape ``(H, W, 3)`` or a PIL image.
            prompt: Free-form text concept.
            confidence_threshold: Optional per-call threshold in ``[0, 1]``.
                ``None`` uses ``inference.confidence_threshold``.
            use_text_cache: Reuse a previously encoded prompt embedding when
                available.  The streaming node always passes ``True``.

        Returns:
            A :class:`SegmentResult` aligned with the input image.
        """
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Text prompt must not be empty")
        pil_image = self._to_pil(image)
        width, height = pil_image.size

        with self._lock:
            self._ensure_initialized()
            torch = self._torch
            processor = self._processor
            assert torch is not None and processor is not None

            threshold = self._resolve_threshold(confidence_threshold)
            processor.confidence_threshold = threshold

            started = time.perf_counter()
            use_cache = (
                use_text_cache
                and self._config.inference.cache_text_embeddings
            )
            with torch.inference_mode(), self._autocast():
                state = processor.set_image(pil_image)
                cached = self._text_cache.get(prompt) if use_cache else None
                if cached is None:
                    if use_cache:
                        # Encode now so subsequent frames hit the cache.
                        cached = self._forward_text_locked(prompt)
                        state["backbone_out"].update(cached)
                        state = self._forward_grounding_locked(state)
                    else:
                        state = processor.set_text_prompt(
                            state=state, prompt=prompt
                        )
                else:
                    self._text_cache.move_to_end(prompt)
                    state["backbone_out"].update(cached)
                    state = self._forward_grounding_locked(state)

            result = self._extract_result(state, width, height)
            elapsed_ms = (time.perf_counter() - started) * 1e3
            result = SegmentResult(
                mask=result.mask,
                num_instances=result.num_instances,
                inference_ms=elapsed_ms,
            )
            self._log(
                "debug",
                f"prompt={prompt!r} instances={result.num_instances} "
                f"latency={elapsed_ms:.1f} ms",
            )
            return result

    def warmup(self, prompt: str, iterations: int = 1) -> None:
        """Run synthetic inferences to populate CUDA/cuDNN caches."""
        if iterations <= 0:
            return
        self._log(
            "info",
            f"Warming up SAM3 with prompt {prompt!r} ({iterations} iteration(s))",
        )
        # 720p is a realistic robotics frame and exercises the full resize path.
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.segment(dummy, prompt, use_text_cache=True)
        self._log("info", "SAM3 warmup complete")

    # ----------------------------------------------------------------- output
    @staticmethod
    def _to_pil(image: np.ndarray | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(
                "image must be an (H, W, 3) RGB array or a PIL image, got "
                f"shape {array.shape}"
            )
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(np.ascontiguousarray(array), mode="RGB")

    def _resolve_threshold(self, value: float | None) -> float:
        threshold = (
            self._config.inference.confidence_threshold if value is None else value
        )
        if threshold < 0.0:
            threshold = self._config.inference.confidence_threshold
        return float(min(max(threshold, 0.0), 1.0))

    def _extract_result(
        self, state: dict[str, Any], width: int, height: int
    ) -> SegmentResult:
        masks = state.get("masks")
        if masks is None:
            return SegmentResult(
                mask=np.zeros((height, width), dtype=np.uint8),
                num_instances=0,
                inference_ms=0.0,
            )

        masks_np = self._to_numpy(masks)
        if masks_np.ndim == 4:
            # (N, 1, H, W) -> (N, H, W)
            masks_np = masks_np[:, 0]
        if masks_np.ndim != 3:
            raise RuntimeError(
                f"Unexpected SAM3 mask shape {masks_np.shape}; expected (N, H, W)"
            )

        count = int(masks_np.shape[0])
        if count == 0:
            return SegmentResult(
                mask=np.zeros((height, width), dtype=np.uint8),
                num_instances=0,
                inference_ms=0.0,
            )

        union = (masks_np > 0).any(axis=0)
        if union.shape != (height, width):
            # Defensive: the processor resizes to the original image, but a
            # mismatch must never corrupt the ROS message.
            union_img = Image.fromarray(union.astype(np.uint8) * 255, mode="L")
            union_img = union_img.resize((width, height), Image.NEAREST)
            union = np.asarray(union_img) > 0

        return SegmentResult(
            mask=union.astype(np.uint8),
            num_instances=count,
            inference_ms=0.0,
        )

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            tensor = value.detach().cpu()
            # NumPy has no bfloat16 dtype; widen to float32 first.
            if str(tensor.dtype) == "torch.bfloat16":
                tensor = tensor.float()
            return tensor.numpy()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value)

    # ------------------------------------------------------------------ misc
    @property
    def device(self) -> str:
        return self._device

    @property
    def is_initialized(self) -> bool:
        return self._model is not None

    def _ensure_initialized(self) -> None:
        if self._model is None:
            raise RuntimeError("Sam3Engine.initialize() must be called first")
        if self._closed:
            raise RuntimeError("Sam3Engine has been closed")

    def _log(self, level: str, message: str) -> None:
        if self._logger is None:
            return
        log_fn = getattr(self._logger, level, None)
        if log_fn is None:
            log_fn = getattr(self._logger, "info", None)
        if log_fn is not None:
            log_fn(message)

    def close(self) -> None:
        """Release the model and cached GPU memory."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._text_cache.clear()
            self._processor = None
            self._model = None
            gc.collect()
            if self._torch is not None and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
            self._log("info", "SAM3 engine released")

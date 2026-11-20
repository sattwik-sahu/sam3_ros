#!/usr/bin/env python3
"""Benchmark SAM 3 latency and GPU memory across input resolutions.

Runs the same Sam3Engine used by the ROS nodes (no ROS graph needed). Writes
docs/assets/benchmark.json; run scripts/plot_benchmark.py afterwards to refresh
the SVG charts.

Example:
    pixi run -e sam3 python scripts/benchmark.py \
        --image /path/to/frame.jpg --prompt puddle
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3_ros.config import EngineConfig, InferenceConfig, ModelConfig
from sam3_ros.sam3_engine import Sam3Engine

SIZES = [
    (320, 240),
    (640, 480),
    (960, 540),
    (1280, 720),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
]


def _driver_version() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="source RGB image")
    parser.add_argument("--prompt", default="puddle", help="text prompt")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument(
        "--out",
        default="docs/assets/benchmark.json",
        help="output JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Image.open(args.image).convert("RGB")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    config = EngineConfig(
        model=ModelConfig(device="cuda", load_from_hf=True),
        inference=InferenceConfig(
            precision=args.precision,
            warmup=True,
            warmup_iterations=1,
            warmup_prompt=args.prompt,
        ),
    )
    engine = Sam3Engine(config)
    started = time.perf_counter()
    engine.initialize()
    load_s = time.perf_counter() - started
    engine.precompute_text(args.prompt)
    torch.cuda.synchronize()
    base_alloc = torch.cuda.memory_allocated() / 2**20
    base_res = torch.cuda.memory_reserved() / 2**20
    free, total = torch.cuda.mem_get_info()
    base_used = (total - free) / 2**20

    rows = []
    for width, height in SIZES:
        rgb = np.asarray(base.resize((width, height), Image.LANCZOS))
        for _ in range(2):
            engine.segment(rgb, args.prompt, use_text_cache=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        result = None
        for _ in range(args.iterations):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            result = engine.segment(rgb, args.prompt, use_text_cache=True)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - tick) * 1e3)
        times.sort()
        peak_alloc = torch.cuda.max_memory_allocated() / 2**20
        peak_res = torch.cuda.max_memory_reserved() / 2**20
        free, total = torch.cuda.mem_get_info()
        row = {
            "width": width,
            "height": height,
            "megapixels": round(width * height / 1e6, 3),
            "mean_ms": round(statistics.mean(times), 2),
            "p50_ms": round(statistics.median(times), 2),
            "p95_ms": round(times[int(0.95 * (len(times) - 1))], 2),
            "min_ms": round(times[0], 2),
            "max_ms": round(times[-1], 2),
            "fps": round(1000.0 / statistics.mean(times), 2),
            "peak_allocated_mib": round(peak_alloc, 0),
            "peak_reserved_mib": round(peak_res, 0),
            "inference_alloc_delta_mib": round(peak_alloc - base_alloc, 0),
            "device_used_mib": round((total - free) / 2**20, 0),
            "instances": result.num_instances,
            "fg_pixels": int(result.mask.sum()),
        }
        rows.append(row)
        print(row)

    payload = {
        "meta": {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_mib": round(total / 2**20, 0),
            "driver": _driver_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "precision": f"{args.precision} autocast",
            "resolution": config.inference.resolution,
            "confidence_threshold": config.inference.confidence_threshold,
            "prompt": args.prompt,
            "model_load_s": round(load_s, 1),
            "baseline_allocated_mib": round(base_alloc, 0),
            "baseline_reserved_mib": round(base_res, 0),
            "baseline_device_used_mib": round(base_used, 0),
            "iterations": args.iterations,
            "image": str(args.image),
        },
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    engine.close()


if __name__ == "__main__":
    main()

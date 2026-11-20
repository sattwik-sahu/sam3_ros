---
title: Parameters
description: Nested YAML parameters for the SAM 3 server and streaming nodes.
---

# Parameters

Each node has its own file, loaded **before** the model is constructed:

* [`config/sam3_server.yaml`](https://github.com/sattwik-sahu/sam3_ros/blob/main/src/sam3_ros/config/sam3_server.yaml)
* [`config/sam3_stream.yaml`](https://github.com/sattwik-sahu/sam3_ros/blob/main/src/sam3_ros/config/sam3_stream.yaml)

Nested YAML maps to dotted ROS names: `model.device`, `inference.precision`, …
Every key is declared with a descriptor, so `ros2 param describe` documents the
running node.

## Model

| Key | Default | Purpose |
| --- | --- | --- |
| `model.checkpoint_path` | `""` | Explicit `.pt` (or directory); skips Hugging Face. |
| `model.device` | `cuda` | `cuda`, `cuda:N`, `cpu`. |
| `model.eval_mode` | `true` | Inference mode. |
| `model.load_from_hf` | `true` | Resolve/download the gated checkpoint. |
| `model.hf_repo_id` | `facebook/sam3` | Weights repository. |
| `model.hf_filename` | `sam3.pt` | Checkpoint filename. |
| `model.hf_revision` | `""` | Branch/tag/commit; empty = default. |
| `model.local_files_only` | `false` | Offline cache-only resolution. |
| `model.bpe_path` | `""` | Override the bundled BPE vocabulary. |
| `model.compile` | `false` | `torch.compile` (slow first call). |
| `model.enable_segmentation` | `true` | Required for masks. |
| `model.enable_inst_interactivity` | `false` | Point/box predictor; extra VRAM. |
| `model.allow_cpu_fallback` | `true` | Warn + CPU if CUDA is unavailable. |

## Inference

| Key | Default | Purpose |
| --- | --- | --- |
| `inference.resolution` | `1008` | Encoder input; matches released weights. |
| `inference.confidence_threshold` | `0.5` | Default score cutoff. |
| `inference.precision` | `bf16` | `bf16`, `fp16`, `fp32`. |
| `inference.tf32` | `true` | TF32 matmul/cuDNN. |
| `inference.cudnn_benchmark` | `true` | cuDNN autotuning. |
| `inference.cache_text_embeddings` | `true` | Reuse prompt embeddings. |
| `inference.text_cache_size` | `8` | LRU size; `0` = keep one prompt. |
| `inference.warmup` | `true` | Synthetic forward(s) at startup. |
| `inference.warmup_iterations` | `1` | Warmup count. |
| `inference.warmup_prompt` | `object` | Server warmup prompt; the stream overrides it with its own. |

## Server

| Key | Default | Purpose |
| --- | --- | --- |
| `server.service_name` | `segment` | `~/segment` = private. |
| `server.stats_log_period_s` | `30.0` | Aggregate stats; `0` disables. |

## Streaming

| Key | Default | Purpose |
| --- | --- | --- |
| `stream.text_prompt` | `object` | Fixed prompt. |
| `stream.image_topic` | `/camera/color/image_raw` | Input. |
| `stream.mask_topic` | `/sam3/mask` | Output mono8 mask. |
| `stream.max_rate_hz` | `0.0` | Processing cap; `0` unlimited. |
| `stream.queue_size` | `1` | Frame queue depth. |
| `stream.drop_stale_frames` | `true` | Drop oldest on overflow. |
| `stream.input_qos_reliability` | `best_effort` | `best_effort` / `reliable`. |
| `stream.input_qos_depth` | `5` | Input history. |
| `stream.output_qos_reliability` | `reliable` | Output reliability. |
| `stream.output_qos_depth` | `1` | Keep latest mask only. |
| `stream.stats_log_period_s` | `30.0` | Throughput/drop/latency log. |

## Inspect and override

```bash
ros2 param list /sam3_server
ros2 param describe /sam3_server model.device
ros2 param get /sam3_server inference.precision

ros2 launch sam3_ros sam3_server.launch.py confidence_threshold:=0.4
```

Precedence: **launch override → YAML file → declared default**.

!!! tip "Docker"
    The config directory is bind-mounted at `/config`. Edit the YAML on the
    host, then `docker compose down` / `up` the service. No rebuild needed.

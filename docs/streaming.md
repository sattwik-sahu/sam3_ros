---
title: Streaming node
description: Fixed-prompt SAM 3 segmentation of an RGB image topic into a mono8 mask topic.
---

# Streaming node

`sam3_stream` subscribes to an RGB image topic, segments every frame with a
prompt fixed at startup, and publishes a `mono8` mask.

## Data flow

```text
sensor_msgs/Image ──▶ callback ──▶ queue(depth 1) ──▶ GPU worker ──▶ /sam3/mask
   (drop stale)                     newest wins        cached prompt      mono8, 0/1
```

## Topics

| Direction | Default | Type | Notes |
| --- | --- | --- | --- |
| in | `/camera/color/image_raw` | `sensor_msgs/Image` | RGB/BGR/RGBA/BGRA/mono8 accepted. |
| out | `/sam3/mask` | `sensor_msgs/Image` | `mono8`, `{0,1}`, exact input W×H, input header. |

Input QoS defaults to `best_effort` (cameras), output to `reliable`. Both are
parameters.

## Run

```bash
pixi run -e sam3 ros2 launch sam3_ros sam3_stream.launch.py \
    image_topic:=/camera/color/image_raw text_prompt:="red mug"
```

Smoke test with a still image:

```bash
pixi run -e sam3 python examples/image_publisher.py \
    --image frame.jpg --topic /camera/color/image_raw --rate-hz 5
```

## Launch arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `config_file` | packaged `sam3_stream.yaml` | Full nested parameter file. |
| `image_topic` | YAML | Input image topic. |
| `mask_topic` | YAML (`/sam3/mask`) | Output mask topic. |
| `text_prompt` | YAML (`object`) | **Fixed** prompt for the whole stream. |
| `max_rate_hz` | YAML (`0`) | Cap processing; `0` = as fast as the GPU. |
| `queue_size` | YAML (`1`) | Callback → worker queue depth. |
| `device`, `checkpoint_path`, `precision`, `confidence_threshold` | YAML | Model/inference overrides. |
| `use_sim_time`, `log_level`, `respawn`, `respawn_delay` | — | Standard ROS controls. |

`--show-args` lists descriptions. The prompt is **not** a dynamic parameter: it
is encoded once during warmup; restart the node to change it.

## Tuning

| Goal | Setting |
| --- | --- |
| Lowest latency under load | `queue_size: 1`, `drop_stale_frames: true` |
| Process every frame | `queue_size` > 1, `drop_stale_frames: false` |
| Protect other GPU work | `max_rate_hz: 5.0` |
| Less VRAM | `precision: fp16`, `model.enable_inst_interactivity: false` |
| Max throughput | keep `cache_text_embeddings: true`, `precision: bf16` |

Benchmarks: [Benchmarks](benchmarks.md).

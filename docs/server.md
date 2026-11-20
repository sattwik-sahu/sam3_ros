---
title: Server node
description: Request/response SAM 3 segmentation service with a per-request text prompt.
---

# Server node

`sam3_server` exposes one ROS 2 service. The model is loaded once at startup;
each request supplies its own prompt.

## Interface

`sensor_msgs/Image` + prompt in, `mono8` mask out.

```text
# sam3_msgs/srv/SegmentImage
sensor_msgs/Image image
string text_prompt
float32 confidence_threshold      # < 0 -> inference.confidence_threshold
---
sensor_msgs/Image mask            # mono8, {0,1}, same W×H + header as input
bool success
string message
```

* `mask` is the union of all SAM 3 instance masks above the threshold.
* `encoding` is always `mono8`; background is `0`, foreground is `1`.
* The header (stamp, `frame_id`) is copied from the request image.
* `success=false` means `mask` is invalid; `message` explains why.

## Run

```bash
pixi run -e sam3 ros2 launch sam3_ros sam3_server.launch.py

# override YAML values without editing it
pixi run -e sam3 ros2 launch sam3_ros sam3_server.launch.py \
    device:=cuda:1 confidence_threshold:=0.35 log_level:=debug
```

Default service name: `segment` (root namespace). The launch file loads
`config/sam3_server.yaml` **before** the model is built.

## Call it

```bash
pixi run -e sam3 python examples/segment_client.py \
    --image frame.jpg --prompt "red mug" --output mask.png
```

The client uses `cv_bridge`, calls `/segment`, prints foreground pixels and can
save the mask. See
[`examples/segment_client.py`](https://github.com/sattwik-sahu/sam3_ros/blob/main/examples/segment_client.py).

## Launch arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `config_file` | packaged `sam3_server.yaml` | Full nested parameter file. |
| `node_name` / `namespace` | `sam3_server` / root | ROS naming. |
| `device` | YAML (`cuda`) | `cuda`, `cuda:1`, `cpu`. |
| `checkpoint_path` | YAML (`""`) | Explicit `.pt`; skips Hugging Face. |
| `load_from_hf` / `local_files_only` | YAML | Checkpoint resolution. |
| `precision` | YAML (`bf16`) | `bf16`, `fp16`, `fp32`. |
| `resolution` | YAML (`1008`) | Encoder resolution (released weights: 1008). |
| `confidence_threshold` | YAML (`0.5`) | Default threshold. |
| `compile` | YAML (`false`) | `torch.compile`. |
| `service_name` | YAML (`segment`) | Service name; `~/segment` makes it private. |
| `use_sim_time`, `log_level`, `respawn`, `respawn_delay` | — | Standard ROS controls. |

`ros2 launch sam3_ros sam3_server.launch.py --show-args` prints the same list
with descriptions.

## Runtime behaviour

* GPU access is serialized with a re-entrant lock; concurrent clients cannot
  trigger concurrent forward passes or OOM.
* Prompt embeddings are cached in an LRU (`inference.text_cache_size`), so
  repeated prompts skip the text encoder.
* `server.stats_log_period_s` prints requests/failures/average latency.
* Every parameter is declared with a descriptor:
  `ros2 param describe /sam3_server model.device`.

Parameters: [Parameters](parameters.md). Docker: [Docker](docker.md).

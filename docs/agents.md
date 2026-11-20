---
title: For agents
description: Orientation map, invariants and commands for coding agents working on SAM 3 ROS.
---

# For agents

The canonical file is [`AGENTS.md`](https://github.com/sattwik-sahu/sam3_ros/blob/main/AGENTS.md)
at the repository root. This page mirrors the essentials.

## Orientation

```text
sam3_msgs/srv/SegmentImage.srv     one service: image + prompt -> mono8 mask
sam3_ros/sam3_ros/
  sam3_engine.py                   model load, autocast, prompt LRU, mask union
  sam3_server_node.py              service adapter + stats
  sam3_stream_node.py              topic adapter + worker queue
  parameters.py                    ParameterDescriptor specs + validation
  image_utils.py                   cv_bridge conversion
  download_weights.py              HF prefetch
config/, launch/                   runtime + startup surface
compose.sam3.yaml                  include-able server/stream/weights services
docs/                              Zensical site (separate pixi env)
```

## Invariants

* Mask is `mono8`, `{0,1}`, **input W×H**, header copied; no boxes.
* Every inference call goes through `Sam3Engine` and its lock.
* Streaming prompt is fixed; embedding is cached after startup.
* Params → validation → model load, in that order.
* Documentation tooling stays in the `docs` environment.
* Weights live in the `sam3-weights` volume, never in the image.
* Do not build with `--symlink-install`.

## Commands

```bash
pixi run -e sam3 build
pixi run -e sam3 colcon test --packages-select sam3_ros sam3_msgs
pixi run -e docs docs-build
pixi run -e sam3 python scripts/benchmark.py --image frame.jpg --prompt puddle
```

## Verify a change

```bash
pixi run -e sam3 ros2 launch sam3_ros sam3_server.launch.py &
pixi run -e sam3 python examples/segment_client.py --image f.jpg --prompt thing
# success=true, encoding=mono8, W/H match, unique(mask) subset of {0,1}
```

## Gotchas and where-to-edit tables

See the root [`AGENTS.md`](https://github.com/sattwik-sahu/sam3_ros/blob/main/AGENTS.md).

---
title: Docker
description: Include-able Compose fragment, persistent weight volume and runtime YAML parameters.
---

# Docker

`compose.sam3.yaml` is an **include-able** Compose fragment, not a standalone
application. It declares three services behind profiles.

| Service | Profile | Purpose |
| --- | --- | --- |
| `sam3-server` | `server` | Request/response node. |
| `sam3-stream` | `stream` | Fixed-prompt streaming node. |
| `sam3-download` | `weights` | One-shot checkpoint prefetch into the volume. |

```yaml
include:
  - path: compose.sam3.yaml
```

A ready main file is included as `compose.example.yaml`.

## Build and start

```bash
echo "HF_TOKEN=hf_..." > .env

# once: populate the persistent volume
docker compose -f compose.example.yaml --profile weights run --rm sam3-download

# server
docker compose -f compose.example.yaml --profile server up -d --build sam3-server

# streaming (prompt via env or edit the YAML)
SAM3_TEXT_PROMPT="red mug" SAM3_IMAGE_TOPIC=/camera/color/image_raw \
  docker compose -f compose.example.yaml --profile stream up -d --build sam3-stream

docker compose -f compose.example.yaml logs -f sam3-server
```

## Persistent weights

The named volume `sam3-weights` is mounted at `/models` and `HF_HOME` points to
`/models/huggingface`. The first start downloads the gated checkpoint; every
later container start reuses it. No weights are baked into the image.

```bash
docker volume inspect sam3-weights
docker compose --profile weights run --rm sam3-download   # explicit prefetch
```

Set `SAM3_PREFETCH_WEIGHTS=1` to make a node service warm the cache in its
entrypoint instead.

## Runtime parameters

`SAM3_CONFIG_DIR` (default `./src/sam3_ros/config`) is bind-mounted read-only at
`/config`. Edit the YAML on the host and recreate the service:

```bash
$EDITOR src/sam3_ros/config/sam3_server.yaml
docker compose --profile server down
docker compose --profile server up -d sam3-server
```

No rebuild. Launch-argument overrides (`SAM3_DEVICE`, `SAM3_TEXT_PROMPT`, …)
work the same way: change the environment, recreate the service.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `HF_TOKEN` | — | Gated model access. |
| `SAM3_IMAGE` | `sam3-ros:jazzy` | Image tag. |
| `SAM3_DEVICE` | `cuda` | `model.device` override. |
| `SAM3_TEXT_PROMPT` | YAML | Streaming prompt override. |
| `SAM3_IMAGE_TOPIC` | YAML | Streaming input override. |
| `SAM3_CONFIG_DIR` | `./src/sam3_ros/config` | Host config dir → `/config`. |
| `SAM3_WEIGHTS_VOLUME` | `sam3-weights` | Volume name. |
| `SAM3_GPUS` | `all` | `'"device=0"'` to pin one GPU. |
| `SAM3_NETWORK_MODE` | `host` | DDS-friendly default. |
| `SAM3_SHM_SIZE` | `2gb` | Shared memory. |
| `SAM3_CUDA_VERSION` | `12.9` | Build-time conda CUDA override. |
| `SAM3_PREFETCH_WEIGHTS` | `0` | `1` = warm cache in entrypoint. |
| `ROS_DOMAIN_ID` | `0` | DDS domain. |
| `RMW_IMPLEMENTATION` | `rmw_fastrtps_cpp` | Middleware. |

## Notes

* The image installs the lean `sam3-runtime` pixi environment (`ros-base`, `cv_bridge`, CUDA PyTorch, SAM 3) instead of the `sam3` desktop dev environment. This keeps the image smaller and avoids RViz/Gazebo.

* `gpus: all` is set per node service; the download service needs no GPU.
* `network_mode: host` lets host ROS 2 processes discover the container. Use the
  same `ROS_DOMAIN_ID`.
* The image sets `CONDA_OVERRIDE_CUDA=12.9` so the CUDA environment resolves
  during build (no GPU) and inside the container. PyTorch still uses the real
  driver via the NVIDIA container runtime.
* Healthchecks wait up to 15 minutes for the first weight download.

[Install](install.md) · [Parameters](parameters.md) · [Benchmarks](benchmarks.md)

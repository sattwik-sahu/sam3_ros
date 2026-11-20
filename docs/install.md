---
title: Install
description: Install the SAM 3 ROS 2 workspace with pixi or Docker.
---

# Install

## Prerequisites

| Requirement | Notes |
| --- | --- |
| Linux x86_64 | The `sam3` pixi environment is linux-64 only. |
| NVIDIA GPU | ~4 GB free VRAM for the model; headroom for your cameras/other nodes. |
| Driver | CUDA 12.9+ driver (tested with 595.99.02). |
| [pixi](https://pixi.sh) | Installs ROS 2 Jazzy, CUDA PyTorch and SAM 3 from `pixi.lock`. |
| Hugging Face access | Accept the license at [facebook/sam3](https://huggingface.co/facebook/sam3). |
| [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/) | Only for Docker. |

## 1. Get the code

```bash
git clone git@github.com:sattwik-sahu/sam3_ros.git
cd sam3_ros
```

## 2. Resolve the environment

```bash
pixi install -e sam3     # ROS 2 Jazzy + CUDA PyTorch + official SAM 3 git package
pixi run -e sam3 build   # colcon: sam3_msgs + sam3_ros
```

The docs site has its own tiny environment — it does not pull ROS or CUDA:

```bash
pixi install -e docs
pixi run -e docs docs-serve   # http://localhost:8000
```

## 3. Authenticate for the weights

The checkpoint is gated. Export a token (or run `hf auth login`):

```bash
export HF_TOKEN=hf_...
pixi run -e sam3 python -m sam3_ros.download_weights
```

Weights land in the standard Hugging Face cache (`HF_HOME`, default
`~/.cache/huggingface`). The Docker deployment stores them in a named volume.

## 4. Smoke test

```bash
pixi run -e sam3 ros2 launch sam3_ros sam3_server.launch.py &
pixi run -e sam3 python examples/segment_client.py \
    --image /path/to/frame.jpg --prompt "red mug" --output /tmp/mask.png
```

Expected: a `mono8` PNG at the input resolution containing only `0` and `255`
(the example writes `mask * 255`; the ROS message itself is `0/1`).

## Verify the mask contract

The included client prints foreground statistics and writes the mask. The
authoritative contract is documented in [Server node](server.md); the complete
client is
[`examples/segment_client.py`](https://github.com/sattwik-sahu/sam3_ros/blob/main/examples/segment_client.py).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `401` / gated repo | Set `HF_TOKEN` with access to `facebook/sam3`, then restart the node/container. |
| `CUDA device requested but torch.cuda.is_available() is false` | Install a CUDA driver or set `model.allow_cpu_fallback: true` (slow). |
| `__cuda` solve error in Docker | The image sets `CONDA_OVERRIDE_CUDA`; rebuild with `SAM3_CUDA_VERSION=12.9`. |
| `colcon build` fails with `--editable` | Do not use `--symlink-install` with this setuptools; the pixi build task already omits it. |
| Two nodes OOM on one GPU | Run one profile at a time or pin them with `SAM3_GPUS='"device=0"'` / `cuda:1`. |
| No service/topics across containers | Use `network_mode: host` (default) and the same `ROS_DOMAIN_ID`. |

Next: [Server node](server.md) · [Streaming node](streaming.md) · [Docker](docker.md).

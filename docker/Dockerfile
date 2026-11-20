# syntax=docker/dockerfile:1.7
#
# Reproducible SAM 3 ROS 2 image.
#
# Built from the same pixi lock file used for local development, so ROS 2 Jazzy,
# CUDA PyTorch and the official SAM 3 package are identical inside and outside
# the container.
#
# SAM 3 weights are *not* baked into the image. They are downloaded on first use
# into the named volume mounted at /models (see compose.sam3.yaml).

ARG PIXI_VERSION=0.84.2
ARG CUDA_VERSION=12.9

# Reuse the official pixi binary instead of downloading it from GitHub releases
# at build time (more reliable in CI and air-gapped builders).
FROM ghcr.io/prefix-dev/pixi:${PIXI_VERSION} AS pixi

FROM ubuntu:24.04

ARG CUDA_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIXI_HOME=/opt/pixi \
    PATH=/opt/pixi/bin:${PATH} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/models/huggingface

# Runtime libraries needed by OpenCV/Pillow/PyTorch. CUDA itself comes from the
# conda packages or the host NVIDIA driver injected at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Lets pixi/conda resolve the CUDA build at build time (no GPU present) and at
# runtime even when nvidia-smi is not installed in the image. PyTorch still
# talks to the real driver through the NVIDIA container runtime.
ENV CONDA_OVERRIDE_CUDA=${CUDA_VERSION}

COPY --from=pixi /usr/local/bin/pixi /usr/local/bin/pixi
RUN pixi --version

WORKDIR /workspace

# Dependency layer: only invalidated when pixi.toml / pixi.lock change.
COPY pixi.toml pixi.lock ./
RUN pixi install --locked -e sam3 \
    && rm -rf /opt/pixi/cache /root/.cache/rattler /root/.cache/uv

# Workspace layer.
COPY src ./src
COPY examples ./examples
COPY docker ./docker
RUN pixi run --locked -e sam3 build \
    && rm -rf build log

RUN mkdir -p /models \
    && cp docker/sam3-shell /usr/local/bin/sam3-shell \
    && cp docker/sam3-entrypoint /usr/local/bin/sam3-entrypoint \
    && chmod +x /usr/local/bin/sam3-shell /usr/local/bin/sam3-entrypoint

ENTRYPOINT ["/usr/local/bin/sam3-entrypoint"]
CMD ["ros2", "launch", "sam3_ros", "sam3_server.launch.py", \
     "config_file:=/config/sam3_server.yaml"]

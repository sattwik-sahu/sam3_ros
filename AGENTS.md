# AGENTS — SAM 3 ROS

Read this before changing code. Human docs: <https://sattwik-sahu.github.io/sam3_ros/>.

## What this repo is

ROS 2 Jazzy wrapper around Meta SAM 3. Two nodes share one inference engine:

```
src/sam3_msgs/srv/SegmentImage.srv      service contract
src/sam3_ros/sam3_ros/sam3_engine.py    model load, BF16 inference, prompt cache, mask union
src/sam3_ros/sam3_ros/sam3_server_node.py  SegmentImage service
src/sam3_ros/sam3_ros/sam3_stream_node.py  image topic -> mask topic (fixed prompt)
src/sam3_ros/sam3_ros/parameters.py     declared + described + validated params
src/sam3_ros/config/*.yaml              runtime params (nested)
src/sam3_ros/launch/*.launch.py         launch args (documented)
compose.sam3.yaml                       include-able Docker services
docs/                                   Zensical site; benchmarks + examples
```

Start with `docs/index.md` for the architecture and `sam3_engine.py` for the
inference path. `docs/agents.md` has the long version.

## Non-negotiable invariants

1. **Mask contract.** Output is `sensor_msgs/Image`, `encoding='mono8'`, exactly
   the input width/height, values `{0,1}` (`1 = foreground`), header copied from
   input. No bounding boxes / detection lists.
2. **One GPU path.** All inference goes through `Sam3Engine` under its lock.
   Do not call SAM 3 directly from a ROS callback.
3. **Prompt caching.** Streaming encodes `stream.text_prompt` once at startup.
   Changing the prompt requires a node restart.
4. **Params before model.** YAML is loaded by the launch file, then params are
   read/validated, then `Sam3Engine.initialize()`.
5. **Docs env is separate.** Zensical lives only in the `docs` pixi feature.
   Never add it (or other doc tooling) to the `sam3` environment.
6. **Weights are not in the image.** They live in the `sam3-weights` volume
   (`/models`, `HF_HOME=/models/huggingface`).
7. **Container env is `sam3-runtime`.** The Docker image uses `ros-base`; the local dev env `sam3` keeps `ros-desktop`. Do not make the image depend on desktop-only packages.
8. **Build without `--symlink-install`.** setuptools + colcon's prefix override
   breaks `setup.py develop --editable`; the pixi build task already omits it.

## Commands

```bash
pixi install -e sam3
pixi run -e sam3 build
pixi run -e sam3 colcon test --packages-select sam3_ros sam3_msgs
pixi run -e sam3 ros2 launch sam3_ros sam3_server.launch.py
pixi run -e sam3 ros2 launch sam3_ros sam3_stream.launch.py text_prompt:="red mug"

pixi install -e docs
pixi run -e docs docs-build     # site -> site/
pixi run -e docs docs-serve     # localhost:8000

pixi run -e sam3 python scripts/benchmark.py --image frame.jpg --prompt puddle
pixi run -e sam3 python scripts/plot_benchmark.py
```

## Where to change what

| Task | Files |
| --- | --- |
| Add/rename a parameter | `parameters.py` (spec + loader + validation), both YAMLs, `docs/parameters.md`, launch overrides if common |
| Add a launch argument | matching `launch/*.launch.py` (`DeclareLaunchArgument` + override map), docs table |
| Change service fields | `SegmentImage.srv`, server node, examples client, `docs/server.md` |
| Tune performance | `sam3_engine.py` (autocast/TF32/text cache), `parameters.py`, `config/*.yaml` |
| Change container behaviour | `docker/Dockerfile`, `docker/sam3-*`, `compose.sam3.yaml`, `docs/docker.md` |
| Change docs | `docs/`, `zensical.toml`, `docs/assets/`; workflow `.github/workflows/docs.yml` |

## Gotchas

* `pixi.toml` overrides SAM 3's `numpy<2` pin (ROS 2 Jazzy is NumPy 2). Keep the
  override; the SAM 3 code is NumPy-2 compatible.
* The `sam3` env uses the named platform `linux-64-cuda129`; `__cuda` must be
  declared on the platform (pixi 0.84 rich platforms) for CUDA builds.
* PyTorch is conda `pytorch-gpu` (CUDA 12.9), not a PyPI wheel; the PyPI CUDA
  wheels are ~2.5 GB and slow to fetch.
* Gated weights need `HF_TOKEN` at runtime; never bake the token into an image.
* Keep docs images small; use `scripts/` generators rather than checked-in
  multi-megabyte raw frames.

## Definition of done

* `colcon build` clean; `test_mask_contract.py` passes.
* Live service check: `success`, `encoding=mono8`, `W/H` match input,
  `np.unique(mask) ⊆ {0,1}`.
* `pixi run -e docs docs-build` succeeds and the new page is in `nav`.
* No changes to the main env from docs tooling.

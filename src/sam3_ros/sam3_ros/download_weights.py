# Copyright (c) 2026 moonlab
# SPDX-License-Identifier: MIT
"""Prefetch the gated SAM 3 weights into a persistent cache or directory.

The Docker deployment mounts a named volume at ``/models`` and sets
``HF_HOME=/models/huggingface``.  Running this module inside that container
(e.g. the ``sam3-download`` compose service) populates the volume once, after
which every node start resolves the checkpoint from the local cache without
downloading anything.

Examples:
    # Warm the standard Hugging Face cache (respects HF_HOME / HF_TOKEN)
    python -m sam3_ros.download_weights

    # Store a stable copy on a mounted volume
    python -m sam3_ros.download_weights --output-dir /models/sam3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the SAM 3 checkpoint into a local/HF cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        default="facebook/sam3",
        help="Hugging Face repository holding the gated weights.",
    )
    parser.add_argument(
        "--filename",
        action="append",
        dest="filenames",
        help="File(s) to download. May be repeated. Defaults to sam3.pt.",
    )
    parser.add_argument(
        "--revision",
        default="",
        help="Optional branch, tag or commit of the repository.",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Hugging Face cache directory. Empty uses HF_HOME or the default.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Copy the files into this directory (e.g. /models/sam3) instead of "
        "only warming the cache.",
    )
    parser.add_argument(
        "--token",
        default="",
        help="Hugging Face access token. Empty uses HF_TOKEN or the stored "
        "huggingface-cli login token.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Never hit the network; only resolve already-cached files.",
    )
    parser.add_argument(
        "--include-config",
        action="store_true",
        help="Also fetch config.json next to the checkpoint.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Download the checkpoint and print the resolved paths."""
    args = _parse_args(argv)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "huggingface_hub is not installed. Install the 'sam3' pixi feature "
            "or run `pixi add --pypi huggingface_hub`.",
            file=sys.stderr,
        )
        return 1

    filenames = args.filenames or ["sam3.pt"]
    if args.include_config and "config.json" not in filenames:
        filenames.append("config.json")

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )
    paths: list[str] = []
    for filename in filenames:
        kwargs: dict[str, object] = {
            "repo_id": args.repo_id,
            "filename": filename,
            "local_files_only": args.local_files_only,
        }
        if args.revision:
            kwargs["revision"] = args.revision
        if args.cache_dir:
            kwargs["cache_dir"] = args.cache_dir
        if args.output_dir:
            kwargs["local_dir"] = args.output_dir
        if token:
            kwargs["token"] = token
        print(f"Resolving {args.repo_id}/{filename} ...", flush=True)
        path = hf_hub_download(**kwargs)
        paths.append(str(path))
        print(f"  -> {path}", flush=True)

    if args.output_dir:
        output = Path(os.path.expanduser(args.output_dir))
        print(f"\nWeights available in {output}", flush=True)
        print(
            "Point model.checkpoint_path at this directory (or at sam3.pt) and "
            "set model.load_from_hf: false to use them.",
            flush=True,
        )
    else:
        print(
            "\nCache warmed. The nodes resolve this checkpoint with "
            "model.load_from_hf: true.",
            flush=True,
        )
    return 0 if paths else 1


if __name__ == "__main__":
    sys.exit(main())

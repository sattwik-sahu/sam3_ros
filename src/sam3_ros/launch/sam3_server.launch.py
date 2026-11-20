# Copyright (c) 2026 moonlab
# SPDX-License-Identifier: MIT
"""Launch file for the request/response SAM 3 segmentation server.

Usage::

    ros2 launch sam3_ros sam3_server.launch.py
    ros2 launch sam3_ros sam3_server.launch.py \
        config_file:=/path/to/sam3_server.yaml device:=cuda:1
    ros2 launch sam3_ros sam3_server.launch.py --show-args

The YAML file configures the model *before* it is loaded.  Launch arguments are
optional overrides: an empty value keeps whatever the YAML file specifies, so
you can keep one parameter file per robot and only override the device or log
level on the command line.

Arguments:
    config_file:           YAML parameter file (default: package config).
    node_name:             ROS node name. Defaults to ``sam3_server``.
    namespace:             ROS namespace for the node and service.
    use_sim_time:          Use /clock for message timestamps.
    log_level:             ROS log level (debug, info, warn, error, fatal).
    device:                Torch device, e.g. ``cuda``, ``cuda:1`` or ``cpu``.
    checkpoint_path:       Explicit .pt checkpoint path (skips Hugging Face).
    load_from_hf:          Download/resolve the checkpoint from Hugging Face.
    local_files_only:      Resolve the checkpoint from the local HF cache only.
    precision:             Autocast precision: bf16, fp16 or fp32.
    resolution:            Square encoder resolution (1008 for released weights).
    confidence_threshold:  Default detection threshold in [0, 1].
    compile:               Enable torch.compile for the model.
    service_name:          Name of the SegmentImage service.
    respawn:               Restart the node if it exits unexpectedly.
    respawn_delay:         Seconds between respawns.
"""

from __future__ import annotations

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

ARGUMENTS = [
    DeclareLaunchArgument(
        "config_file",
        default_value=PathJoinSubstitution(
            [FindPackageShare("sam3_ros"), "config", "sam3_server.yaml"]
        ),
        description=(
            "YAML parameter file. It is loaded before the model is built and "
            "may contain the full nested model/inference/server configuration."
        ),
    ),
    DeclareLaunchArgument(
        "node_name",
        default_value="sam3_server",
        description="ROS node name.",
    ),
    DeclareLaunchArgument(
        "namespace",
        default_value="",
        description="ROS namespace (empty = root namespace).",
    ),
    DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulated time from /clock for headers and timers.",
    ),
    DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="ROS log level: debug, info, warn, error or fatal.",
    ),
    DeclareLaunchArgument(
        "device",
        default_value="",
        description=(
            "Override model.device (e.g. cuda, cuda:1, cpu). Empty keeps YAML."
        ),
    ),
    DeclareLaunchArgument(
        "checkpoint_path",
        default_value="",
        description=(
            "Override model.checkpoint_path (absolute .pt path). Empty keeps "
            "YAML."
        ),
    ),
    DeclareLaunchArgument(
        "load_from_hf",
        default_value="",
        description=(
            "Override model.load_from_hf (true/false). Empty keeps YAML."
        ),
    ),
    DeclareLaunchArgument(
        "local_files_only",
        default_value="",
        description=(
            "Override model.local_files_only (true/false). Empty keeps YAML."
        ),
    ),
    DeclareLaunchArgument(
        "precision",
        default_value="",
        description="Override inference.precision (bf16, fp16, fp32).",
    ),
    DeclareLaunchArgument(
        "resolution",
        default_value="",
        description="Override inference.resolution (integer).",
    ),
    DeclareLaunchArgument(
        "confidence_threshold",
        default_value="",
        description="Override inference.confidence_threshold in [0, 1].",
    ),
    DeclareLaunchArgument(
        "compile",
        default_value="",
        description="Override model.compile (true/false). Empty keeps YAML.",
    ),
    DeclareLaunchArgument(
        "service_name",
        default_value="",
        description="Override server.service_name. Empty keeps YAML.",
    ),
    DeclareLaunchArgument(
        "respawn",
        default_value="false",
        description="Restart the node if it exits unexpectedly.",
    ),
    DeclareLaunchArgument(
        "respawn_delay",
        default_value="2.0",
        description="Seconds to wait before respawning.",
    ),
]


def _optional(context: LaunchContext, name: str) -> str:
    return LaunchConfiguration(name).perform(context).strip()


def _bool(value: str, name: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(
        f"Launch argument {name!r} expects a boolean, got {value!r}"
    )


def _launch_setup(context: LaunchContext, *args, **kwargs):
    config_file = _optional(context, "config_file")
    if not config_file:
        raise RuntimeError("config_file must point to a YAML parameter file")

    overrides: dict[str, ParameterValue] = {
        "use_sim_time": ParameterValue(
            _bool(_optional(context, "use_sim_time"), "use_sim_time"),
            value_type=bool,
        )
    }

    string_overrides = {
        "device": "model.device",
        "precision": "inference.precision",
        "service_name": "server.service_name",
    }
    for argument, parameter in string_overrides.items():
        value = _optional(context, argument)
        if value:
            overrides[parameter] = ParameterValue(value)

    checkpoint = _optional(context, "checkpoint_path")
    if checkpoint:
        overrides["model.checkpoint_path"] = ParameterValue(checkpoint)

    boolean_overrides = {
        "load_from_hf": "model.load_from_hf",
        "local_files_only": "model.local_files_only",
        "compile": "model.compile",
    }
    for argument, parameter in boolean_overrides.items():
        value = _optional(context, argument)
        if value:
            overrides[parameter] = ParameterValue(
                _bool(value, argument), value_type=bool
            )

    resolution = _optional(context, "resolution")
    if resolution:
        overrides["inference.resolution"] = ParameterValue(
            int(resolution), value_type=int
        )

    threshold = _optional(context, "confidence_threshold")
    if threshold:
        overrides["inference.confidence_threshold"] = ParameterValue(
            float(threshold), value_type=float
        )

    arguments: list[str] = []
    log_level = _optional(context, "log_level")
    if log_level:
        arguments = ["--ros-args", "--log-level", log_level]

    node = Node(
        package="sam3_ros",
        executable="sam3_server",
        name=_optional(context, "node_name") or "sam3_server",
        namespace=_optional(context, "namespace"),
        parameters=[config_file, overrides],
        arguments=arguments,
        output="screen",
        emulate_tty=True,
        respawn=_bool(_optional(context, "respawn"), "respawn"),
        respawn_delay=float(_optional(context, "respawn_delay") or 2.0),
    )
    return [node]


def generate_launch_description() -> LaunchDescription:
    """Build the launch description."""
    return LaunchDescription([*ARGUMENTS, OpaqueFunction(function=_launch_setup)])

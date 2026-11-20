# Copyright (c) 2026 moonlab
# SPDX-License-Identifier: MIT
"""Launch file for the streaming SAM 3 segmentation node.

The text prompt is fixed when the node starts (``text_prompt``), its embedding
is cached, and every incoming frame is segmented against it.

Usage::

    ros2 launch sam3_ros sam3_stream.launch.py \
        image_topic:=/camera/color/image_raw text_prompt:="red mug"
    ros2 launch sam3_ros sam3_stream.launch.py --show-args

Arguments:
    config_file:           YAML parameter file (default: package config).
    node_name:             ROS node name. Defaults to ``sam3_stream``.
    namespace:             ROS namespace for topics.
    use_sim_time:          Use /clock for message timestamps.
    log_level:             ROS log level (debug, info, warn, error, fatal).
    image_topic:           Input sensor_msgs/Image topic (empty keeps YAML).
    mask_topic:            Output mono8 mask topic (empty keeps YAML).
    text_prompt:           Fixed prompt used for every frame (empty keeps YAML).
    max_rate_hz:           Cap the processing rate; 0 means unlimited.
    queue_size:            Frame queue depth between callback and GPU worker.
    device:                Torch device, e.g. ``cuda``, ``cuda:1`` or ``cpu``.
    checkpoint_path:       Explicit .pt checkpoint path (skips Hugging Face).
    precision:             Autocast precision: bf16, fp16 or fp32.
    confidence_threshold:  Default detection threshold in [0, 1].
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
            [FindPackageShare("sam3_ros"), "config", "sam3_stream.yaml"]
        ),
        description=(
            "YAML parameter file. It is loaded before the model is built and "
            "may contain the full nested model/inference/stream configuration."
        ),
    ),
    DeclareLaunchArgument(
        "node_name",
        default_value="sam3_stream",
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
        "image_topic",
        default_value="",
        description="Override stream.image_topic. Empty keeps YAML.",
    ),
    DeclareLaunchArgument(
        "mask_topic",
        default_value="",
        description="Override stream.mask_topic. Empty keeps YAML.",
    ),
    DeclareLaunchArgument(
        "text_prompt",
        default_value="",
        description=(
            "Fixed text prompt for the whole stream. Empty keeps the YAML "
            "value. The prompt cannot change at runtime; restart the node."
        ),
    ),
    DeclareLaunchArgument(
        "max_rate_hz",
        default_value="",
        description="Override stream.max_rate_hz (0 = unlimited).",
    ),
    DeclareLaunchArgument(
        "queue_size",
        default_value="",
        description="Override stream.queue_size (>= 1).",
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
        description="Override model.checkpoint_path. Empty keeps YAML.",
    ),
    DeclareLaunchArgument(
        "precision",
        default_value="",
        description="Override inference.precision (bf16, fp16, fp32).",
    ),
    DeclareLaunchArgument(
        "confidence_threshold",
        default_value="",
        description="Override inference.confidence_threshold in [0, 1].",
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
        "image_topic": "stream.image_topic",
        "mask_topic": "stream.mask_topic",
        "text_prompt": "stream.text_prompt",
        "device": "model.device",
        "precision": "inference.precision",
        "checkpoint_path": "model.checkpoint_path",
    }
    for argument, parameter in string_overrides.items():
        value = _optional(context, argument)
        if value:
            overrides[parameter] = ParameterValue(value)

    integer_overrides = {
        "queue_size": "stream.queue_size",
    }
    for argument, parameter in integer_overrides.items():
        value = _optional(context, argument)
        if value:
            overrides[parameter] = ParameterValue(int(value), value_type=int)

    float_overrides = {
        "max_rate_hz": "stream.max_rate_hz",
        "confidence_threshold": "inference.confidence_threshold",
    }
    for argument, parameter in float_overrides.items():
        value = _optional(context, argument)
        if value:
            overrides[parameter] = ParameterValue(
                float(value), value_type=float
            )

    arguments: list[str] = []
    log_level = _optional(context, "log_level")
    if log_level:
        arguments = ["--ros-args", "--log-level", log_level]

    node = Node(
        package="sam3_ros",
        executable="sam3_stream",
        name=_optional(context, "node_name") or "sam3_stream",
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

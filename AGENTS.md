# realsense-yolov8-nitros-bridge

Follow [workspace rules](../AGENTS.md) and [CI](../docs/CI.md).
ROS package: `realsense_yolov8_nitros_bridge`.
Read [copy-boundary analysis](README.md#1-the-problem) and
[IPC composition](README.md#2-copy-a-ros-2-middleware-copy-realsense--encoder)
before changing the pipeline; incorrect composition silently adds a frame copy.

## Scope

Own camera → NITROS → YOLO and `/detections_output`. Depth/bearing belongs to
`Realsense_ROI_Depth_Rectifier`, selection/tracking to `thornbots_pkg`.

## Open

Robot acceptance and rebuilding the TensorRT plan per Orin:
[hardware status](../JAZZY_FLASH.md#hardware-status) and
[full pipeline](README.md#full-robot-pipeline).

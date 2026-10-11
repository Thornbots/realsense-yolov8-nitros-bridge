# realsense-yolov8-nitros-bridge

Follow [workspace rules](../AGENTS.md) and [CI](../docs/CI.md).
ROS package: `realsense_yolov8_nitros_bridge`.
Read [copy boundary](README.md#copy-boundary) and
[pipeline composition](README.md#pipeline)
before changing the pipeline; incorrect composition silently adds a frame copy.

## Scope

Own camera → NITROS → YOLO and `/detections_output`. Depth/bearing belongs to
`Realsense_ROI_Depth_Rectifier`, selection/tracking to `thornbots_pkg`.

## Open

Robot acceptance and rebuilding the TensorRT plan per Orin:
[hardware status](../JAZZY_FLASH.md#hardware-status) and
[full pipeline](README.md#full-robot-pipeline).

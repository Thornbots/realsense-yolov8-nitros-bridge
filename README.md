# realsense-yolov8-nitros-bridge

Camera → YOLO detection for the sentry: one launch file runs the RealSense
driver, the Isaac ROS DNN encoder, TensorRT, the YOLOv8 decoder and the ROI
depth node in a single component container. Targets Jazzy, Isaac ROS 4.6 and
realsense-ros 4.56. Humble-era design notes live on the frozen
[`humble` branch](https://github.com/Thornbots/realsense-yolov8-nitros-bridge/blob/humble/README.md).

## Full robot pipeline

The YOLO launch publishes `/cv/panel_detections`. `auto.launch.py` runs
selection, tracking, aiming, localization and the hardware drivers; see
[the node diagram](../thornbots_pkg/README.md#nodes).

For a manual run, stop the boot service on the robot host first:
`sudo systemctl stop thornbots`. Use two terminals in the robot container.
If using a locally built overlay, source
`/workspaces/isaac_ros-dev/install/setup.bash` in each terminal first.

Terminal 1 — robot and aiming:

```bash
ros2 launch thornbots_pkg auto.launch.py
```

Terminal 2 — perception, using the TensorRT engine built for this robot:

```bash
ros2 launch realsense_yolov8_nitros_bridge isaac_ros_yolov8_realsense.launch.py \
    engine_file_path:=/workspaces/isaac_ros-dev/isaac_ros_assets/models/yolo11/yolo11s_fp16.plan \
    enable_serial_bridge:=False
```

Disabling this launch's serial bridge leaves `auto.launch.py` as the sole
owner of `/dev/ttyTHS1`. Selection settings belong on `auto.launch.py`;
its referee subscription supplies team colour. Stop both launches with
Ctrl+C. For perception alone, run only terminal 2. Stop any standalone
ROI camera launch before starting it.

The engine is a per-Orin TensorRT plan built from `best.onnx`
(`Thornbots/trained-models`, LFS, `detect/yolo11s_realsense/v1/weights/`);
rebuild it on each Orin after a TensorRT upgrade. See the
[boot service](../isaac-ros-startup/README.md) (`launch.env` holds the same
paths).

## Build

In the container workspace:

```bash
colcon build --symlink-install --packages-up-to realsense_yolov8_nitros_bridge
source install/setup.bash
```

Runtime dependencies are `isaac_ros_dnn_image_encoder`, `isaac_ros_yolov8`,
`isaac_ros_tensor_rt`, `realsense2_camera`, `roi_depth_query` and
`dji_serial_bridge`. The package installs `launch/` and `config/` and builds
one component library (CUDA runtime, `cv_bridge`, OpenCV).

## Pipeline

`isaac_ros_yolov8_realsense.launch.py` starts container
`yolov8_realsense_container` (`component_container_mt`) holding:

| Node | Role |
| --- | --- |
| `camera` (`realsense2_camera`) | Color and depth, intra-process on |
| `tensor_rt` | TensorRT inference |
| `yolov8_decoder_node` | Publishes `/detections_output` (`Detection2DArray`, 640×640 network space) |
| `roi_depth_node` | Scales boxes to color space, samples depth, publishes `/cv/panel_detections` |
| `image_snapshot` | Only with `enable_snapshot:=True` |

The DNN encoder (`isaac_ros_dnn_image_encoder`, namespace `yolov8_encoder`) is
attached to the same container by including its launch file. Outside the
container, `extrinsics_relay` (one-shot) copies `/extrinsics/depth_to_color`
into `roi_depth_node`'s parameters; `dji_serial_bridge` and
`detection_picker_visualizer` start only when enabled.

Topics are at root, not under `/camera`. realsense-ros 4.56 publishes
node-private `/camera/...` topics, so the camera's remappings move these back
to root (where sim also publishes them):

```
/color/image_raw               → encoder, snapshot, visualizer
/color/camera_info             → encoder, roi_depth_node
/depth/image_rect_raw          → roi_depth_node
/depth/camera_info             → roi_depth_node
/extrinsics/depth_to_color     → extrinsics_relay → roi_depth_node params
```

Adding a namespace to the camera requires updating those remappings and the
topic constants in the launch file.

## Launch arguments

Only `engine_file_path` has no usable default. Pass others as `name:=value`;
list-valued ones as JSON strings.

| Group | Arguments (default) |
| --- | --- |
| Model | `engine_file_path` (empty), `model_file_path` (empty), `force_engine_update` (False), `verbose` (False) |
| Tensor names | `input_tensor_names` (`["input_tensor"]`), `input_binding_names` (`["images"]`), `output_tensor_names` (`["output_tensor"]`), `output_binding_names` (`["output0"]`) |
| Image sizes | `input_image_width`/`height` (640/480), `network_image_width`/`height` (640/640), `input_encoding` (rgb8), `image_mean` (`[0,0,0]`), `image_stddev` (`[1,1,1]`) |
| Decoder | `confidence_threshold` (0.25), `nms_threshold` (0.45), `num_classes` (8) |
| ROI depth | `center_sample_fraction` (0.25), `depth_max_age_s` (0.05), `max_detections` (16) |
| Serial bridge | `enable_serial_bridge` (True), `serial_device` (`/dev/ttyTHS1`), `serial_baudrate` (115200), `debug_log` (True) |
| Camera | `camera_initial_reset` (True; USB reset, about 5 s) |
| Bench overlay | `enable_visualizer` (False), `min_detection_score` (0.0), `ref_sys_topic` (`/dji_serial_bridge/ref_sys`), `center_weight` (1.0), `priority_class_bonus` (0.5), `priority_class_ids` (`[2, 6]`) |
| Snapshot | `enable_snapshot` (False), `snapshot_output_dir` (`$ISAAC_ROS_WS/data/realsense-captures`), `snapshot_interval_ms` (500), `snapshot_format` (jpg), `snapshot_disk_limit_pct` (75.0) |

Notes:

- `num_classes` must match the model. The decoder defaults to 80 (COCO) and
  assumes 8400 anchors; a smaller model makes it read past the output tensor
  and segfault silently.
- The encoder writes its output tensor under `tensor_name`; the launch sets it
  to `input_tensor_names[0]` because `TensorRTNode` looks the tensor up by that
  name (the encoder's own default is `output_tensor`).
- The visualizer arguments only drive the bench overlay
  (`/yolov8_processed_image`), which letterboxes the color image itself. The
  real selector takes its settings from `auto.launch.py`.

## Copy boundary

`sensor_msgs/Image` is a CPU `std::vector`, so two copies are in question:

- **Camera → encoder (CPU → CPU):** removed by intra-process comms. In one
  process rclcpp passes the publisher's `shared_ptr<Image>` to the subscriber;
  across processes, or with intra-process off, the message is serialised and
  copied. Everything on the image path must therefore share
  `yolov8_realsense_container`, and the camera node must keep
  `use_intra_process_comms`. A different composition silently adds a frame
  copy.
- **Encoder input (CPU → GPU):** remains. `DnnImageEncoderNode` takes a CPU
  `Image`, letterboxes it (uniform scale, centred zero padding) and uploads
  it once per frame. Removing it would need realsense-ros to publish
  `NitrosImage`; librealsense frames live in USB/UVC driver memory with no
  pluggable allocator, so some host-to-device copy always stays.

On 4.6, NITROS is an rclcpp `TypeAdapter` over a CUDA buffer synchronised with
CUDA events, published with a plain `create_publisher<NitrosImage>`.

## Components

- `realsense_nitros_bridge::ImageSnapshotNode`: timer-driven training-image
  capture (`<sec>_<nanosec>.<fmt>`, never overwritten). It saves the latest
  frame at `interval_ms`, refuses to load above `disk_limit_pct` (the launch
  also checks beforehand and fails with a readable message), and re-checks
  every `disk_check_interval` saves, stopping capture if the limit is crossed.
- `realsense_nitros_bridge::NitrosRealsenseBridgeNode`: unused prototype, not
  launched. It stages each `Image` in pinned memory, copies it to a fresh
  device buffer with a synchronous `cudaMemcpy` and publishes a `NitrosImage`
  on `nitros_image`. It adds a CPU memcpy and does not remove the H2D copy,
  so it is not a replacement for the encoder's input path.

## Camera profile

`config/realsense_640x480x60.yaml` runs color and depth at 640×480, 60 fps,
with the IR emitter always on. realsense-ros 4.56 reads
`rgb_camera.color_profile` and `depth_module.depth_profile` and ignores the
bare `profile` key (the file keeps both). `roi_depth_node` only caches the
latest depth frame and samples it on detection events, so depth's own rate need
not match detection load. If "Depth stream start failure" or UVC watchdog
errors appear on a bandwidth-limited USB controller, cap
`depth_module.depth_profile` at `640x480x30`; that reduces USB bandwidth, not
processing load. Not yet re-verified on hardware at 60+60.

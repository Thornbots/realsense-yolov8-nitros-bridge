# realsense-yolov8-nitros-bridge

Node and launch file optimising the interface between the realsense node
and the isaac-ros-3.2 yolov8 example. Most of this README is the copy-
boundary analysis behind those optimisations.

## 1. The problem

The standard `yolov8_tensor_rt.launch.py` pipeline:

```
realsense2_camera node          (publishes sensor_msgs/Image)
        │
        │  ← DDS serialise + memcpy to DDS heap (inter-process)  ← if separate containers
        ▼
dnn_image_encoder (ResizeNode + ImageToTensorNode)
        │  cudaMemcpyDefault: CPU pinned → GPU
        ▼
TensorRTNode                    (GPU, NITROS zero-copy internally)
        ▼
YoloV8DecoderNode
```

There are two distinct copy events to reason about separately:

## 2. Copy A: ROS 2 middleware copy (realsense → encoder)

`sensor_msgs/Image` carries a `std::vector<uint8_t> data` field, and the
realsense-ros node wraps each librealsense frame in an `Image` to publish
it.

| Scenario | What happens to the bytes |
|---|---|
| Separate processes / containers | rmw serialises the vector into a DDS loan buffer; the subscriber deserialises into a new `vector`. One full memcpy on the CPU. |
| Same container, IPC disabled | rclcpp still serialises/deserialises even intra-process. Same full copy. |
| Same container, IPC enabled | rclcpp hands the publisher's `shared_ptr<Image>` directly to the subscriber, with no copy. The subscriber receives a `const shared_ptr<Image>`. |

### How to enable IPC (what the launch file does)

```python
ComposableNode(
    package='realsense2_camera',
    plugin='realsense2_camera::RealSenseNodeFactory',
    ...
    extra_arguments=[{'use_intra_process_comms': True}]
)
```

realsense-ros already supports IPC (it ships `rs_intra_process_demo_launch.py`,
and `image_publisher.cpp` uses `rclcpp::Publisher` normally). The
dnn_image_encoder `ResizeNode` is a standard rclcpp composable node, so in
the same `component_container_mt` with IPC enabled rclcpp hands it the same
`shared_ptr`.

> Status: eliminated by the accompanying launch file.

## 3. Copy B: cudaMemcpyDefault (CPU → GPU, inside dnn_image_encoder)

### What actually happens

Inside `dnn_image_encoder` /
`custom_nitros_dnn_image_encoder::ImageEncoderNode::InputCallback`:

```cpp
cudaMemcpy(
    input_image_buffer_.basePtr,   // GPU allocation
    msg->data.data(),              // CPU std::vector<uint8_t>
    buffer_size,
    cudaMemcpyDefault              // runtime picks H2D
);
```

This is a host-to-device transfer. Even with IPC eliminating Copy A, the
image data still lives in CPU RAM (the `Image::data` vector) and has to be
transferred to GPU for CUDA/TensorRT processing.

### Can it be eliminated?

Not on the current realsense-ros architecture:

1. librealsense frames live in CPU-accessible memory allocated by the
   UVC/USB driver subsystem. There's no zero-copy path from the camera USB
   buffer to GPU memory without an explicit `cudaMemcpy`.
2. realsense-ros publishes `sensor_msgs/Image`, not `NitrosImage`. Until
   someone writes a plugin wrapping librealsense frames in a GXF
   `VideoBuffer` behind the NITROS type adapter, Copy B is unavoidable.
3. Unified/pinned memory reduces the *cost* of Copy B but can't remove it.
   The data originates in a kernel DMA buffer that isn't a CUDA allocation.

NITROS's "zero-copy" claim applies between NITROS nodes (encoder →
TensorRT → decoder), where both sides share a GXF `VideoBuffer` backed by a
CUDA allocation and the pointer passes directly. That's separate from the
CPU→GPU transfer, which still has to happen once.

## 4. The only real path to eliminating Copy B

To remove the H2D transfer, realsense-ros would need to:

1. Allocate a CUDA pinned buffer (`cudaHostAlloc`) of sufficient size for a color frame.
2. Use `rs2::frame::get_data()` to get the librealsense frame pointer, then
   copy into the pinned buffer (or use a custom allocator if librealsense ever
   supports pluggable allocators).
3. Publish a `NitrosImage` wrapping a
   `NitrosImageBuilder().WithGpuData(pinned_ptr)` (pinned memory is accessible
   to CUDA kernels as device memory via UVA).
4. Load the `NitrosTypeManager` and use `ManagedNitrosPublisher<NitrosImage>`.

This is exactly what `gpu_image_builder_node.cpp` in the
`custom_nitros_image` example demonstrates, minus the librealsense
integration. A prototype bridge node could be written using that example as
a template.

## 5. IPC compatibility

`realsense2_camera` supports IPC but not NITROS; the three downstream nodes
(`dnn_image_encoder`, `TensorRTNode`, `YoloV8DecoderNode`) are NITROS. All
four have to share a container for any of this to apply, and the launch
file puts them in one `component_container_mt`.

## 6. CUDA stream ordering

Both the `custom_nitros_dnn_image_encoder` example and the
`dnn_image_encoder` package pass the default stream (`(cudaStream_t) 0`) to
all cvcuda operations. Against a NITROS node using its own stream pool, the
implicit synchronisation on stream 0 is safe but not optimal. Sharing a
stream via the GXF `CudaStreamPool` would be faster, but needs the encoder
to accept a `CudaStreamHandle`, which the existing launch fragment does not
do.

## 7. Summary

`isaac_ros_yolov8_realsense.launch.py` is the best achievable on the
current open-source stack. Copy A (CPU→CPU, in rclcpp publish/subscribe) is
gone, via IPC in a shared container. Copy B (CPU→GPU, the
`cudaMemcpyDefault` in the encoder) remains: eliminating it needs a
NitrosImage-native realsense driver. It happens once per frame as data
enters the GPU pipeline.

## Notes

Trimmed-out detail from in-code comments, kept here for reference.

### `launch/isaac_ros_yolov8_realsense.launch.py`

#### Verified runtime topic layout

With `ComposableNode(name='camera', namespace='')`, realsense-ros resolves
all topics against the root namespace, and there is NO `/camera/` prefix:

```
/color/image_raw               → dnn_image_encoder
/color/camera_info             → roi_depth_node (LUT build)
/depth/image_rect_raw          → roi_depth_node (sampling)
/depth/camera_info             → roi_depth_node (LUT build)
/extrinsics/depth_to_color     → extrinsics_relay_node → roi_depth_node params
```

If you launch with an explicit namespace (e.g. `namespace='camera'`), all
topics gain a `/camera/` prefix and these constants must be updated to
match.

#### Full inference chain

```
/color/image_raw
  → dnn_image_encoder (resize 640×480 → 640×640, normalise, interleave→planar)
  → /tensor_pub → tensor_rt (TensorRT YOLOv8 inference)
  → /tensor_sub → yolov8_decoder_node
  → /detections_output  (Detection2DArray, bbox in 640×640 NETWORK space, ALL detections)
  → roi_depth_node  (scales each bbox to color space, LUT lookup +
                      center-sample depth, deprojects bbox corners + center)
  → /cv/panel_detections  (dji_serial_bridge/msg/PanelDetectionArray, REP-103
                            camera frame, one entry per detection)
  → target_selector.py  (thornbots_pkg package, does team filter, 3D robot
                          grouping, per-frame panel pick)
  → /cv/panel_detection  (dji_serial_bridge/msg/PanelDetection: the winner)
  → target_tracker.py  (thornbots_pkg, spin-centre KF estimate in odom)
  → /cv/target_state  (dji_serial_bridge/msg/TargetState)
  → point_to_cv_target_node  (thornbots_pkg, converts to root frame, optional
                               lead solve; also republishes
                               /cv/panel_polygon for visualization)
  → /cv/target  (dji_serial_bridge/msg/CVTarget)
  → dji_serial_bridge_node  → UART → MCB / gimbal controller
```

#### Team-colour filtering

`target_selector.py` (in `thornbots_pkg`, launched from `auto.launch.py`)
subscribes to the referee system status published by
`dji_serial_bridge_node` on `/dji_serial_bridge/ref_sys` (`RefSysStatus`).
Blue team excludes class IDs 0-3, red team excludes 4-7. Until the first
status arrives, all detections pass through, with a throttled warning.

Set `enable_serial_bridge:=false` to omit the last two nodes, for example
when bench-testing the vision pipeline without the MCB attached.

#### Usage

Only `engine_file_path` is required; everything else has a default. Pass
args as plain `name:=value`, since bracketing one makes the token part of
the *name* and the override is then silently ignored.
`priority_class_ids:=[2,6]` is the sole exception, where the brackets are
the list value.

```bash
ros2 launch realsense_yolov8_nitros_bridge isaac_ros_yolov8_realsense.launch.py \
    engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolo11/yolo11s_fp16.plan \
    num_classes:=8 confidence_threshold:=0.25 nms_threshold:=0.45 \
    priority_class_ids:=[2,6] serial_device:=/dev/ttyTHS1
```

### `src/image_snapshot_node.cpp`

`rclcpp::Subscription::take()` in Humble (and Galactic) accepts a value
reference (`ROSMessageType&`), not a `SharedPtr`. The message is moved into
a `shared_ptr` before being passed to `cv_bridge` so `toCvShare` can alias
the buffer without a pixel copy. The `SharedPtr` overload was added in Iron.

### `src/nitros_realsense_bridge_node.cpp`

This is the bridge node described in section 4 above, a drop-in replacement
for the realsense→dnn_image_encoder connection.

librealsense does not support pluggable allocators, so this node still pays
one `cudaMemcpyHostToDevice`. What it saves versus the stock encoder:

- No intermediate CPU resize (the raw frame is pushed to GPU, then resized on GPU).
- The frame sits in pinned memory so the H2D transfer can be DMA-pipelined
  while the GPU is busy with the previous frame's inference.

For a truly zero-copy path, realsense-ros would need to allocate its image
buffers in CUDA pinned memory from the start, which requires patching
librealsense's frame allocator.

### `config/realsense_640x480x60.yaml`

Both streams run at 60 fps (checked 2026-07-29: `depth_module.profile` is
`640x480x60`, matching the filename). An earlier revision of this file
described a 30fps depth cap, but that halving was never actually present in
the config; the paragraph was simply stale. `roi_depth_node` drives off
`/detections_output` and only caches the latest depth frame, so it genuinely
samples depth on detection events rather than every depth frame. That
doesn't require lowering depth's own publish rate, since the node isn't
reacting to every depth frame regardless of what rate it arrives at. If the
UVC watchdog / "Depth stream start failure" reappears on
bandwidth-constrained USB controllers at 60+60, cap `depth_module.profile`
back to `640x480x30` in the yaml: that reduces USB bandwidth, not
detection-processing load.

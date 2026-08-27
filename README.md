# realsense-yolov8-nitros-bridge
Node and launchfile implementing optimizations at interface between realsense node and yolov8 isaac-ros-3.2 example

# RealSense → Isaac ROS NITROS: Copy Boundary Analysis & Elimination Path

## 1. The problem, precisely stated

The standard `yolov8_tensor_rt.launch.py` pipeline looks like this:

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

There are **two** distinct copy events to reason about separately:

---

## 2. Copy A: ROS 2 middleware copy (realsense → encoder)

### What actually happens

`sensor_msgs/Image` carries a `std::vector<uint8_t> data` field.  
The realsense-ros node wraps each librealsense frame in an `Image` and **publishes** it.

| Scenario | What happens to the bytes |
|---|---|
| **Separate processes / containers** | rmw serialises the vector into a DDS loan buffer; subscriber deserialises into a new `vector`. **One full memcpy on the CPU.** |
| **Same container, IPC disabled** | rclcpp still serialises/deserialises even intra-process. **Same full copy.** |
| **Same container, IPC enabled** | rclcpp hands the publisher's `shared_ptr<Image>` directly to the subscriber. **Zero copy.** The subscriber receives a `const shared_ptr<Image>`. |

### How to enable IPC (what the launch file does)

```python
ComposableNode(
    package='realsense2_camera',
    plugin='realsense2_camera::RealSenseNodeFactory',
    ...
    extra_arguments=[{'use_intra_process_comms': True}]
)
```

The realsense-ros node already supports IPC. It has a dedicated demo launch (`rs_intra_process_demo_launch.py`), and `image_publisher.cpp` uses `rclcpp::Publisher` normally, so IPC just works.

The dnn_image_encoder `ResizeNode` is a standard rclcpp composable node that subscribes to `sensor_msgs/Image`.  When it lives in the same `component_container_mt` with IPC enabled, rclcpp will hand it the **same `shared_ptr`** the realsense node created. **No copy.**

> **Status: ELIMINATED** by the accompanying launch file.

---

## 3. Copy B: cudaMemcpyDefault (CPU → GPU, inside dnn_image_encoder)

### What actually happens

Inside `dnn_image_encoder` / `custom_nitros_dnn_image_encoder::ImageEncoderNode::InputCallback`:

```cpp
cudaMemcpy(
    input_image_buffer_.basePtr,   // GPU allocation
    msg->data.data(),              // CPU std::vector<uint8_t>
    buffer_size,
    cudaMemcpyDefault              // runtime picks H2D
);
```

This is a **host-to-device transfer**.  Even with IPC eliminating Copy A, the image data still lives in CPU RAM (the `Image::data` vector) and must be transferred to GPU for CUDA/TensorRT processing.

### Can it be eliminated?

The current realsense-ros architecture doesn't support this, for three reasons:

1. librealsense frames live in CPU-accessible memory, allocated by the UVC/USB driver subsystem. There is no zero-copy path from the camera USB buffer to GPU memory without an explicit `cudaMemcpy`.

2. realsense-ros does not publish `NitrosImage`. It publishes `sensor_msgs/Image`, whose `data` field is a CPU `std::vector`. Until NVIDIA or the Intel team writes a realsense-ros plugin that wraps librealsense frames in a GXF `VideoBuffer` and publishes them through the NITROS type adapter, Copy B is unavoidable.

3. CUDA Unified Memory / pinned memory can reduce the *cost* of Copy B but cannot eliminate it. The camera data originates in a kernel DMA buffer that is not a CUDA allocation.

### What NITROS zero-copy actually covers

The "zero-copy" claim in NITROS applies **between NITROS nodes** (encoder → TensorRT → decoder). The `NitrosImage` and `NitrosTensorList` types use a GXF `VideoBuffer` backed by a CUDA allocation. When both publisher and subscriber are NITROS nodes in the same GXF context, the buffer pointer is passed directly, with **no copy**. This is separate from the CPU→GPU transfer that must still happen once.

---

## 4. The only real path to eliminating Copy B

To remove the H2D transfer, realsense-ros would need to:

1. Allocate a CUDA pinned buffer (`cudaHostAlloc`) of sufficient size for a color frame.
2. Use `rs2::frame::get_data()` to get the librealsense frame pointer, then copy into the pinned buffer (or use a custom allocator if librealsense ever supports pluggable allocators).
3. Publish a `NitrosImage` wrapping a `NitrosImageBuilder().WithGpuData(pinned_ptr)` (pinned memory is accessible to CUDA kernels as device memory via UVA).
4. Load the `NitrosTypeManager` and use `ManagedNitrosPublisher<NitrosImage>`.

This is exactly what `gpu_image_builder_node.cpp` in the `custom_nitros_image` example demonstrates, minus the librealsense integration. A prototype bridge node could be written using that example as a template.

---

## 5. Practical IPC compatibility matrix

| Component | IPC | NITROS | In same container? |
|---|---|---|---|
| `realsense2_camera` | ✅ yes | ❌ no | ✅ required for IPC |
| `dnn_image_encoder` nodes | ✅ yes (input) | ✅ yes (output) | ✅ required |
| `TensorRTNode` | N/A | ✅ yes | ✅ required |
| `YoloV8DecoderNode` | N/A | ✅ yes (input) | ✅ required |

The launch file places all four in a single `component_container_mt`.

---

## 6. CUDA stream ordering note

The `custom_nitros_dnn_image_encoder` example passes `(cudaStream_t) 0` (the default stream) to all cvcuda operations.  The `dnn_image_encoder` package similarly uses a single stream.  When combined with a NITROS node that uses its own CUDA stream pool, the implicit synchronisation on stream 0 is safe but not optimal.  For maximum throughput, the encoder and TensorRT node should share a stream via the GXF `CudaStreamPool`.  This is not addressed by the existing `dnn_image_encoder` launch fragment and would require modifying the encoder to accept a GXF `CudaStreamHandle`.

---

## 7. Summary

| Copy | Location | Eliminated by this PR? | How |
|---|---|---|---|
| A: CPU → CPU (DDS) | rclcpp publish/subscribe | **Yes** | IPC in shared container |
| B: CPU → GPU (H2D) | `cudaMemcpyDefault` in encoder | **No** | Requires NitrosImage-native realsense driver |

The launch file in `isaac_ros_yolov8_realsense.launch.py` is the best achievable with the current open-source stack: Copy A is removed via IPC; Copy B is the one unavoidable H2D transfer, occurring once per frame as data enters the GPU pipeline.

## Notes

Trimmed-out detail from in-code comments, kept here for reference.

### `launch/isaac_ros_yolov8_realsense.launch.py`

**Verified runtime topic layout** (with `ComposableNode(name='camera', namespace='')`, realsense-ros resolves all topics against the root namespace, and there is NO `/camera/` prefix):

```
/color/image_raw               → dnn_image_encoder
/color/camera_info             → roi_depth_node (LUT build)
/depth/image_rect_raw          → roi_depth_node (sampling)
/depth/camera_info             → roi_depth_node (LUT build)
/extrinsics/depth_to_color     → extrinsics_relay_node → roi_depth_node params
```

If you launch with an explicit namespace (e.g. `namespace='camera'`), all topics gain a `/camera/` prefix and these constants must be updated to match.

**Full inference chain:**

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
  → target_selector.py  (sentry_pkg package, does team filter, 3D robot
                          grouping, per-frame panel pick)
  → /cv/panel_detection  (dji_serial_bridge/msg/PanelDetection: the winner)
  → point_to_cv_target_node  (sentry_pkg package, frame convert; also
                               republishes /cv/panel_polygon for
                               visualization)
  → /cv_target  (dji_serial_bridge/msg/CVTarget)
  → dji_serial_bridge_node  → UART → MCB / gimbal controller
```

**Team-colour filtering:** `target_selector.py` (in `sentry_pkg`, launched from `auto.launch.py`) subscribes to the referee system status published by `dji_serial_bridge_node` on `/dji_serial_bridge/ref_sys` (`RefSysStatus`). Blue team excludes class IDs 0–3, red team excludes 4–7. Until the first status arrives, all detections pass through (with a throttled warning).

Set `enable_serial_bridge:=false` to omit the last two nodes (e.g. when bench-testing the vision pipeline without the MCB attached).

**Usage example**: every argument besides `engine_file_path` has a default; pass args as plain `name:=value` pairs (do not bracket them, or the token becomes part of the name and the override is silently ignored; `priority_class_ids:=[2,6]` is the sole exception, where the brackets are the list value itself):

```
ros2 launch realsense_yolov8_nitros_bridge isaac_ros_yolov8_realsense.launch.py \
    engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolo11/yolo11s_fp16.plan \
    num_classes:=8 \
    confidence_threshold:=0.25 nms_threshold:=0.45 \
    center_sample_fraction:=0.25 \
    center_weight:=1.0 priority_class_bonus:=0.5 priority_class_ids:=[2,6] \
    ref_sys_topic:=/dji_serial_bridge/ref_sys \
    serial_device:=/dev/ttyTHS1 serial_baudrate:=115200 \
    enable_sentry_pkg:=True lidar_serial_port:=/dev/ttyUSB0 enable_rviz:=False \
    enable_snapshot:=False snapshot_output_dir:=/data/realsense-captures
```

### `src/image_snapshot_node.cpp`

`rclcpp::Subscription::take()` in Humble (and Galactic) accepts a value reference (`ROSMessageType&`), not a `SharedPtr`. The message is moved into a `shared_ptr` before being passed to `cv_bridge` so `toCvShare` can alias the buffer without a pixel copy. The `SharedPtr` overload was added in Iron.

### `src/nitros_realsense_bridge_node.cpp`

This is the bridge node described in section 4 above, a drop-in replacement for the realsense→dnn_image_encoder connection.

librealsense does not support pluggable allocators, so this node still pays one `cudaMemcpyHostToDevice`. What it saves versus the stock encoder:

- No intermediate CPU resize (the raw frame is pushed to GPU, then resized on GPU).
- The frame sits in pinned memory so the H2D transfer can be DMA-pipelined while the GPU is busy with the previous frame's inference.

For a truly zero-copy path, realsense-ros would need to allocate its image buffers in CUDA pinned memory from the start, which requires patching librealsense's frame allocator.

### `config/realsense_640x480x60.yaml`

Both streams run at 60 fps (checked 2026-07-29: `depth_module.profile` is `640x480x60`, matching the filename). An earlier revision of this file described a 30fps depth cap, but that halving was never actually present in the config; the paragraph was simply stale. `roi_depth_node` drives off `/detections_output` and only caches the latest depth frame (Phase 0 of the CV tracking plan), so it genuinely samples depth on detection events rather than every depth frame. That doesn't require lowering depth's own publish rate, since the node isn't reacting to every depth frame regardless of what rate it arrives at. If the UVC watchdog / "Depth stream start failure" reappears on bandwidth-constrained USB controllers at 60+60, cap `depth_module.profile` back to `640x480x30` in the yaml: that reduces USB bandwidth, not detection-processing load.

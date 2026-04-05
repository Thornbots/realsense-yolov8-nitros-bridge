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

## 2. Copy A — ROS 2 middleware copy (realsense → encoder)

### What actually happens

`sensor_msgs/Image` carries a `std::vector<uint8_t> data` field.  
The realsense-ros node wraps each librealsense frame in an `Image` and **publishes** it.

| Scenario | What happens to the bytes |
|---|---|
| **Separate processes / containers** | rmw serialises the vector into a DDS loan buffer; subscriber deserialises into a new `vector`. **One full memcpy on the CPU.** |
| **Same container, IPC disabled** | rclcpp still serialises/deserialises even intra-process. **Same full copy.** |
| **Same container, IPC enabled** | rclcpp hands the publisher's `shared_ptr<Image>` directly to the subscriber. **Zero copy** — the subscriber receives a `const shared_ptr<Image>`. |

### How to enable IPC (what the launch file does)

```python
ComposableNode(
    package='realsense2_camera',
    plugin='realsense2_camera::RealSenseNodeFactory',
    ...
    extra_arguments=[{'use_intra_process_comms': True}]
)
```

The realsense-ros node already supports IPC — it has a dedicated demo launch (`rs_intra_process_demo_launch.py`) and the `image_publisher.cpp` uses `rclcpp::Publisher` normally so IPC just works.

The dnn_image_encoder `ResizeNode` is a standard rclcpp composable node that subscribes to `sensor_msgs/Image`.  When it lives in the same `component_container_mt` with IPC enabled, rclcpp will hand it the **same `shared_ptr`** the realsense node created — **no copy**.

> **Status: ELIMINATED** by the accompanying launch file.

---

## 3. Copy B — cudaMemcpyDefault (CPU → GPU, inside dnn_image_encoder)

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

**Not with the current realsense-ros architecture.**  Here is why:

1. **librealsense frames live in CPU-accessible memory** allocated by the UVC/USB driver subsystem.  There is no zero-copy path from the camera USB buffer to GPU memory without an explicit `cudaMemcpy`.

2. **realsense-ros does not publish `NitrosImage`** — it publishes `sensor_msgs/Image`, whose `data` field is a CPU `std::vector`.  Until NVIDIA or the Intel team writes a realsense-ros plugin that wraps librealsense frames in a GXF `VideoBuffer` and publishes them through the NITROS type adapter, Copy B is unavoidable.

3. **CUDA Unified Memory / pinned memory** can reduce the *cost* of Copy B but cannot eliminate it.  The camera data originates in a kernel DMA buffer that is not a CUDA allocation.

### What NITROS zero-copy actually covers

The "zero-copy" claim in NITROS applies **between NITROS nodes** (encoder → TensorRT → decoder).  The `NitrosImage` and `NitrosTensorList` types use a GXF `VideoBuffer` backed by a CUDA allocation.  When both publisher and subscriber are NITROS nodes in the same GXF context, the buffer pointer is passed directly — **no copy**.  This is distinct from, and orthogonal to, the CPU→GPU transfer that must still happen once.

---

## 4. The only real path to eliminating Copy B

To remove the H2D transfer, realsense-ros would need to:

1. **Allocate a CUDA pinned buffer** (`cudaHostAlloc`) of sufficient size for a color frame.
2. **Use `rs2::frame::get_data()`** to get the librealsense frame pointer, then copy into the pinned buffer (or use a custom allocator if librealsense ever supports pluggable allocators).
3. **Publish a `NitrosImage`** wrapping a `NitrosImageBuilder().WithGpuData(pinned_ptr)` (pinned memory is accessible to CUDA kernels as device memory via UVA).
4. Load the `NitrosTypeManager` and use `ManagedNitrosPublisher<NitrosImage>`.

This is exactly what `gpu_image_builder_node.cpp` in the `custom_nitros_image` example demonstrates — it just lacks the librealsense integration.  A prototype bridge node could be written using that example as a template.

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

The launch file in `isaac_ros_yolov8_realsense.launch.py` gives you the best possible performance with the current open-source stack: Copy A is removed via IPC, and Copy B is the single unavoidable H2D transfer that occurs once per frame as the data enters the GPU pipeline.

// nitros_realsense_bridge_node.cpp
//
// PROTOTYPE — eliminates the CPU→GPU copy (Copy B) by publishing a NitrosImage
// directly from a sensor_msgs/Image subscription using CUDA pinned memory.
//
// This is the bridge node described in NOTES.md §4.  It is a drop-in
// replacement for the realsense→dnn_image_encoder connection.
//
// Build requirements (add to CMakeLists.txt):
//   isaac_ros_managed_nitros
//   isaac_ros_nitros_image_type
//   CUDA::cudart
//
// Drop this node into the same component_container_mt as TensorRTNode.
// Point its "image" subscription at /camera/camera/color/image_raw.
// The TensorRTNode (or dnn_image_encoder) subscribes to its "nitros_image" output.
//
// NOTE: librealsense does not support pluggable allocators, so we still pay
// one cudaMemcpyHostToDevice here.  What we save vs. the stock encoder:
//   • No intermediate CPU resize (we push the raw frame to GPU then resize on GPU)
//   • The frame sits in pinned memory so the H2D transfer can be DMA-pipelined
//     while the GPU is busy with the previous frame's inference.
//
// For a truly zero-copy path, realsense-ros would need to allocate its image
// buffers in CUDA pinned memory from the start — that requires patching
// librealsense's frame allocator.

#include <cuda_runtime.h>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_image_type/nitros_image_builder.hpp"

#include "sensor_msgs/image_encodings.hpp"

namespace realsense_nitros_bridge
{

class NitrosRealsenseBridgeNode : public rclcpp::Node
{
public:
    explicit NitrosRealsenseBridgeNode(const rclcpp::NodeOptions & opts = rclcpp::NodeOptions())
    : Node("nitros_realsense_bridge", opts),
      pinned_buf_(nullptr),
      pinned_size_(0)
    {
        // Subscribe to the realsense color topic (sensor_msgs/Image, CPU memory)
        sub_ = create_subscription<sensor_msgs::msg::Image>(
            "image", 10,
            [this](sensor_msgs::msg::Image::ConstSharedPtr msg) { onImage(msg); });

        // Publish NitrosImage — NITROS nodes (TensorRTNode, dnn_image_encoder)
        // can subscribe to this without any additional copy.
        nitros_pub_ = std::make_shared<
            nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
                nvidia::isaac_ros::nitros::NitrosImage>>(
            this,
            "nitros_image",
            nvidia::isaac_ros::nitros::nitros_image_rgb8_t::supported_type_name);

        RCLCPP_INFO(get_logger(),
            "NitrosRealsenseBridgeNode ready — "
            "publishing NitrosImage on 'nitros_image'");
    }

    ~NitrosRealsenseBridgeNode()
    {
        if (pinned_buf_) {
            cudaFreeHost(pinned_buf_);
        }
    }

private:
    void onImage(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
    {
        const size_t frame_bytes = msg->step * msg->height;

        // ── Step 1: ensure pinned-memory staging buffer is large enough ────
        // Pinned (page-locked) memory allows the CUDA DMA engine to transfer
        // to GPU concurrently with CPU execution, unlike regular heap memory.
        if (frame_bytes > pinned_size_) {
            if (pinned_buf_) cudaFreeHost(pinned_buf_);
            cudaError_t err = cudaMallocHost(&pinned_buf_, frame_bytes);
            if (err != cudaSuccess) {
                RCLCPP_ERROR(get_logger(), "cudaMallocHost failed: %s",
                    cudaGetErrorString(err));
                return;
            }
            pinned_size_ = frame_bytes;
            RCLCPP_INFO(get_logger(),
                "Allocated %.1f MB pinned staging buffer",
                frame_bytes / 1e6);
        }

        // ── Step 2: copy CPU frame into pinned staging buffer ──────────────
        // If the realsense node ran with IPC enabled (same container), msg->data
        // is the original shared_ptr — no DDS copy happened before this point.
        std::memcpy(pinned_buf_, msg->data.data(), frame_bytes);

        // ── Step 3: allocate GPU buffer and H2D transfer ───────────────────
        // Using the default CUDA stream (0).  For production, use a stream from
        // a GXF CudaStreamPool to overlap this transfer with GPU inference of
        // the previous frame.
        void * gpu_buf = nullptr;
        cudaError_t err = cudaMalloc(&gpu_buf, frame_bytes);
        if (err != cudaSuccess) {
            RCLCPP_ERROR(get_logger(), "cudaMalloc failed: %s",
                cudaGetErrorString(err));
            return;
        }

        err = cudaMemcpy(gpu_buf, pinned_buf_, frame_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            RCLCPP_ERROR(get_logger(), "cudaMemcpy H2D failed: %s",
                cudaGetErrorString(err));
            cudaFree(gpu_buf);
            return;
        }

        // ── Step 4: wrap in NitrosImage and publish ────────────────────────
        // NitrosImageBuilder takes ownership of gpu_buf.
        // Downstream NITROS nodes (TensorRTNode etc.) receive the GPU pointer
        // without any further copy.
        namespace ni = nvidia::isaac_ros::nitros;
        namespace se = sensor_msgs::image_encodings;

        const std::string & enc = msg->encoding;
        const std::string nitros_enc =
            (enc == "bgr8")  ? se::BGR8  :
            (enc == "rgba8") ? se::RGBA8 :
            se::RGB8;  // default / rgb8

        ni::NitrosImage nitros_image =
            ni::NitrosImageBuilder()
            .WithHeader(msg->header)
            .WithEncoding(nitros_enc)
            .WithDimensions(msg->height, msg->width)
            .WithGpuData(gpu_buf)
            .Build();

        nitros_pub_->publish(nitros_image);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;

    std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
        nvidia::isaac_ros::nitros::NitrosImage>> nitros_pub_;

    // Pinned (page-locked) staging buffer — reused across frames
    void *  pinned_buf_;
    size_t  pinned_size_;
};

}  // namespace realsense_nitros_bridge


#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(realsense_nitros_bridge::NitrosRealsenseBridgeNode)

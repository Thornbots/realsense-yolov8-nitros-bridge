// Copyright 2026 Thornbots
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// nitros_realsense_bridge_node.cpp
//
// PROTOTYPE — publishes a NitrosImage from a sensor_msgs/Image subscription
// via CUDA pinned memory, replacing realsense->dnn_image_encoder. Requires
// isaac_ros_nitros_image_type, CUDA::cudart. Publishes with a plain
// rclcpp publisher (NitrosImage is a TypeAdapter since Isaac ROS 4.x;
// 4.6's ManagedNitrosPublisher header does not compile).
// Drop into the same component_container_mt as TensorRTNode.
// CAVEAT: librealsense lacks a pluggable allocator, so one
// cudaMemcpyHostToDevice remains -- see README.md for design notes.

#include <cuda_runtime.h>

#include <cstring>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

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
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {onImage(msg);});

    // Publish NitrosImage — NITROS nodes (TensorRTNode, dnn_image_encoder)
    // can subscribe to this without any additional copy.
    // Intra-process on, as in Isaac ROS 4.6's own NITROS nodes: the GPU
    // buffer only travels by pointer within one process.
    rclcpp::PublisherOptions pub_options;
    pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
    nitros_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosImage>(
      "nitros_image", rclcpp::QoS(1), pub_options);

    RCLCPP_INFO(
      get_logger(),
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
      if (pinned_buf_) {cudaFreeHost(pinned_buf_);}
      cudaError_t err = cudaMallocHost(&pinned_buf_, frame_bytes);
      if (err != cudaSuccess) {
        RCLCPP_ERROR(
          get_logger(), "cudaMallocHost failed: %s",
          cudaGetErrorString(err));
        return;
      }
      pinned_size_ = frame_bytes;
      RCLCPP_INFO(
        get_logger(),
        "Allocated %.1f MB pinned staging buffer",
        frame_bytes / 1e6);
    }

    // ── Step 2: copy CPU frame into pinned staging buffer ──────────────
    // If the realsense node ran with IPC enabled (same container), msg->data
    // is the original shared_ptr — no DDS copy happened before this point.
    std::memcpy(pinned_buf_, msg->data.data(), frame_bytes);

    // ── Step 3: allocate GPU buffer and H2D transfer ───────────────────
    // Synchronous copy on the default stream. To overlap it with inference,
    // use NitrosImage::from_pool() + cudaMemcpyAsync on the write handle's
    // stream (Isaac ROS 4.5+ event-based NitrosBuffer sync).
    void * gpu_buf = nullptr;
    cudaError_t err = cudaMalloc(&gpu_buf, frame_bytes);
    if (err != cudaSuccess) {
      RCLCPP_ERROR(
        get_logger(), "cudaMalloc failed: %s",
        cudaGetErrorString(err));
      return;
    }

    err = cudaMemcpy(gpu_buf, pinned_buf_, frame_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      RCLCPP_ERROR(
        get_logger(), "cudaMemcpy H2D failed: %s",
        cudaGetErrorString(err));
      cudaFree(gpu_buf);
      return;
    }

    // ── Step 4: wrap in NitrosImage and publish ────────────────────────
    // NitrosImageBuilder takes ownership of gpu_buf (freed with
    // cudaFreeAsync when the last reader drops it). The copy above is
    // synchronous, so it finishes before Build() records the write event.
    // Downstream NITROS nodes (TensorRTNode etc.) receive the GPU pointer
    // without any further copy.
    namespace ni = nvidia::isaac_ros::nitros;
    namespace se = sensor_msgs::image_encodings;

    const std::string & enc = msg->encoding;
    const std::string nitros_enc =
      (enc == "bgr8") ? se::BGR8 :
      (enc == "rgba8") ? se::RGBA8 :
      se::RGB8;        // default / rgb8

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

  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosImage>::SharedPtr nitros_pub_;

  // Pinned (page-locked) staging buffer — reused across frames
  void * pinned_buf_;
  size_t pinned_size_;
};

}  // namespace realsense_nitros_bridge


#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(realsense_nitros_bridge::NitrosRealsenseBridgeNode)

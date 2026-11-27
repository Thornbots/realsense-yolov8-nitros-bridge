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
// Subscribes to TensorRT's output tensor list and publishes
// /cv/armor_keypoints. Subscription choice: the plain
// isaac_ros_tensor_list_interfaces message (the NITROS adapter does the
// device-to-host copy; the ~22 KB output makes that cheaper than the
// NitrosTensorListView dependency and stream handling).
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "dji_serial_bridge/msg/armor_keypoints.hpp"
#include "dji_serial_bridge/msg/armor_keypoints_array.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "realsense_yolov8_nitros_bridge/yolo26_decoder.hpp"

namespace realsense_nitros_bridge
{

class Yolo26DecoderNode : public rclcpp::Node
{
public:
  explicit Yolo26DecoderNode(const rclcpp::NodeOptions & options)
  : rclcpp::Node("yolo26_decoder", options)
  {
    config_.confidence_threshold =
      static_cast<float>(declare_parameter("confidence_threshold", 0.25));
    config_.num_keypoints = static_cast<int>(declare_parameter("num_keypoints", 4));
    config_.max_detections = static_cast<std::size_t>(declare_parameter("max_detections", 0));
    tensor_name_ = declare_parameter("output_tensor_name", std::string("output_tensor"));
    const auto out_topic = declare_parameter("output_topic", std::string("/cv/armor_keypoints"));

    pub_ = create_publisher<dji_serial_bridge::msg::ArmorKeypointsArray>(out_topic, 10);
    // Same relative topic isaac_ros_yolov8's decoder reads from TensorRT.
    sub_ = create_subscription<isaac_ros_tensor_list_interfaces::msg::TensorList>(
      "tensor_sub", 10,
      [this](isaac_ros_tensor_list_interfaces::msg::TensorList::ConstSharedPtr m) {
        on_tensors(*m);
      });
  }

private:
  void on_tensors(const isaac_ros_tensor_list_interfaces::msg::TensorList & list)
  {
    const isaac_ros_tensor_list_interfaces::msg::Tensor * t = nullptr;
    for (const auto & cand : list.tensors) {
      if (cand.name == tensor_name_) {
        t = &cand;
      }
    }
    if (t == nullptr) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000, "no tensor named '%s' in tensor list",
        tensor_name_.c_str());
      return;
    }
    constexpr int kFloat32 = 9;  // TensorList.msg data_type code
    if (t->data_type != kFloat32 || t->data.size() % sizeof(float) != 0) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000, "tensor '%s' is not float32 (data_type %d, %zu bytes)",
        tensor_name_.c_str(), t->data_type, t->data.size());
      return;
    }
    std::vector<float> values(t->data.size() / sizeof(float));
    if (!values.empty()) {
      std::memcpy(values.data(), t->data.data(), t->data.size());
    }
    const std::vector<std::int64_t> dims(t->shape.dims.begin(), t->shape.dims.end());

    const auto result = decode_yolo26(values.data(), values.size(), dims, config_);
    if (!result.ok) {
      // Refuse loudly and publish nothing: an empty array would read as "no plates".
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000, "YOLO26 output rejected: %s (check num_keypoints)",
        result.error.c_str());
      return;
    }

    dji_serial_bridge::msg::ArmorKeypointsArray out;
    out.header = list.header;  // capture stamp from the encoder
    for (const auto & d : result.detections) {
      dji_serial_bridge::msg::ArmorKeypoints m;
      m.header = list.header;
      m.center_x = d.center_x;
      m.center_y = d.center_y;
      m.size_x = d.size_x;
      m.size_y = d.size_y;
      m.confidence = d.confidence;
      m.class_id = d.class_id;
      m.has_keypoints = d.has_keypoints;
      for (int k = 0; k < kArmorKeypointCount; ++k) {
        m.keypoints[k].x = d.keypoints[k].x;
        m.keypoints[k].y = d.keypoints[k].y;
        m.keypoints[k].z = d.keypoints[k].confidence;
      }
      out.detections.push_back(m);
    }
    pub_->publish(out);
  }

  Yolo26Config config_;
  std::string tensor_name_;
  rclcpp::Publisher<dji_serial_bridge::msg::ArmorKeypointsArray>::SharedPtr pub_;
  rclcpp::Subscription<isaac_ros_tensor_list_interfaces::msg::TensorList>::SharedPtr sub_;
};

}  // namespace realsense_nitros_bridge

RCLCPP_COMPONENTS_REGISTER_NODE(realsense_nitros_bridge::Yolo26DecoderNode)

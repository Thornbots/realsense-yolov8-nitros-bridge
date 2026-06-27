// image_snapshot_node.cpp
//
// Timer-driven training image capture with disk-space guardrail.
//
// Files are named <sec>_<nanosec>.<fmt> — unique per frame, sort
// chronologically, never overwrite.
//
// Constructor throws std::runtime_error if the output filesystem is
// already over the disk_limit_pct threshold, preventing the component
// from loading. The save() method also checks periodically (every
// disk_check_interval saves) and cancels the timer if the threshold
// is crossed while running.

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "cv_bridge/cv_bridge.h"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <opencv2/imgcodecs.hpp>

namespace realsense_nitros_bridge
{

class ImageSnapshotNode : public rclcpp::Node
{
public:
  explicit ImageSnapshotNode(const rclcpp::NodeOptions & options)
  : Node("image_snapshot", options), save_count_(0)
  {
    declare_parameter("output_dir",           std::string("/workspaces/isaac_ros-dev/data/realsense-captures"));
    declare_parameter("interval_ms",          500);
    declare_parameter("format",               std::string("jpg"));
    declare_parameter("disk_limit_pct",       75.0);
    declare_parameter("disk_check_interval",  20);   // check every N saves

    output_dir_          = get_parameter("output_dir").as_string();
    format_              = get_parameter("format").as_string();
    interval_ms_         = get_parameter("interval_ms").as_int();
    disk_limit_pct_      = get_parameter("disk_limit_pct").as_double();
    disk_check_interval_ = get_parameter("disk_check_interval").as_int();

    // Create the output directory before the disk check so space() has
    // a valid path to stat — it needs the directory to actually exist
    // on the target filesystem (important for bind-mounted paths in Docker).
    std::filesystem::create_directories(output_dir_);

    // ── Pre-flight disk check ────────────────────────────────────────────
    // Throws here so the ComponentManager refuses to load this component,
    // which propagates as a launch failure before any camera or inference
    // nodes have started.
    check_disk_or_throw();

    // ── Subscription ─────────────────────────────────────────────────────
    // Empty callback: the executor never fires on frames we don't want.
    // take() in the timer below polls for the latest frame at our cadence.
    // QoS matches the realsense publisher: best-effort, volatile, depth 1.
    // depth 1 → stale frames are dropped by the middleware before take() runs.
    auto qos = rclcpp::QoS(1).best_effort().durability_volatile();
    sub_ = create_subscription<sensor_msgs::msg::Image>(
      "image", qos,
      [](sensor_msgs::msg::Image::SharedPtr) {});

    // ── Timer ────────────────────────────────────────────────────────────
    timer_ = create_wall_timer(
      std::chrono::milliseconds(interval_ms_),
      [this]() {
        sensor_msgs::msg::Image::SharedPtr msg;
        rclcpp::MessageInfo info;
        if (sub_->take(msg, info)) {
          save(msg);
        }
      });

    RCLCPP_INFO(get_logger(),
      "ImageSnapshotNode ready — writing %s to '%s' every %d ms (disk limit %.0f%%)",
      format_.c_str(), output_dir_.c_str(), interval_ms_, disk_limit_pct_);
  }

private:
  // ── Disk space helpers ────────────────────────────────────────────────────

  double used_pct() const
  {
    auto sp = std::filesystem::space(output_dir_);
    // space().capacity can be 0 on some virtual filesystems; guard against /0.
    if (sp.capacity == 0) { return 0.0; }
    return 100.0 * (1.0 - static_cast<double>(sp.available) /
                           static_cast<double>(sp.capacity));
  }

  void check_disk_or_throw() const
  {
    double pct = used_pct();
    if (pct > disk_limit_pct_) {
      throw std::runtime_error(
        "ImageSnapshotNode: disk at '" + output_dir_ + "' is " +
        std::to_string(static_cast<int>(pct)) + "% full " +
        "(limit: " + std::to_string(static_cast<int>(disk_limit_pct_)) + "%). "
        "Free space or set a different output_dir / disk_limit_pct.");
    }
  }

  // ── Save ─────────────────────────────────────────────────────────────────

  void save(const sensor_msgs::msg::Image::SharedPtr & msg)
  {
    // Periodic disk check while running — cancel cleanly rather than
    // filling the disk and corrupting the last file.
    if (++save_count_ % disk_check_interval_ == 0) {
      double pct = used_pct();
      if (pct > disk_limit_pct_) {
        RCLCPP_ERROR(get_logger(),
          "Disk %.1f%% full (limit %.0f%%) — stopping capture. "
          "Free space and restart the node to resume.",
          pct, disk_limit_pct_);
        timer_->cancel();
        return;
      }
      RCLCPP_DEBUG(get_logger(), "Disk %.1f%% used after %zu saves",
        pct, save_count_);
    }

    try {
      // toCvShare aliases the message buffer — no pixel copy.
      // Safe here because imwrite() completes before shared_ptr drops.
      // If you ever move the write to a thread pool, switch to toCvCopy.
      auto cv_img = cv_bridge::toCvShare(msg, "bgr8");

      const std::string path =
        output_dir_ + "/" +
        std::to_string(msg->header.stamp.sec) + "_" +
        std::to_string(msg->header.stamp.nanosec) + "." + format_;

      if (!cv::imwrite(path, cv_img->image)) {
        RCLCPP_WARN(get_logger(), "imwrite failed: %s", path.c_str());
      }
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge error: %s", e.what());
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::string output_dir_;
  std::string format_;
  int         interval_ms_;
  double      disk_limit_pct_;
  int         disk_check_interval_;
  std::size_t save_count_;
};

}  // namespace realsense_nitros_bridge

RCLCPP_COMPONENTS_REGISTER_NODE(realsense_nitros_bridge::ImageSnapshotNode)
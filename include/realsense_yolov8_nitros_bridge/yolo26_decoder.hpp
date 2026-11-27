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
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace realsense_nitros_bridge
{

// Number of keypoints carried by dji_serial_bridge/msg/ArmorKeypoints.
constexpr int kArmorKeypointCount = 4;

struct Yolo26Config
{
  float confidence_threshold = 0.25F;
  // 0 = detect model ([1, N, 6]); 4 = pose model ([1, N, 6 + 4*3]).
  int num_keypoints = 0;
  // Keep at most this many rows (highest score first); 0 = no cap.
  std::size_t max_detections = 0;
};

struct Keypoint
{
  float x = 0.0F;
  float y = 0.0F;
  float confidence = 0.0F;
};

// Mirrors ArmorKeypoints without the header; all values in network pixels.
struct ArmorDetection
{
  float center_x = 0.0F;
  float center_y = 0.0F;
  float size_x = 0.0F;
  float size_y = 0.0F;
  float confidence = 0.0F;
  std::int32_t class_id = 0;
  bool has_keypoints = false;
  std::array<Keypoint, kArmorKeypointCount> keypoints{};
};

struct DecodeResult
{
  bool ok = false;
  std::string error;  // set when !ok
  std::vector<ArmorDetection> detections;
};

// Decodes an end-to-end (NMS-free) YOLO26 output of dims [1, N, 6 + 3K];
// each row is x1, y1, x2, y2, score, class, then K x (kx, ky, kconf).
// Rejects any shape, K or element-count mismatch with ok=false.
DecodeResult decode_yolo26(
  const float * data, std::size_t element_count,
  const std::vector<std::int64_t> & dims, const Yolo26Config & config);

}  // namespace realsense_nitros_bridge

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
#include "realsense_yolov8_nitros_bridge/yolo26_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace realsense_nitros_bridge
{

namespace
{

DecodeResult fail(const std::string & why)
{
  DecodeResult r;
  r.error = why;
  return r;
}

}  // namespace

DecodeResult decode_yolo26(
  const float * data, std::size_t element_count,
  const std::vector<std::int64_t> & dims, const Yolo26Config & config)
{
  if (config.num_keypoints != 0 && config.num_keypoints != kArmorKeypointCount) {
    return fail("num_keypoints must be 0 or 4, got " + std::to_string(config.num_keypoints));
  }
  const std::int64_t cols = 6 + 3 * static_cast<std::int64_t>(config.num_keypoints);
  if (dims.size() != 3 || dims[0] != 1 || dims[1] < 0 || dims[2] != cols) {
    std::ostringstream s;
    s << "output shape [";
    for (std::size_t i = 0; i < dims.size(); ++i) {
      s << (i ? ", " : "") << dims[i];
    }
    s << "] does not match [1, N, " << cols << "] for num_keypoints="
      << config.num_keypoints;
    return fail(s.str());
  }
  const auto rows = static_cast<std::size_t>(dims[1]);
  if (element_count != rows * static_cast<std::size_t>(cols) ||
    (data == nullptr && element_count != 0))
  {
    return fail(
      "tensor holds " + std::to_string(element_count) + " floats, shape needs " +
      std::to_string(rows * static_cast<std::size_t>(cols)));
  }

  DecodeResult out;
  out.ok = true;
  for (std::size_t i = 0; i < rows; ++i) {
    const float * row = data + i * static_cast<std::size_t>(cols);
    // Negated compare also drops NaN scores.
    if (!(row[4] >= config.confidence_threshold)) {
      continue;
    }
    if (!(std::isfinite(row[0]) && std::isfinite(row[1]) && std::isfinite(row[2]) &&
      std::isfinite(row[3])))
    {
      continue;
    }
    ArmorDetection d;
    d.center_x = 0.5F * (row[0] + row[2]);
    d.center_y = 0.5F * (row[1] + row[3]);
    d.size_x = row[2] - row[0];
    d.size_y = row[3] - row[1];
    d.confidence = row[4];
    d.class_id = static_cast<std::int32_t>(std::lround(row[5]));
    d.has_keypoints = config.num_keypoints == kArmorKeypointCount;
    for (int k = 0; d.has_keypoints && k < kArmorKeypointCount; ++k) {
      d.keypoints[k] = {row[6 + 3 * k], row[7 + 3 * k], row[8 + 3 * k]};
    }
    out.detections.push_back(d);
  }
  if (config.max_detections > 0 && out.detections.size() > config.max_detections) {
    std::stable_sort(
      out.detections.begin(), out.detections.end(),
      [](const ArmorDetection & a, const ArmorDetection & b) {return a.confidence > b.confidence;});
    out.detections.resize(config.max_detections);
  }
  return out;
}

}  // namespace realsense_nitros_bridge

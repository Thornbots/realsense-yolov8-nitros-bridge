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
#include "realsense_yolov8_nitros_bridge/yolo26_c.h"
#include "realsense_yolov8_nitros_bridge/yolo26_decoder.hpp"

#include <cstdio>
#include <exception>
#include <limits>

extern "C" int64_t yolo26_decode_pose(
  const float * tensor, size_t element_count, int64_t rows, int64_t columns,
  float threshold, float * output, size_t output_rows, char * error, size_t error_size)
{
  try {
    if (rows < 0 || static_cast<uint64_t>(rows) >
      std::numeric_limits<size_t>::max() / 18)
    {
      if (error && error_size) {std::snprintf(error, error_size, "invalid row count");}
      return -1;
    }
    realsense_nitros_bridge::Yolo26Config config;
    config.num_keypoints = 4;
    config.confidence_threshold = threshold;
    const auto decoded = realsense_nitros_bridge::decode_yolo26(
      tensor, element_count, {1, rows, columns}, config);
    if (!decoded.ok || decoded.detections.size() > output_rows ||
      (!decoded.detections.empty() && !output))
    {
      if (error && error_size) {
        std::snprintf(error, error_size, "%s", decoded.ok ? "output buffer too small" :
          decoded.error.c_str());
      }
      return -1;
    }
    for (size_t i = 0; i < decoded.detections.size(); ++i) {
      const auto & d = decoded.detections[i];
      float * row = output + 18 * i;
      row[0] = d.center_x; row[1] = d.center_y;
      row[2] = d.size_x; row[3] = d.size_y;
      row[4] = d.confidence; row[5] = static_cast<float>(d.class_id);
      for (size_t k = 0; k < d.keypoints.size(); ++k) {
        row[6 + 3 * k] = d.keypoints[k].x;
        row[7 + 3 * k] = d.keypoints[k].y;
        row[8 + 3 * k] = d.keypoints[k].confidence;
      }
    }
    return static_cast<int64_t>(decoded.detections.size());
  } catch (const std::exception & e) {
    if (error && error_size) {std::snprintf(error, error_size, "%s", e.what());}
    return -1;
  } catch (...) {
    if (error && error_size) {std::snprintf(error, error_size, "decoder failed");}
    return -1;
  }
}

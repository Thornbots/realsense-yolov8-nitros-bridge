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

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Each output row contains cx, cy, width, height, score, class, then four
// (x, y, confidence) keypoints. Returns row count, or -1 on invalid input.
int64_t yolo26_decode_pose(
  const float * tensor, size_t element_count,
  int64_t rows, int64_t columns, float threshold, float * output, size_t output_rows,
  char * error, size_t error_size);

#ifdef __cplusplus
}
#endif

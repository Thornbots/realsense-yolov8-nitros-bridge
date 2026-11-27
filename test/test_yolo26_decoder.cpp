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
#include <gtest/gtest.h>

#include <vector>

#include "realsense_yolov8_nitros_bridge/yolo26_decoder.hpp"

using realsense_nitros_bridge::decode_yolo26;
using realsense_nitros_bridge::Yolo26Config;

namespace
{

std::vector<float> pose_row(float score, float cls, float x1 = 100, float y1 = 200)
{
  return {x1, y1, x1 + 40, y1 + 20, score, cls,
    x1, y1, 0.9F, x1 + 40, y1, 0.8F, x1 + 40, y1 + 20, 0.7F, x1, y1 + 20, 0.6F};
}

std::vector<float> concat(const std::vector<std::vector<float>> & rows)
{
  std::vector<float> v;
  for (const auto & r : rows) {
    v.insert(v.end(), r.begin(), r.end());
  }
  return v;
}

}  // namespace

TEST(Yolo26Decoder, DetectRowsFilteredByThreshold)
{
  const std::vector<float> t = {
    10, 20, 50, 80, 0.9F, 3,
    0, 0, 5, 5, 0.1F, 1,
    60, 60, 100, 100, 0.25F, 7};
  Yolo26Config c;
  const auto r = decode_yolo26(t.data(), t.size(), {1, 3, 6}, c);
  ASSERT_TRUE(r.ok) << r.error;
  ASSERT_EQ(r.detections.size(), 2U);
  EXPECT_FLOAT_EQ(r.detections[0].center_x, 30.0F);
  EXPECT_FLOAT_EQ(r.detections[0].center_y, 50.0F);
  EXPECT_FLOAT_EQ(r.detections[0].size_x, 40.0F);
  EXPECT_FLOAT_EQ(r.detections[0].size_y, 60.0F);
  EXPECT_EQ(r.detections[0].class_id, 3);
  EXPECT_FALSE(r.detections[0].has_keypoints);
  EXPECT_EQ(r.detections[1].class_id, 7);  // score == threshold is kept
}

TEST(Yolo26Decoder, PoseKeepsKeypointsAndConfidence)
{
  const auto t = concat({pose_row(0.8F, 2), pose_row(0.05F, 1)});
  Yolo26Config c;
  c.num_keypoints = 4;
  const auto r = decode_yolo26(t.data(), t.size(), {1, 2, 18}, c);
  ASSERT_TRUE(r.ok) << r.error;
  ASSERT_EQ(r.detections.size(), 1U);
  const auto & d = r.detections[0];
  EXPECT_TRUE(d.has_keypoints);
  EXPECT_EQ(d.class_id, 2);
  EXPECT_FLOAT_EQ(d.keypoints[0].x, 100.0F);
  EXPECT_FLOAT_EQ(d.keypoints[0].confidence, 0.9F);
  EXPECT_FLOAT_EQ(d.keypoints[2].x, 140.0F);
  EXPECT_FLOAT_EQ(d.keypoints[2].y, 220.0F);
  EXPECT_FLOAT_EQ(d.keypoints[3].confidence, 0.6F);
}

TEST(Yolo26Decoder, AllBelowThresholdIsOkAndEmpty)
{
  const auto t = concat({pose_row(0.01F, 0), pose_row(0.02F, 1)});
  Yolo26Config c;
  c.num_keypoints = 4;
  const auto r = decode_yolo26(t.data(), t.size(), {1, 2, 18}, c);
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(r.detections.empty());
}

TEST(Yolo26Decoder, ZeroRowsIsOk)
{
  Yolo26Config c;
  const auto r = decode_yolo26(nullptr, 0, {1, 0, 6}, c);
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(r.detections.empty());
}

TEST(Yolo26Decoder, ShapeMismatchIsReported)
{
  const auto t = concat({pose_row(0.8F, 0)});
  Yolo26Config detect;  // pose tensor fed to a detect config
  auto r = decode_yolo26(t.data(), t.size(), {1, 1, 18}, detect);
  EXPECT_FALSE(r.ok);
  EXPECT_NE(r.error.find("[1, 1, 18]"), std::string::npos) << r.error;

  Yolo26Config pose;
  pose.num_keypoints = 4;
  EXPECT_FALSE(decode_yolo26(t.data(), t.size(), {1, 1, 6}, pose).ok);   // wrong columns
  EXPECT_FALSE(decode_yolo26(t.data(), t.size(), {2, 1, 18}, pose).ok);  // batch != 1
  EXPECT_FALSE(decode_yolo26(t.data(), t.size(), {1, 18}, pose).ok);     // rank 2
  EXPECT_FALSE(decode_yolo26(t.data(), t.size() - 1, {1, 1, 18}, pose).ok);  // short buffer
  pose.num_keypoints = 3;
  EXPECT_FALSE(decode_yolo26(t.data(), t.size(), {1, 1, 15}, pose).ok);  // unsupported K
}

TEST(Yolo26Decoder, MaxDetectionsKeepsHighestScores)
{
  const std::vector<float> t = {
    0, 0, 2, 2, 0.4F, 0,
    0, 0, 2, 2, 0.9F, 1,
    0, 0, 2, 2, 0.6F, 2};
  Yolo26Config c;
  c.max_detections = 2;
  const auto r = decode_yolo26(t.data(), t.size(), {1, 3, 6}, c);
  ASSERT_TRUE(r.ok);
  ASSERT_EQ(r.detections.size(), 2U);
  EXPECT_EQ(r.detections[0].class_id, 1);
  EXPECT_EQ(r.detections[1].class_id, 2);
}

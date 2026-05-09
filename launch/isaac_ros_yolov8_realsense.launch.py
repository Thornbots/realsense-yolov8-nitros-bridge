# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
isaac_ros_yolov8_realsense.launch.py

Starts a D435i and connects it to the isaac_ros YOLOv8 TensorRT pipeline,
then feeds the best detection into the ROI depth node via a relay node.

── Topic layout ──────────────────────────────────────────────────────────────

  Camera (name=camera, namespace=''):
    /camera/color/image_raw          → dnn_image_encoder
    /camera/color/camera_info        → roi_depth_node (LUT build)
    /camera/depth/image_rect_raw     → roi_depth_node (sampling)
    /camera/depth/camera_info        → roi_depth_node (LUT build)
    /camera/camera/extrinsics/...    → extrinsics_relay_node → roi_depth_node params

  Inference chain:
    /tensor_pub       (dnn_image_encoder → tensor_rt)
    /tensor_sub       (tensor_rt → yolov8_decoder_node)
    /detections_output  (yolov8_decoder_node → detection_picker_node)
      ↳ Detection2DArray in 640×640 NETWORK space

  Relay node (detection_picker_node):
    /detections_output → /roi
      ↳ Picks highest-confidence detection, scales bbox to color image space

  ROI depth node:
    /roi + /camera/depth/image_rect_raw → /roi_depth_m (mean depth in metres)

── Usage ─────────────────────────────────────────────────────────────────────

  ros2 launch realsense_yolov8_nitros_bridge isaac_ros_yolov8_realsense.launch.py \\
      engine_file_path:=/path/to/model.plan \\
      [json_file_path:=/path/to/realsense.json] \\
      [input_image_width:=640] [input_image_height:=480] \\
      [confidence_threshold:=0.25] [nms_threshold:=0.45] \\
      [center_sample_fraction:=0.25]
"""

import json
import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import Node as LaunchNode


# ── Topic roots ───────────────────────────────────────────────────────────────
# With ComposableNode(name='camera', namespace=''), realsense-ros publishes:
#   /camera/color/image_raw, /camera/depth/image_rect_raw, etc.
# The extrinsics topic retains realsense-ros's internal 'camera' sub-namespace:
#   /camera/camera/extrinsics/depth_to_color
REALSENSE_COLOR_TOPIC    = '/camera/color/image_raw'
REALSENSE_INFO_TOPIC     = '/camera/color/camera_info'
REALSENSE_DEPTH_NS       = '/camera/depth'
REALSENSE_COLOR_NS       = '/camera/color'
REALSENSE_EXTRINSICS_TOPIC = '/camera/camera/extrinsics/depth_to_color'

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_INPUT_W   = '640'
DEFAULT_INPUT_H   = '480'
DEFAULT_NETWORK_W = '640'
DEFAULT_NETWORK_H = '640'


def generate_launch_description():

    launch_args = [
        DeclareLaunchArgument(
            'json_file_path',
            default_value=os.path.join(
                get_package_share_directory('realsense_yolov8_nitros_bridge'),
                'config', 'realsense_640x480x60.json'),
            description='Absolute path to realsense-ros JSON config (profile, format, QoS)'),
        DeclareLaunchArgument('serial_no', default_value='',
                              description='Select camera by serial number (empty = any)'),
        DeclareLaunchArgument('input_image_width',  default_value=DEFAULT_INPUT_W,
                              description='Color image width (must match JSON profile)'),
        DeclareLaunchArgument('input_image_height', default_value=DEFAULT_INPUT_H,
                              description='Color image height (must match JSON profile)'),
        DeclareLaunchArgument('network_image_width',  default_value=DEFAULT_NETWORK_W,
                              description='TensorRT model input width'),
        DeclareLaunchArgument('network_image_height', default_value=DEFAULT_NETWORK_H,
                              description='TensorRT model input height'),
        DeclareLaunchArgument('image_mean',    default_value='[0.0, 0.0, 0.0]'),
        DeclareLaunchArgument('image_stddev',  default_value='[1.0, 1.0, 1.0]'),
        DeclareLaunchArgument('input_encoding', default_value='rgb8'),
        DeclareLaunchArgument('model_file_path',     default_value=''),
        DeclareLaunchArgument('engine_file_path',    default_value=''),
        DeclareLaunchArgument('input_tensor_names',  default_value='["input_tensor"]'),
        DeclareLaunchArgument('input_binding_names', default_value='["images"]'),
        DeclareLaunchArgument('output_tensor_names', default_value='["output_tensor"]'),
        DeclareLaunchArgument('output_binding_names', default_value='["output0"]'),
        DeclareLaunchArgument('verbose',             default_value='False'),
        DeclareLaunchArgument('force_engine_update', default_value='False'),
        DeclareLaunchArgument('confidence_threshold', default_value='0.25'),
        DeclareLaunchArgument('nms_threshold',        default_value='0.45'),
        # Depth sampling: fraction of bbox center to average (0.25 = inner 25% per axis)
        DeclareLaunchArgument('center_sample_fraction', default_value='0.25',
                              description='Fraction of bbox center to sample for depth (0.05–1.0)'),
        DeclareLaunchArgument('min_detection_score', default_value='0.0',
                              description='Relay node ignores detections below this confidence'),
    ]

    def create_nodes(context):
        json_file_path  = LaunchConfiguration('json_file_path').perform(context)
        input_w         = LaunchConfiguration('input_image_width').perform(context)
        input_h         = LaunchConfiguration('input_image_height').perform(context)
        network_w       = LaunchConfiguration('network_image_width').perform(context)
        network_h       = LaunchConfiguration('network_image_height').perform(context)
        image_mean      = LaunchConfiguration('image_mean').perform(context)
        image_stddev    = LaunchConfiguration('image_stddev').perform(context)
        encoding        = LaunchConfiguration('input_encoding').perform(context)

        model_file_path      = LaunchConfiguration('model_file_path').perform(context)
        engine_file_path     = LaunchConfiguration('engine_file_path').perform(context)
        input_tensor_names   = json.loads(LaunchConfiguration('input_tensor_names').perform(context))
        input_binding_names  = json.loads(LaunchConfiguration('input_binding_names').perform(context))
        output_tensor_names  = json.loads(LaunchConfiguration('output_tensor_names').perform(context))
        output_binding_names = json.loads(LaunchConfiguration('output_binding_names').perform(context))
        verbose              = LaunchConfiguration('verbose').perform(context) == 'True'
        force_engine_update  = LaunchConfiguration('force_engine_update').perform(context) == 'True'
        confidence_threshold = float(LaunchConfiguration('confidence_threshold').perform(context))
        nms_threshold        = float(LaunchConfiguration('nms_threshold').perform(context))
        center_sample_frac   = float(LaunchConfiguration('center_sample_fraction').perform(context))
        min_det_score        = float(LaunchConfiguration('min_detection_score').perform(context))

        if not os.path.isfile(json_file_path):
            raise FileNotFoundError(
                f'[isaac_ros_yolov8_realsense] JSON config not found: {json_file_path}')

        print(f'[isaac_ros_yolov8_realsense] JSON config: {json_file_path}')
        print(f'[isaac_ros_yolov8_realsense] Color: {input_w}x{input_h} → network: {network_w}x{network_h}')
        print(f'[isaac_ros_yolov8_realsense] Depth center_sample_fraction: {center_sample_frac}')

        # ── RealSense composable node ─────────────────────────────────────────
        # Parameters come from YAML (applied at construction time before sensor start).
        # JSON path is also read early by realsense-ros, so it wins the race.
        realsense_node = ComposableNode(
            package='realsense2_camera',
            plugin='realsense2_camera::RealSenseNodeFactory',
            name='camera',
            namespace='',
            parameters=[
                os.path.join(
                    get_package_share_directory('realsense_yolov8_nitros_bridge'),
                    'config', 'realsense_640x480x60.yaml'),
            ],
            extra_arguments=[{'use_intra_process_comms': True}],
        )

        # ── TensorRT inference node ───────────────────────────────────────────
        tensor_rt_node = ComposableNode(
            name='tensor_rt',
            package='isaac_ros_tensor_rt',
            plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
            parameters=[{
                'model_file_path':      model_file_path,
                'engine_file_path':     engine_file_path,
                'output_binding_names': output_binding_names,
                'output_tensor_names':  output_tensor_names,
                'input_tensor_names':   input_tensor_names,
                'input_binding_names':  input_binding_names,
                'verbose':              verbose,
                'force_engine_update':  force_engine_update,
            }],
        )

        # ── YOLOv8 decoder node ───────────────────────────────────────────────
        # Publishes: /detections_output (Detection2DArray, network image space)
        yolov8_decoder_node = ComposableNode(
            name='yolov8_decoder_node',
            package='isaac_ros_yolov8',
            plugin='nvidia::isaac_ros::yolov8::YoloV8DecoderNode',
            parameters=[{
                'confidence_threshold': confidence_threshold,
                'nms_threshold':        nms_threshold,
            }],
        )

        # ── Detection ROI relay node ──────────────────────────────────────────
        # Subscribes: /detections_output (Detection2DArray, 640×640 network space)
        # Publishes:  /roi              (Detection2D,      color image space)
        # Applies scale: x *= color_w/network_w, y *= color_h/network_h
        detection_picker_node = ComposableNode(
            package='roi_depth_query',
            plugin='roi_depth_query::DetectionRoiRelayNode',
            name='detection_picker_node',
            parameters=[{
                'detections_topic': '/detections_output',
                'roi_topic':        '/roi',
                'network_width':    int(network_w),
                'network_height':   int(network_h),
                'color_width':      int(input_w),
                'color_height':     int(input_h),
                'min_score':        min_det_score,
            }],
            extra_arguments=[{'use_intra_process_comms': True}],
        )

        # ── ROI depth node ────────────────────────────────────────────────────
        # Subscribes: /roi (Detection2D, color image space) + depth streams
        # Publishes:  /roi_depth_m (mean depth in metres)
        # NOTE: depth_ns and color_ns must match realsense-ros topic layout.
        # With name='camera', namespace='', realsense publishes at:
        #   /camera/depth/image_rect_raw, /camera/color/camera_info, etc.
        roi_depth_node = ComposableNode(
            package='roi_depth_query',
            plugin='roi_depth_query::RoiDepthNode',
            name='roi_depth_node',
            parameters=[{
                'depth_ns':               REALSENSE_DEPTH_NS,
                'color_ns':               REALSENSE_COLOR_NS,
                'extrinsics_topic':       REALSENSE_EXTRINSICS_TOPIC,
                'depth_scale':            0.001,   # D435i Z16 default (mm → m)
                'min_depth_m':            0.1,
                'max_depth_m':            10.0,
                'center_sample_fraction': center_sample_frac,
            }],
            extra_arguments=[{'use_intra_process_comms': True}],
        )

        # ── Shared component container ────────────────────────────────────────
        container = ComposableNodeContainer(
            name='yolov8_realsense_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[
                realsense_node,
                tensor_rt_node,
                yolov8_decoder_node,
                detection_picker_node,
                roi_depth_node,
            ],
            output='screen',
            arguments=['--ros-args', '--log-level', 'INFO'],
        )

        # ── DNN Image Encoder (loaded into the shared container) ──────────────
        encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
        yolov8_encoder_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')
            ),
            launch_arguments={
                'image_input_topic':       REALSENSE_COLOR_TOPIC,
                'camera_info_input_topic': REALSENSE_INFO_TOPIC,
                'tensor_output_topic':     '/tensor_pub',
                'input_image_width':       input_w,
                'input_image_height':      input_h,
                'network_image_width':     network_w,
                'network_image_height':    network_h,
                'image_mean':              image_mean,
                'image_stddev':            image_stddev,
                'input_encoding':          encoding,
                'attach_to_shared_component_container': 'True',
                'component_container_name':             'yolov8_realsense_container',
                'dnn_image_encoder_namespace':          'yolov8_encoder',
            }.items(),
        )

        # ── Extrinsics relay (standalone: subscribes transient_local, pushes params) ──
        extrinsics_relay = LaunchNode(
            package='roi_depth_query',
            executable='extrinsics_relay_node',
            name='extrinsics_relay',
            parameters=[{
                'extrinsics_topic': REALSENSE_EXTRINSICS_TOPIC,
                'target_node':      '/roi_depth_node',
            }],
            output='screen',
        )

        return [container, yolov8_encoder_launch, extrinsics_relay]

    return launch.LaunchDescription(launch_args + [OpaqueFunction(function=create_nodes)])

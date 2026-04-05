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

Starts a D435i (or any D4xx) and connects it to the isaac_ros YOLOv8
TensorRT pipeline. All nodes share a single component_container_mt so
that ROS 2 intra-process communication (IPC) is available between them.

─── Topic layout ────────────────────────────────────────────────────────────

The published topic path depends on how the realsense node is loaded:

  Via rs_launch.py (plain Node, camera_namespace='camera', camera_name='camera'):
    FQN = /camera/camera  →  /camera/camera/color/image_raw   (docs default)

  Via isaac_ros_realsense ComposableNode (namespace='', name='camera'):
    The component loader injects __ns:='' which overrides the hardcoded
    "/camera" absolute namespace in RealSenseNodeFactory's constructor,
    leaving the node in the root namespace with only its name "camera".
    Effective FQN = /camera  →  /camera/color/image_raw       (actual behaviour)

This file matches the isaac_ros_realsense / ComposableNode behaviour,
which is what ros-humble-isaac-ros-realsense (installed in the Dockerfile)
produces.  If you are instead running a standalone rs_launch.py with its
default camera_namespace='camera', change the two REALSENSE_*_TOPIC
constants to the /camera/camera/... variants.

─── Usage ───────────────────────────────────────────────────────────────────

  ros2 launch realsense_yolov8_nitros_bridge isaac_ros_yolov8_realsense.launch.py \\
      engine_file_path:=/path/to/yolov8n.plan \\
      [input_image_width:=640] [input_image_height:=480] \\
      [confidence_threshold:=0.25] [nms_threshold:=0.45]
"""

import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes
from launch_ros.descriptions import ComposableNode


# ── Topic roots ───────────────────────────────────────────────────────────────
# ComposableNode(namespace='') causes the component loader to inject __ns:=''
# which overrides the hardcoded "/camera" absolute namespace inside
# RealSenseNodeFactory, leaving FQN=/camera and topics at /camera/color/...
REALSENSE_COLOR_TOPIC = '/camera/color/image_raw'
REALSENSE_INFO_TOPIC  = '/camera/color/camera_info'

DEFAULT_INPUT_W   = '640'
DEFAULT_INPUT_H   = '480'
DEFAULT_INPUT_FPS = '60'


def generate_launch_description():

    # ── Launch arguments ──────────────────────────────────────────────────────
    launch_args = [
        DeclareLaunchArgument('serial_no', default_value='',
                              description='Select D435i by serial number (empty = any)'),
        DeclareLaunchArgument('input_image_width',  default_value=DEFAULT_INPUT_W,
                              description='Color stream width'),
        DeclareLaunchArgument('input_image_height', default_value=DEFAULT_INPUT_H,
                              description='Color stream height'),
        DeclareLaunchArgument('color_fps', default_value=DEFAULT_INPUT_FPS,
                              description='Color stream FPS'),
        DeclareLaunchArgument('network_image_width',  default_value='640'),
        DeclareLaunchArgument('network_image_height', default_value='640'),
        DeclareLaunchArgument('image_mean',    default_value='[0.0, 0.0, 0.0]'),
        DeclareLaunchArgument('image_stddev',  default_value='[1.0, 1.0, 1.0]'),
        DeclareLaunchArgument('input_encoding', default_value='rgb8'),
        DeclareLaunchArgument('model_file_path',     default_value=''),
        DeclareLaunchArgument('engine_file_path',    default_value=''),
        DeclareLaunchArgument('input_tensor_names',  default_value='["input_tensor"]'),
        DeclareLaunchArgument('input_binding_names', default_value='["images"]'),
        DeclareLaunchArgument('output_tensor_names', default_value='["output_tensor"]'),
        DeclareLaunchArgument('output_binding_names',default_value='["output0"]'),
        DeclareLaunchArgument('verbose',             default_value='False'),
        DeclareLaunchArgument('force_engine_update', default_value='False'),
        DeclareLaunchArgument('confidence_threshold', default_value='0.25'),
        DeclareLaunchArgument('nms_threshold',        default_value='0.45'),
    ]

    def create_nodes(context):
        """
        OpaqueFunction so all LaunchConfigurations are fully resolved to plain
        Python strings before we use them.  This is required for the profile
        string: passing a list or a Substitution object to a string ROS parameter
        produces a YAML-array representation ("[640, x, 480, x, 60]") that does
        NOT match the realsense-ros regex parser, causing a silent fallback to
        the hardware default FPS.
        """
        # Resolve every argument to a concrete string
        serial_no  = LaunchConfiguration('serial_no').perform(context)
        input_w    = LaunchConfiguration('input_image_width').perform(context)
        input_h    = LaunchConfiguration('input_image_height').perform(context)
        color_fps  = LaunchConfiguration('color_fps').perform(context)
        network_w  = LaunchConfiguration('network_image_width').perform(context)
        network_h  = LaunchConfiguration('network_image_height').perform(context)
        image_mean   = LaunchConfiguration('image_mean').perform(context)
        image_stddev = LaunchConfiguration('image_stddev').perform(context)
        encoding     = LaunchConfiguration('input_encoding').perform(context)

        model_file_path      = LaunchConfiguration('model_file_path').perform(context)
        engine_file_path     = LaunchConfiguration('engine_file_path').perform(context)
        input_tensor_names   = LaunchConfiguration('input_tensor_names').perform(context)
        input_binding_names  = LaunchConfiguration('input_binding_names').perform(context)
        output_tensor_names  = LaunchConfiguration('output_tensor_names').perform(context)
        output_binding_names = LaunchConfiguration('output_binding_names').perform(context)
        verbose              = LaunchConfiguration('verbose').perform(context)
        force_engine_update  = LaunchConfiguration('force_engine_update').perform(context)
        confidence_threshold = LaunchConfiguration('confidence_threshold').perform(context)
        nms_threshold        = LaunchConfiguration('nms_threshold').perform(context)

        # Build the profile string the way realsense-ros expects it:
        #   "<width>x<height>x<fps>"
        # The regex in profile_manager.cpp is:
        #   \s*([0-9]+)\s*[xX,]\s*([0-9]+)\s*[xX,]\s*([0-9]+)\s*
        # Any other format (including Python list serialisation) silently falls
        # back to the hardware default FPS.
        color_profile_str = f'{input_w}x{input_h}x{color_fps}'

        # ── RealSense composable node ─────────────────────────────────────────
        realsense_node = ComposableNode(
            package='realsense2_camera',
            plugin='realsense2_camera::RealSenseNodeFactory',
            name='camera',
            namespace='',
            parameters=[{
                'serial_no':                 serial_no,
                'enable_color':              True,
                'enable_depth':              False,
                'enable_infra1':             False,
                'enable_infra2':             False,
                'enable_gyro':               False,
                'enable_accel':              False,
                # Plain "WxHxFPS" string — the only format the regex accepts
                'rgb_camera.color_profile':  color_profile_str,
                'rgb_camera.color_format':   'RGB8',
                # DEFAULT = rmw_qos_profile_default = explicitly VOLATILE,
                # required for rclcpp intra-process communication.
                'color_qos':                 'DEFAULT',
            }],
            extra_arguments=[{'use_intra_process_comms': True}],
        )

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
                'verbose':              verbose == 'True',
                'force_engine_update':  force_engine_update == 'True',
            }],
        )

        yolov8_decoder_node = ComposableNode(
            name='yolov8_decoder_node',
            package='isaac_ros_yolov8',
            plugin='nvidia::isaac_ros::yolov8::YoloV8DecoderNode',
            parameters=[{
                'confidence_threshold': float(confidence_threshold),
                'nms_threshold':        float(nms_threshold),
            }],
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
            ],
            output='screen',
            arguments=['--ros-args', '--log-level', 'INFO'],
        )

        # ── DNN Image Encoder ─────────────────────────────────────────────────
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

        return [container, yolov8_encoder_launch]

    return launch.LaunchDescription(launch_args + [OpaqueFunction(function=create_nodes)])
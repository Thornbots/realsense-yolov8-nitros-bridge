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
TensorRT pipeline.  All nodes share a single component_container_mt so
that ROS 2 intra-process communication (IPC) is available between them.

─── Camera configuration via JSON ───────────────────────────────────────────

Camera stream parameters (resolution, FPS, format, QoS) are loaded from a
JSON file rather than passed as individual ROS parameters.  This matters
because realsense-ros reads stream parameters during its internal
getParameters() call at node construction time.  Parameters arriving
through the ComposableNode parameter dict may be processed AFTER the first
sensor start, causing a Stop/Start cycle that resets the profile to the
hardware default (1280×720@30).

The JSON file is read once before any sensor is started, so the profile is
applied on the one and only hardware start — no Stop/Start cycle, no
fallback.

Default config file: share/realsense_yolov8_nitros_bridge/config/realsense_640x480x60.json
Override at launch time:
    json_file_path:=/absolute/path/to/your.json

─── QoS compatibility with NITROS ───────────────────────────────────────────

The NITROS ResizeNode subscriber uses RELIABLE reliability.
realsense-ros defaults to SENSOR_DATA (BEST_EFFORT), which is incompatible.
The JSON config sets color_qos to "DEFAULT" (RELIABLE + VOLATILE), which:
  • matches the NITROS subscriber QoS, allowing the connection to form
  • satisfies the VOLATILE durability requirement for rclcpp IPC

─── Topic layout ────────────────────────────────────────────────────────────

ComposableNode(namespace='', name='camera') leaves FQN=/camera, so topics
are at /camera/color/image_raw (not /camera/camera/color/image_raw).
Change REALSENSE_*_TOPIC below if using standalone rs_launch.py instead.

─── Usage ───────────────────────────────────────────────────────────────────

  ros2 launch realsense_yolov8_nitros_bridge isaac_ros_yolov8_realsense.launch.py \\
      engine_file_path:=/path/to/model.plan \\
      [json_file_path:=/path/to/realsense.json] \\
      [input_image_width:=640] [input_image_height:=480] \\
      [confidence_threshold:=0.25] [nms_threshold:=0.45]
"""

"""
ros2 launch realsense_yolov8_nitros_bridge isaac_ros_yolov8_realsense.launch.py engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolo11/yolo11s_fp16.plan
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


# ── Topic roots ───────────────────────────────────────────────────────────────
REALSENSE_COLOR_TOPIC = '/color/image_raw'
REALSENSE_INFO_TOPIC  = '/color/camera_info'

# ── Defaults that mirror the shipped JSON config ──────────────────────────────
DEFAULT_INPUT_W   = '640'
DEFAULT_INPUT_H   = '480'
DEFAULT_INPUT_FPS = '60'


def generate_launch_description():

    # ── Launch arguments ──────────────────────────────────────────────────────
    launch_args = [
        DeclareLaunchArgument(
            'json_file_path',
            default_value=os.path.join(
                get_package_share_directory('realsense_yolov8_nitros_bridge'),
                'config', 'realsense_640x480x60.json'),
            description=(
                'Absolute path to a realsense-ros JSON parameter file. '
                'Stream profile, format, and QoS settings should all live '
                'here so they are applied before the first sensor start.')),
        DeclareLaunchArgument('serial_no', default_value='',
                              description='Select D435i by serial number (empty = any)'),
        DeclareLaunchArgument('input_image_width',  default_value=DEFAULT_INPUT_W,
                              description='Must match rgb_camera.color_profile width in the JSON'),
        DeclareLaunchArgument('input_image_height', default_value=DEFAULT_INPUT_H,
                              description='Must match rgb_camera.color_profile height in the JSON'),
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
        DeclareLaunchArgument('output_binding_names', default_value='["output0"]'),
        DeclareLaunchArgument('verbose',             default_value='False'),
        DeclareLaunchArgument('force_engine_update', default_value='False'),
        DeclareLaunchArgument('confidence_threshold', default_value='0.25'),
        DeclareLaunchArgument('nms_threshold',        default_value='0.45'),
    ]

    def create_nodes(context):
        # Resolve all launch arguments to plain Python strings/values.
        json_file_path = LaunchConfiguration('json_file_path').perform(context)
        serial_no      = LaunchConfiguration('serial_no').perform(context)
        input_w        = LaunchConfiguration('input_image_width').perform(context)
        input_h        = LaunchConfiguration('input_image_height').perform(context)
        network_w      = LaunchConfiguration('network_image_width').perform(context)
        network_h      = LaunchConfiguration('network_image_height').perform(context)
        image_mean     = LaunchConfiguration('image_mean').perform(context)
        image_stddev   = LaunchConfiguration('image_stddev').perform(context)
        encoding       = LaunchConfiguration('input_encoding').perform(context)

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

        # ── Validate JSON file ────────────────────────────────────────────────
        if not os.path.isfile(json_file_path):
            raise FileNotFoundError(
                f'[isaac_ros_yolov8_realsense] JSON config not found: {json_file_path}\n'
                f'  Install the package (colcon build) so the config is copied to share/,\n'
                f'  or pass json_file_path:=/absolute/path/to/config.json at launch.')

        print(f'[isaac_ros_yolov8_realsense] Using realsense JSON config: {json_file_path}')
        print(f'[isaac_ros_yolov8_realsense] Encoder input: {input_w}x{input_h}'
              f'  →  network: {network_w}x{network_h}')

        # ── RealSense composable node ─────────────────────────────────────────
        #
        # IMPORTANT: stream parameters (profile, format, QoS) live in the JSON
        # file, NOT in the parameters dict below.
        #
        # Why: realsense-ros calls getParameters() inside the node constructor,
        # before the ROS parameter server has applied the ComposableNode
        # parameter dict.  Any stream parameter set here arrives late, triggers
        # a "re-enable stream for change to take effect" warning, causes a
        # Stop/Start cycle, and the profile reverts to the hardware default.
        #
        # The JSON path IS read during getParameters() so it wins the race.
        # Only serial_no is safe to pass here because it is used for device
        # selection before getParameters() runs.
        realsense_node = ComposableNode(
            package='realsense2_camera',
            plugin='realsense2_camera::RealSenseNodeFactory',
            name='camera',
            namespace='',
            parameters=[
                    os.path.join(get_package_share_directory('realsense_yolov8_nitros_bridge'), 'config', 'realsense_640x480x60.yaml'),
                ],
            extra_arguments=[{'use_intra_process_comms': True}],
        )
        roi_depth = ComposableNode(
        package="roi_depth_query",
        executable="roi_depth_node",
        name="roi_depth_node",
        output="screen",
        parameters=[{
            "depth_ns":        "/camera/depth",
            "color_ns":        "/camera/color",
            "extrinsics_topic": "/camera/camera/extrinsics/depth_to_color",
            "depth_scale":     0.001,   # D435i Z16 default (,, to m)
            "min_depth_m":     0.1,
            "max_depth_m":     10.0,
        }],
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
        yolov8_decoder_node = ComposableNode(
            name='yolov8_decoder_node',
            package='isaac_ros_yolov8',
            plugin='nvidia::isaac_ros::yolov8::YoloV8DecoderNode',
            parameters=[{
                'confidence_threshold': confidence_threshold,
                'nms_threshold':        nms_threshold,
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

        return [container, yolov8_encoder_launch]

    return launch.LaunchDescription(launch_args + [OpaqueFunction(function=create_nodes)])
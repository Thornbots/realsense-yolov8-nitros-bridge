# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
isaac_ros_yolov8_realsense.launch.py

── Actual topic layout (verified from runtime log) ──────────────────────────

  With ComposableNode(name='camera', namespace=''), realsense-ros resolves
  all topics against namespace '' (root). There is NO /camera/ prefix:

    /color/image_raw               → dnn_image_encoder
    /color/camera_info             → roi_depth_node (LUT build)
    /depth/image_rect_raw          → roi_depth_node (sampling)
    /depth/camera_info             → roi_depth_node (LUT build)
    /extrinsics/depth_to_color     → extrinsics_relay_node → roi_depth_node params

  NOTE: If you launch with an explicit namespace (e.g. namespace='camera'),
  all topics will gain a /camera/ prefix and these constants must be updated.

── Inference chain ───────────────────────────────────────────────────────────

  /color/image_raw
    → dnn_image_encoder (resize 640×480 → 640×640, normalise, interleave→planar)
    → /tensor_pub → tensor_rt (TensorRT YOLOv8 inference)
    → /tensor_sub → yolov8_decoder_node
    → /detections_output  (Detection2DArray, bbox in 640×640 NETWORK space)
    → detection_picker_node  (picks best, scales bbox to color image space)
    → /roi  (Detection2D, bbox in 640×480 COLOR image space)
    → roi_depth_node  (LUT lookup + center-sample depth)
    → /roi_point  (geometry_msgs/PointStamped, REP-103 camera body frame)
    → point_to_cv_target_node  (dji_serial_bridge package — frame convert +
                                 finite-difference velocity/acceleration)
    → /cv_target  (dji_serial_bridge/msg/CVTarget)
    → dji_serial_bridge_node  → UART → MCB / gimbal controller

  Set enable_serial_bridge:=false to omit the last two nodes (e.g. when
  bench-testing the vision pipeline without the MCB attached).

── Usage ─────────────────────────────────────────────────────────────────────

  ros2 launch realsense_yolov8_nitros_bridge isaac_ros_yolov8_realsense.launch.py \
      engine_file_path:=/path/to/model.plan \
      [confidence_threshold:=0.25] [nms_threshold:=0.45] \
      [center_sample_fraction:=0.25] \
      [serial_device:=/dev/ttyTHS1] [serial_baudrate:=115200]
"""

import json
import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import Node as LaunchNode


# ── Topic roots ───────────────────────────────────────────────────────────────
# Verified against runtime: with name='camera', namespace='', all realsense-ros
# topics resolve to root (no /camera/ prefix).
REALSENSE_COLOR_TOPIC      = '/color/image_raw'
REALSENSE_INFO_TOPIC       = '/color/camera_info'
REALSENSE_DEPTH_NS         = '/depth'
REALSENSE_COLOR_NS         = '/color'
REALSENSE_EXTRINSICS_TOPIC = '/extrinsics/depth_to_color'

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_INPUT_W   = '640'
DEFAULT_INPUT_H   = '480'
DEFAULT_NETWORK_W = '640'
DEFAULT_NETWORK_H = '640'


def generate_launch_description():

    launch_args = [
        DeclareLaunchArgument('input_image_width',  default_value=DEFAULT_INPUT_W),
        DeclareLaunchArgument('input_image_height', default_value=DEFAULT_INPUT_H),
        DeclareLaunchArgument('network_image_width',  default_value=DEFAULT_NETWORK_W),
        DeclareLaunchArgument('network_image_height', default_value=DEFAULT_NETWORK_H),
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
        DeclareLaunchArgument('center_sample_fraction', default_value='0.25',
            description='Fraction of bbox center to sample for depth (0.05–1.0)'),
        DeclareLaunchArgument('min_detection_score', default_value='0.0',
            description='Relay node ignores detections below this confidence'),
        # ── DJI serial bridge ────────────────────────────────────────────────
        DeclareLaunchArgument('enable_serial_bridge', default_value='True',
            description='Also launch dji_serial_bridge_node and the '
                        'point_to_cv_target_node adapter that feeds it'),
        DeclareLaunchArgument('enable_cv_target_bridge', default_value='True',
            description='Within the serial bridge launch, also launch the '
                        '/roi_point -> CVTarget adapter (vs. cv_target node only)'),
        DeclareLaunchArgument('serial_device', default_value='/dev/ttyTHS1',
            description='MCB serial device path'),
        DeclareLaunchArgument('serial_baudrate', default_value='115200',
            description='MCB serial baud rate'),
        DeclareLaunchArgument('estimate_velocity', default_value='True',
            description='Finite-difference v_x/v_y/v_z/a_x/a_y/a_z for CVTarget '
                        'from consecutive /roi_point samples'),
    ]

    def create_nodes(context):
        input_w   = LaunchConfiguration('input_image_width').perform(context)
        input_h   = LaunchConfiguration('input_image_height').perform(context)
        network_w = LaunchConfiguration('network_image_width').perform(context)
        network_h = LaunchConfiguration('network_image_height').perform(context)
        image_mean    = LaunchConfiguration('image_mean').perform(context)
        image_stddev  = LaunchConfiguration('image_stddev').perform(context)
        encoding      = LaunchConfiguration('input_encoding').perform(context)

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

        pkg_share = get_package_share_directory('realsense_yolov8_nitros_bridge')

        print(f'[isaac_ros_yolov8_realsense] Color: {input_w}x{input_h} → network: {network_w}x{network_h}')
        print(f'[isaac_ros_yolov8_realsense] Depth center_sample_fraction: {center_sample_frac}')
        print(f'[isaac_ros_yolov8_realsense] Extrinsics topic: {REALSENSE_EXTRINSICS_TOPIC}')

        # ── RealSense composable node ─────────────────────────────────────────
        # Stream profiles are set via YAML so they are available in the parameter
        # server before the node's constructor runs, preventing the Stop/Start
        # cycle that causes the "re-enable stream" warnings and UVC watchdog
        # failures at high FPS.
        realsense_node = ComposableNode(
            package='realsense2_camera',
            plugin='realsense2_camera::RealSenseNodeFactory',
            name='camera',
            namespace='',
            parameters=[
                os.path.join(pkg_share, 'config', 'realsense_640x480x60.yaml'),
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
        # Publishes /detections_output (Detection2DArray) in 640×640 network space
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
        # /detections_output (Detection2DArray, network space)
        #   → /roi (Detection2D, color image space, bbox scaled by color/network ratio)
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
        # depth_ns / color_ns must match the actual published topic roots.
        # With name='camera', namespace='': topics are at /depth/... and /color/...
        roi_depth_node = ComposableNode(
            package='roi_depth_query',
            plugin='roi_depth_query::RoiDepthNode',
            name='roi_depth_node',
            parameters=[{
                'depth_ns':               REALSENSE_DEPTH_NS,         # '/depth'
                'color_ns':               REALSENSE_COLOR_NS,         # '/color'
                'depth_scale':            0.001,   # D435i Z16: raw uint16 → metres
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

        # ── DNN Image Encoder ─────────────────────────────────────────────────
        encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
        yolov8_encoder_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')
            ),
            launch_arguments={
                'image_input_topic':       REALSENSE_COLOR_TOPIC,  # '/color/image_raw'
                'camera_info_input_topic': REALSENSE_INFO_TOPIC,   # '/color/camera_info'
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

        # ── Extrinsics relay ──────────────────────────────────────────────────
        # Standalone node (not composable) — subscribes to the extrinsics topic
        # with VOLATILE QoS (matching the realsense publisher) and pushes the
        # result into roi_depth_node's parameter server, then exits.
        extrinsics_relay = LaunchNode(
            package='roi_depth_query',
            executable='extrinsics_relay_node',
            name='extrinsics_relay',
            parameters=[{
                'extrinsics_topic': REALSENSE_EXTRINSICS_TOPIC,  # '/extrinsics/depth_to_color'
                'target_node':      '/roi_depth_node',
            }],
            output='screen',
        )

        # ── DJI serial bridge ──────────────────────────────────────────────────
        # Brings up dji_serial_bridge_node (talks to the MCB over UART) plus
        # point_to_cv_target_node, which converts roi_depth_node's /roi_point
        # into the CVTarget message the bridge expects. This is the link that
        # actually gets detections to the gimbal/MCB — without it the vision
        # pipeline above only ever produces /roi_point with nothing downstream.
        serial_bridge = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('dji_serial_bridge'),
                    'launch', 'dji_bridge.launch.py')
            ),
            launch_arguments={
                'device':                  LaunchConfiguration('serial_device'),
                'baudrate':                LaunchConfiguration('serial_baudrate'),
                'enable_cv_target_bridge': LaunchConfiguration('enable_cv_target_bridge'),
                'roi_point_topic':         '/roi_point',
                'roi_topic':               '/roi',
                'cv_target_topic':         '/cv_target',
                'estimate_velocity':       LaunchConfiguration('estimate_velocity'),
            }.items(),
            condition=IfCondition(LaunchConfiguration('enable_serial_bridge')),
        )

        return [container, yolov8_encoder_launch, extrinsics_relay, serial_bridge]

    return launch.LaunchDescription(launch_args + [OpaqueFunction(function=create_nodes)])

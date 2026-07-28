# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
isaac_ros_yolov8_realsense.launch.py

Runtime topics have NO /camera/ prefix (e.g. /color/image_raw, /roi,
/cv_target). Full topic layout/inference chain/usage are in README.md —
see there before renaming topics or adding a namespace. Only
engine_file_path is required; pass other args as plain name:=value.
"""

import json
import os
import shutil

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

# ── Snapshot defaults ─────────────────────────────────────────────────────────
# ISAAC_ROS_WS is set in the Isaac ROS Docker environment. Falls back to the
# standard path so the default works both inside and outside the container.
ISAAC_ROS_WS         = os.environ.get('ISAAC_ROS_WS', '/workspaces/isaac_ros-dev')
DEFAULT_SNAPSHOT_DIR = os.path.join(ISAAC_ROS_WS, 'data', 'realsense-captures')


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
        DeclareLaunchArgument('num_classes', default_value='8',
            description='Number of object classes your model was trained on. '
                        'MUST match your model — the decoder default of 80 (COCO) '
                        'will cause a silent out-of-bounds crash if your model has '
                        'a different class count. (ours has 8)'),
        DeclareLaunchArgument('center_sample_fraction', default_value='0.25',
            description='Fraction of bbox center to sample for depth (0.05–1.0)'),
        DeclareLaunchArgument('min_detection_score', default_value='0.0',
            description='Relay node ignores detections below this confidence'),
        DeclareLaunchArgument('ref_sys_topic', default_value='/dji_serial_bridge/ref_sys',
            description='RefSysStatus topic the detection picker reads to learn '
                        'the referee team colour for allied-detection filtering. '
                        'Must match dji_serial_bridge_node\'s ~/ref_sys output '
                        '(node name "dji_serial_bridge", so /dji_serial_bridge/ref_sys).'),
        DeclareLaunchArgument('center_weight', default_value='1.0',
            description='Weight of centrality (1 at image centre, 0 at corners) '
                        'in the picker\'s target score — favours what the robot '
                        'is already aimed at'),
        DeclareLaunchArgument('priority_class_bonus', default_value='0.5',
            description='Score bonus added to a detection whose class is in '
                        'priority_class_ids'),
        DeclareLaunchArgument('priority_class_ids', default_value='[2, 6]',
            description='Class IDs treated as high-value targets (the 3rd target '
                        'in each 0-3 / 4-7 team group)'),
        # ── DJI serial bridge ────────────────────────────────────────────────
        DeclareLaunchArgument('enable_serial_bridge', default_value='True',
            description='Also launch dji_serial_bridge_node and the '
                        'point_to_cv_target_node adapter that feeds it'),
        DeclareLaunchArgument('enable_cv_target_bridge', default_value='True',
            description='Within the serial bridge launch, also launch the '
                        '/roi_point -> CVTarget adapter (vs. cv_target node only)'),
        DeclareLaunchArgument('enable_sentry_pkg', default_value='True',
            description='Also launch the sentry_pkg navigation/SLAM stack '
                        '(auto.launch.py). Set False to run vision only.'),
        DeclareLaunchArgument('enable_visualizer', default_value='False',
            description='Launch detection_picker_visualizer.py: overlays the '
                        'picker\'s scoring factors (conf/centrality/priority/'
                        'team-exclusion/score) on the network-space resize image '
                        'and tags the detection the picker would pick. Publishes '
                        '/yolov8_processed_image. For bench debugging.'),
        DeclareLaunchArgument('lidar_serial_port', default_value='/dev/ttyUSB0',
            description='Serial device path for the SLLIDAR, forwarded to '
                        'sentry_pkg auto.launch.py. Inside the Isaac ROS '
                        'container the hotplug USB lidar is read via the '
                        '/host-dev bind, e.g. /host-dev/ttyUSB0.'),
        DeclareLaunchArgument('debug_log', default_value='True',
            description='Enable for lots more logs' ),
        DeclareLaunchArgument('serial_device', default_value='/dev/ttyTHS1',
            description='MCB serial device path'),
        DeclareLaunchArgument('serial_baudrate', default_value='115200',
            description='MCB serial baud rate'),
        # ── Image snapshot ───────────────────────────────────────────────────
        DeclareLaunchArgument('enable_snapshot', default_value='False',
            description='Capture training images from /color/image_raw to disk'),
        DeclareLaunchArgument('snapshot_output_dir', default_value=DEFAULT_SNAPSHOT_DIR,
            description='Directory to write captured frames'),
        DeclareLaunchArgument('snapshot_interval_ms', default_value='500',
            description='Milliseconds between captures (500 = 2 Hz)'),
        DeclareLaunchArgument('snapshot_format', default_value='jpg',
            description='Image format written to disk: jpg or png'),
        DeclareLaunchArgument('snapshot_disk_limit_pct', default_value='75.0',
            description='Refuse to launch (and stop capturing) above this disk usage %'),
    ]

    def create_nodes(context):

        # ── Pre-launch disk check ─────────────────────────────────────────────
        # This block is the ONLY thing that is conditional on enable_snapshot.
        # It runs in Python before any ROS nodes start, so a full disk produces
        # a clean error message instead of a C++ exception buried in the log.
        if LaunchConfiguration('enable_snapshot').perform(context) == 'True':
            snap_dir   = LaunchConfiguration('snapshot_output_dir').perform(context)
            limit_pct  = float(LaunchConfiguration('snapshot_disk_limit_pct').perform(context))
            os.makedirs(snap_dir, exist_ok=True)
            usage    = shutil.disk_usage(snap_dir)
            used_pct = usage.used / usage.total * 100.0
            if used_pct > limit_pct:
                raise RuntimeError(
                    f'\n\n[ImageSnapshotNode] Disk at \'{snap_dir}\' is '
                    f'{used_pct:.1f}% full (limit: {limit_pct:.0f}%).\n'
                    f'  Used:      {usage.used  / 1e9:.1f} GB\n'
                    f'  Available: {usage.free  / 1e9:.1f} GB\n'
                    f'  Total:     {usage.total / 1e9:.1f} GB\n'
                    'Free up space or set snapshot_output_dir:=<path> '
                    'or snapshot_disk_limit_pct:=<higher_value>.\n'
                )

        # ── Resolve launch arguments ──────────────────────────────────────────
        # Everything below runs unconditionally — snapshot or not.
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
        num_classes          = int(LaunchConfiguration('num_classes').perform(context))
        center_sample_frac   = float(LaunchConfiguration('center_sample_fraction').perform(context))
        min_det_score        = float(LaunchConfiguration('min_detection_score').perform(context))
        ref_sys_topic        = LaunchConfiguration('ref_sys_topic').perform(context)
        center_weight        = float(LaunchConfiguration('center_weight').perform(context))
        priority_class_bonus = float(LaunchConfiguration('priority_class_bonus').perform(context))
        priority_class_ids   = [int(c) for c in
                                json.loads(LaunchConfiguration('priority_class_ids').perform(context))]

        pkg_share = get_package_share_directory('realsense_yolov8_nitros_bridge')

        print(f'[isaac_ros_yolov8_realsense] Color: {input_w}x{input_h} → network: {network_w}x{network_h}')
        print(f'[isaac_ros_yolov8_realsense] Depth center_sample_fraction: {center_sample_frac}')
        print(f'[isaac_ros_yolov8_realsense] Extrinsics topic: {REALSENSE_EXTRINSICS_TOPIC}')
        print(f'[isaac_ros_yolov8_realsense] Team-filter RefSysStatus topic: {ref_sys_topic}')

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
                # CRITICAL: must match the number of classes in your model.
                # The C++ decoder defaults to 80 (COCO) and hardcodes 8400
                # output anchors. With a smaller model the decoder walks off
                # the end of the output tensor → silent segfault, no error log.
                'num_classes':          num_classes,
            }],
        )

        # ── Detection ROI relay node ──────────────────────────────────────────
        # /detections_output (Detection2DArray, network space)
        #   → /roi (Detection2D, color image space, bbox scaled by color/network ratio)
        # Also subscribes to ref_sys_topic (RefSysStatus from dji_serial_bridge_node)
        # to drop allied-team detections: blue team excludes class IDs 0–3, red
        # team excludes 4–7. All detections pass through until the first status
        # message arrives.
        detection_picker_node = ComposableNode(
            package='roi_depth_query',
            plugin='roi_depth_query::DetectionRoiRelayNode',
            name='detection_picker_node',
            parameters=[{
                'detections_topic':     '/detections_output',
                'roi_topic':            '/roi',
                'ref_sys_topic':        ref_sys_topic,
                'network_width':        int(network_w),
                'network_height':       int(network_h),
                'color_width':          int(input_w),
                'color_height':         int(input_h),
                'min_score':            min_det_score,
                'center_weight':        center_weight,
                'priority_class_bonus': priority_class_bonus,
                'priority_class_ids':   priority_class_ids,
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

        # ── Image snapshot node ───────────────────────────────────────────────
        # Defined unconditionally; only appended to the container when
        # enable_snapshot:=True. Remaps its generic 'image' sub to the actual
        # realsense color topic so the same node works with any camera source.
        snapshot_node = ComposableNode(
            package='realsense_yolov8_nitros_bridge',
            plugin='realsense_nitros_bridge::ImageSnapshotNode',
            name='image_snapshot',
            remappings=[('image', REALSENSE_COLOR_TOPIC)],
            parameters=[{
                'output_dir':          LaunchConfiguration('snapshot_output_dir').perform(context),
                'interval_ms':         int(LaunchConfiguration('snapshot_interval_ms').perform(context)),
                'format':              LaunchConfiguration('snapshot_format').perform(context),
                'disk_limit_pct':      float(LaunchConfiguration('snapshot_disk_limit_pct').perform(context)),
                'disk_check_interval': 20,
            }],
            extra_arguments=[{'use_intra_process_comms': True}],
        )

        # ── Build container node list ─────────────────────────────────────────
        # Start with the nodes that always run, then conditionally append
        # the snapshot node. The container is created from this list below.
        node_descriptions = [
            realsense_node,
            tensor_rt_node,
            yolov8_decoder_node,
            detection_picker_node,
            roi_depth_node,
        ]
        if LaunchConfiguration('enable_snapshot').perform(context) == 'True':
            node_descriptions.append(snapshot_node)
            print(f'[isaac_ros_yolov8_realsense] Snapshot enabled → '
                  f'{LaunchConfiguration("snapshot_output_dir").perform(context)} '
                  f'every {LaunchConfiguration("snapshot_interval_ms").perform(context)} ms')

        # ── Shared component container ────────────────────────────────────────
        container = ComposableNodeContainer(
            name='yolov8_realsense_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=node_descriptions,
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
                'debug_log':       LaunchConfiguration('debug_log'),

            }.items(),
            condition=IfCondition(LaunchConfiguration('enable_serial_bridge')),
        )
        # ── Detection picker visualizer ───────────────────────────────────────
        # Regular rclpy node (not composable). Params mirror detection_picker_node
        # so the overlay reflects the live scoring config. Subscribes with
        # best-effort SensorDataQoS, matching the NITROS resize image stream.
        visualizer = LaunchNode(
            package='roi_depth_query',
            executable='detection_picker_visualizer.py',
            name='detection_picker_visualizer',
            parameters=[{
                'detections_topic':     '/detections_output',
                'image_topic':          '/yolov8_encoder/resize/image',
                'ref_sys_topic':        ref_sys_topic,
                'network_width':        int(network_w),
                'network_height':       int(network_h),
                'min_score':            min_det_score,
                'center_weight':        center_weight,
                'priority_class_bonus': priority_class_bonus,
                'priority_class_ids':   priority_class_ids,
            }],
            output='screen',
            condition=IfCondition(LaunchConfiguration('enable_visualizer')),
        )

        sentry_pkg = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('sentry_pkg'),
                    'launch', 'auto.launch.py')
            ),
            launch_arguments={
                'lidar_serial_port': LaunchConfiguration('lidar_serial_port'),
            }.items(),
            condition=IfCondition(LaunchConfiguration('enable_sentry_pkg')),
        )
        return [container, yolov8_encoder_launch, extrinsics_relay,
                visualizer, serial_bridge, sentry_pkg]

    return launch.LaunchDescription(launch_args + [OpaqueFunction(function=create_nodes)])

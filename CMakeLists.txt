cmake_minimum_required(VERSION 3.22)
project(realsense_yolov8_nitros_bridge LANGUAGES CXX CUDA)

# ── C++ standard ──────────────────────────────────────────────────────────────
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── ROS 2 / ament dependencies ────────────────────────────────────────────────
find_package(ament_cmake          REQUIRED)
find_package(rclcpp               REQUIRED)
find_package(rclcpp_components    REQUIRED)
find_package(sensor_msgs          REQUIRED)
find_package(isaac_ros_managed_nitros    REQUIRED)
find_package(isaac_ros_nitros_image_type REQUIRED)

# ── CUDA ─────────────────────────────────────────────────────────────────────
# CUDAToolkit is preferred over the legacy FindCUDA module (removed in CMake 3.27).
find_package(CUDAToolkit REQUIRED)

# ── Shared library (composable node component) ────────────────────────────────
add_library(${PROJECT_NAME}_component SHARED
  src/nitros_realsense_bridge_node.cpp
)

# Headers from our own package (none yet, but keeps the pattern consistent
# if an include/ directory is added later)
target_include_directories(${PROJECT_NAME}_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Link CUDA runtime — CUDA::cudart is the modern CMake target for libcudart
target_link_libraries(${PROJECT_NAME}_component
  CUDA::cudart
)

ament_target_dependencies(${PROJECT_NAME}_component
  rclcpp
  rclcpp_components
  sensor_msgs
  isaac_ros_managed_nitros
  isaac_ros_nitros_image_type
)

# Register the component so `ros2 component load` and ComposableNodeContainer
# can discover it by its fully-qualified plugin name:
#   realsense_nitros_bridge::NitrosRealsenseBridgeNode
rclcpp_components_register_nodes(${PROJECT_NAME}_component
  "realsense_nitros_bridge::NitrosRealsenseBridgeNode"
)

# ── Installation ──────────────────────────────────────────────────────────────
install(TARGETS ${PROJECT_NAME}_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
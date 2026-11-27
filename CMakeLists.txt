cmake_minimum_required(VERSION 3.22)
project(realsense_yolov8_nitros_bridge LANGUAGES CXX)

# ── C++ standard ──────────────────────────────────────────────────────────────
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Dependencies ──────────────────────────────────────────────────────────────
# Only ament_cmake and rclcpp are required everywhere (the decoder core and its
# test build on any ROS machine, including the Mac container). Each component
# library builds only when its Isaac ROS / CUDA dependencies are found.
find_package(ament_cmake REQUIRED)
find_package(rclcpp QUIET)
find_package(rclcpp_components QUIET)
find_package(sensor_msgs QUIET)
find_package(isaac_ros_nitros_image_type QUIET)
find_package(cv_bridge QUIET)
find_package(OpenCV QUIET)
find_package(CUDAToolkit QUIET)
find_package(isaac_ros_tensor_list_interfaces QUIET)
find_package(dji_serial_bridge QUIET)

# ── Pure C++ YOLO26 decoding core (no ROS, no CUDA) ───────────────────────────
add_library(${PROJECT_NAME}_yolo26_core STATIC src/yolo26_decoder.cpp)
target_include_directories(${PROJECT_NAME}_yolo26_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
set_target_properties(${PROJECT_NAME}_yolo26_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(INSTALL_TARGETS ${PROJECT_NAME}_yolo26_core)

# ── Camera bridge + snapshot components (need NITROS, CUDA, cv_bridge) ───────
if(rclcpp_FOUND AND rclcpp_components_FOUND AND sensor_msgs_FOUND AND
  isaac_ros_nitros_image_type_FOUND AND cv_bridge_FOUND AND OpenCV_FOUND AND
  CUDAToolkit_FOUND)
  add_library(${PROJECT_NAME}_component SHARED
    src/nitros_realsense_bridge_node.cpp
    src/image_snapshot_node.cpp
  )
  target_include_directories(${PROJECT_NAME}_component PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(${PROJECT_NAME}_component CUDA::cudart)
  ament_target_dependencies(${PROJECT_NAME}_component
    rclcpp rclcpp_components sensor_msgs isaac_ros_nitros_image_type cv_bridge OpenCV
  )
  rclcpp_components_register_nodes(${PROJECT_NAME}_component
    "realsense_nitros_bridge::NitrosRealsenseBridgeNode"
    "realsense_nitros_bridge::ImageSnapshotNode"
  )
  list(APPEND INSTALL_TARGETS ${PROJECT_NAME}_component)
else()
  message(STATUS "Isaac ROS/CUDA deps not found: skipping camera bridge and snapshot components")
endif()

# ── YOLO26 decoder component (needs the tensor list and armor messages) ──────
if(rclcpp_FOUND AND rclcpp_components_FOUND AND
  isaac_ros_tensor_list_interfaces_FOUND AND dji_serial_bridge_FOUND)
  add_library(${PROJECT_NAME}_yolo26_component SHARED src/yolo26_decoder_node.cpp)
  target_link_libraries(${PROJECT_NAME}_yolo26_component ${PROJECT_NAME}_yolo26_core)
  ament_target_dependencies(${PROJECT_NAME}_yolo26_component
    rclcpp rclcpp_components isaac_ros_tensor_list_interfaces dji_serial_bridge
  )
  rclcpp_components_register_nodes(${PROJECT_NAME}_yolo26_component
    "realsense_nitros_bridge::Yolo26DecoderNode"
  )
  list(APPEND INSTALL_TARGETS ${PROJECT_NAME}_yolo26_component)
else()
  message(STATUS "tensor list/armor message deps not found: skipping Yolo26DecoderNode")
endif()

# ── Installation ──────────────────────────────────────────────────────────────
install(TARGETS ${INSTALL_TARGETS}
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
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_yolo26_decoder test/test_yolo26_decoder.cpp)
  target_link_libraries(test_yolo26_decoder ${PROJECT_NAME}_yolo26_core)
endif()

ament_package()
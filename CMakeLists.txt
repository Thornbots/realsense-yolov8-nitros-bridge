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
# ISAAC_COMPONENTS=OFF builds only the YOLO26 decoding core and its test, for
# hosts without Isaac ROS/CUDA (the Mac container). Robots and CI keep the
# default ON, where every dependency is REQUIRED so nothing is silently skipped.
option(ISAAC_COMPONENTS "Build the Isaac ROS / CUDA components" ON)
find_package(ament_cmake REQUIRED)
if(ISAAC_COMPONENTS)
  find_package(rclcpp REQUIRED)
  find_package(rclcpp_components REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(isaac_ros_nitros_image_type REQUIRED)
  find_package(cv_bridge REQUIRED)
  find_package(OpenCV REQUIRED)
  find_package(CUDAToolkit REQUIRED)
  find_package(isaac_ros_tensor_list_interfaces REQUIRED)
  find_package(dji_serial_bridge REQUIRED)
endif()

# ── Pure C++ YOLO26 decoding core (no ROS, no CUDA) ───────────────────────────
add_library(${PROJECT_NAME}_yolo26_core STATIC src/yolo26_decoder.cpp)
target_include_directories(${PROJECT_NAME}_yolo26_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
set_target_properties(${PROJECT_NAME}_yolo26_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(${PROJECT_NAME}_yolo26_c SHARED src/yolo26_c.cpp)
target_link_libraries(${PROJECT_NAME}_yolo26_c ${PROJECT_NAME}_yolo26_core)
set(INSTALL_TARGETS ${PROJECT_NAME}_yolo26_core ${PROJECT_NAME}_yolo26_c)

# ── Camera bridge + snapshot components (need NITROS, CUDA, cv_bridge) ───────
if(ISAAC_COMPONENTS)
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
endif()

# ── YOLO26 decoder component (needs the tensor list and armor messages) ──────
if(ISAAC_COMPONENTS)
  add_library(${PROJECT_NAME}_yolo26_component SHARED src/yolo26_decoder_node.cpp)
  target_link_libraries(${PROJECT_NAME}_yolo26_component ${PROJECT_NAME}_yolo26_core)
  ament_target_dependencies(${PROJECT_NAME}_yolo26_component
    rclcpp rclcpp_components isaac_ros_tensor_list_interfaces dji_serial_bridge
  )
  rclcpp_components_register_nodes(${PROJECT_NAME}_yolo26_component
    "realsense_nitros_bridge::Yolo26DecoderNode"
  )
  list(APPEND INSTALL_TARGETS ${PROJECT_NAME}_yolo26_component)
endif()

# ── Installation ──────────────────────────────────────────────────────────────
install(TARGETS ${INSTALL_TARGETS}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME test_yolo26_ctypes COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_yolo26_ctypes.py
    $<TARGET_FILE:${PROJECT_NAME}_yolo26_c>)
  ament_add_gtest(test_yolo26_decoder test/test_yolo26_decoder.cpp)
  target_link_libraries(test_yolo26_decoder ${PROJECT_NAME}_yolo26_core ${PROJECT_NAME}_yolo26_c)
endif()

ament_package()

cmake_minimum_required(VERSION 3.5)

project(sensors_ros_pkg LANGUAGES C CXX)

# define standards
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# generate compile_commands.json (for VSCode)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(ament_cmake REQUIRED)
# ROS2
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(OpenCV 4 REQUIRED)

# Publisher
add_library(Publisher
  STATIC
  src/Publisher/publisher.hpp
  src/Publisher/publisher.cpp
)

add_executable(image_publisher src/Publisher/publisher_node.cpp)

target_include_directories(image_publisher PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(image_publisher Publisher ${OpenCV_LIBS})
# convenience function for ROS2 libraries (they can be linked with the above options)
ament_target_dependencies(Publisher rclcpp std_msgs sensor_msgs cv_bridge)

install(
  TARGETS image_publisher #camera_app
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
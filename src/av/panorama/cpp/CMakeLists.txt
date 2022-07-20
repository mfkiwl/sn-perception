cmake_minimum_required(VERSION 3.5)

project(sensors-cpp LANGUAGES C CXX)

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


find_package(PCL REQUIRED)
find_package(OpenCV 4 REQUIRED)

message(STATUS "PCL library DIRS: ${PCL_LIBRARY_DIRS}")
message(STATUS "PCL include DIRS: ${PCL_INCLUDE_DIRS}")

add_executable(main src/Camera/Panorama.cpp)

target_include_directories(main PUBLIC ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

target_link_libraries(main ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_definitions(${PCL_DEFINITIONS})

cmake_minimum_required(VERSION 3.8)
project(slam_toolbox)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# srv/ is populated by the Dockerfile from the official slam_toolbox .deb.
# Glob rather than list: the whole point is to mirror upstream, and a
# hand-kept list would be one more thing to forget.
file(GLOB SRV_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "srv/*.srv")
if(NOT SRV_FILES)
  message(FATAL_ERROR "srv/ is empty - the .deb extraction step did not run")
endif()

rosidl_generate_interfaces(${PROJECT_NAME}
  ${SRV_FILES}
  DEPENDENCIES geometry_msgs std_msgs
)

ament_package()

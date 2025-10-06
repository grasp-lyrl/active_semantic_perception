cmake_minimum_required(VERSION 3.0.2)
project(active_semantic_perception)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
  rosbag
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES habitat_collect
#  CATKIN_DEPENDS rospy roscpp std_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
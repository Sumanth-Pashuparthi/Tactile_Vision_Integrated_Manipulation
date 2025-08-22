cmake_minimum_required(VERSION 3.8)
project(pcl)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(PCL REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_py REQUIRED)
find_package(tf2_eigen REQUIRED)  

include_directories(
  ${PCL_INCLUDE_DIRS}
  ${tf2_ros_INCLUDE_DIRS}
  ${tf2_eigen_INCLUDE_DIRS}
  include
)

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})

add_executable(transform_pointcloud src/transform_pointcloud.cpp)
target_link_libraries(transform_pointcloud ${PCL_LIBRARIES} ${tf2_ros_LIBRARIES} ${tf2_eigen_LIBRARIES})

add_executable(transform_pointcloud2 src/transform_pointcloud2.cpp)
target_link_libraries(transform_pointcloud2 ${PCL_LIBRARIES} ${tf2_ros_LIBRARIES} ${tf2_eigen_LIBRARIES})

add_executable(merge_cloud src/merge_cloud.cpp)
target_link_libraries(merge_cloud ${PCL_LIBRARIES} ${tf2_ros_LIBRARIES} ${tf2_eigen_LIBRARIES})

ament_target_dependencies(
  transform_pointcloud
  rclcpp
  std_msgs
  sensor_msgs
  pcl_conversions
  tf2_ros
  tf2_eigen  
)

ament_target_dependencies(
  transform_pointcloud2
  rclcpp
  std_msgs
  sensor_msgs
  pcl_conversions
  tf2_ros
  tf2_eigen  
)

ament_target_dependencies(
  merge_cloud
  rclcpp
  std_msgs
  sensor_msgs
  pcl_conversions
  tf2_ros
  tf2_eigen  
)


# Install the executables
install(TARGETS
  transform_pointcloud
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  transform_pointcloud2
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  merge_cloud
  DESTINATION lib/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})

# Testing 
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

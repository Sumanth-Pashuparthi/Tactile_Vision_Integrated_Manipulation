import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the package directories
    gazebo_pkg_share = get_package_share_directory('panda_ros2_gazebo')
    
    # Set path to the URDF
    robot_description_path = os.path.join(gazebo_pkg_share, 'urdf', 'panda_macro_test.urdf.xacro')
    
    # Set path to the world file
    world_path = os.path.join(gazebo_pkg_share, 'worlds', 'panda.world')
    
    # Robot state publisher node
    robot_description_content = Command(
        ['xacro ', robot_description_path, ' EE_no:=0']  # Added EE_no parameter
    )
    
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': True
        }]
    )
    
    # Joint state publisher node
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen'
    )

    # Gazebo launch
    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', world_path,
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    # Spawn robot
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_entity',
        output='screen',
        arguments=['-entity', 'panda',
                  '-topic', 'robot_description',
                  '-x', '0',
                  '-y', '0',
                  '-z', '0']
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        joint_state_publisher,
        spawn_robot,
    ])
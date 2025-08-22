
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='force_sensor_array',
            executable='force_extraction',
            name='force_extraction'
        ),
        Node(
            package='force_sensor_array',
            executable='heat_node',
            name='heat_node'
        ),
        Node(
            package='force_sensor_array',
            executable='grasp_state_assessment',
            name='grasp_assessment'
        ),
        Node(
            package='force_sensor_array',
            executable='final_node',
            name='final_node'
        )
    ])

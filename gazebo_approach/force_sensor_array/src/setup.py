# setup.py
from setuptools import setup
import os
from glob import glob

package_name = 'force_sensor_array'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 
                 'control_msgs',
                 'trajectory_msgs',
                 'sensor_msgs',
                 'geometry_msgs',
                 'cv_bridge',
                 'numpy',
                 'opencv-python'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Force sensor array data collector',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'grasp_state_assessment = force_sensor_array.grasp_state_assessment:main',
            'final_node = force_sensor_array.final_node:main',
            'heat_node = force_sensor_array.heat_node:main',
        ],
    },
)
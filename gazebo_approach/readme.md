# RBE 595
# Tactile-Vision Integrated Robotic Systems for Safe Manipulation of Fragile ObjectsCooper 
# Authors:Ducharme, Farhan Seliya, Sumanth Pashuparthi

Download this Zip File and extract the folders
cd tactile_vision
There will be 3 Folder force_sensor_array,FrankaSimulator_Ws,vbm_project
Build and source all the workspaces
In FrankaSimulator_Ws sourced terminal:
ros2 launch panda_ros2_moveit2 panda_interface.launch.py

In vbm_project sourced terminal:
ros2 run pcl transfrom_pointcloud
ros2 run pcl transform_pointcloud2
ros2 run pcl merge_cloud

cd /home/user/tactile/vbm_project/src/pcl/src/
python3 find_grasp_point.py

In force_sensor_array and FrankaSimulator_Ws sourced terminal:
ros2 run force_sensor_array heat_node
ros2 run force_sensor_array grasp_state_assesment
ros2 run force_sensor_array final_node

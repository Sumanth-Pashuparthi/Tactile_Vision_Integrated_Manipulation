#!/usr/bin/env python3

"""
A ROS2 node that implements a smart pick-and-place operation using a Panda robot.
The node controls the robot based on grasp state feedback to pick up a can object 
and place it on the ground.

The sequence involves:
1. Moving to initial position
2. Opening gripper
3. Approaching object
4. Monitoring grasp state
5. Executing pick when sufficient force detected
6. Lifting object
7. Moving to placement position
8. Placing object
9. Retreating
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from ros2_data.action import MoveG, MoveL, Attacher
from std_msgs.msg import String
import time
import asyncio

class SmartPickAndPlace(Node):
    """
    ROS2 node for performing intelligent pick and place operations with a Panda robot.
    
    This node manages multiple action clients and integrates grasp state feedback for:
    - Joint trajectory control
    - Gripper control with force feedback
    - Linear movement
    - Object attachment
    """
    
    def __init__(self):
        super().__init__('smart_pick_and_place')
        
        # Use ReentrantCallbackGroup to allow concurrent action execution
        self.callback_group = ReentrantCallbackGroup()
        
        # Initialize action clients
        self._setup_action_clients()
        
        # Internal state tracking
        self.attacher_goal_handle = None
        self.executor = None
        self.grasp_state = None
        self.awaiting_grasp = False
        
        # Subscribe to grasp state predictions
        self.grasp_sub = self.create_subscription(
            String,
            'grasp_state',
            self.grasp_state_callback,
            10
        )
        
        # Wait for action servers and start sequence
        self.get_logger().info("Waiting for action servers...")
        if self.wait_for_servers():
            self.timer = self.create_timer(10.0, self.start_movement_sequence)

    def _setup_action_clients(self):
        """Initialize all required action clients with appropriate topics."""
        self.joint_traj_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/panda_arm_controller/follow_joint_trajectory',
            callback_group=self.callback_group
        )
        
        self.gripper_client = ActionClient(
            self, 
            MoveG, 
            '/MoveG',
            callback_group=self.callback_group
        )
        
        self.linear_move_client = ActionClient(
            self, 
            MoveL, 
            '/MoveL',
            callback_group=self.callback_group
        )
        
        self.attacher_client = ActionClient(
            self,
            Attacher,
            '/Attacher',
            callback_group=self.callback_group
        )

    def grasp_state_callback(self, msg):
        """
        Process incoming grasp state predictions.
        
        Args:
            msg (String): Grasp state message ("Sufficient" or "Excessive Force")
        """
        self.grasp_state = msg.data
        self.get_logger().info(f"Received grasp state: {self.grasp_state}")
        
        if self.awaiting_grasp and self.grasp_state == "Sufficient":
            self.get_logger().info("Sufficient force detected, executing grasp...")
            self.awaiting_grasp = False
            self.execute_grasp()

    def wait_for_servers(self):
        """
        Wait for all action servers to become available.
        
        Returns:
            bool: True if all servers are available, False otherwise
        """
        servers = [
            (self.joint_traj_client, "Joint trajectory server"),
            (self.gripper_client, "Gripper server"),
            (self.linear_move_client, "Linear move server"),
            (self.attacher_client, "Attacher server")
        ]
        
        for client, name in servers:
            if not client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error(f"{name} not available!")
                return False
        
        self.get_logger().info("All action servers are ready")
        return True

    def start_movement_sequence(self):
        """Start the complete pick and place sequence."""
        self.get_logger().info("Starting pick and place sequence...")
        self.move_arm_to_initial_position()
        self.timer.cancel()

    def move_arm_to_initial_position(self):
        """Move the robot arm to the initial picking position."""
        goal_msg = FollowJointTrajectory.Goal()
        
        traj = JointTrajectory()
        traj.joint_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3',
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]
        
        point = JointTrajectoryPoint()
        point.positions = [-0.5, 0.0, 1.77, -1.55, 0.0, 1.25, 0.0]
        point.time_from_start = Duration(sec=2)
        traj.points = [point]
        goal_msg.trajectory = traj
        
        self.get_logger().info("Moving arm to initial position...")
        self._send_goal_and_wait(self.joint_traj_client, goal_msg, "arm movement")
        self.execute_pick_sequence()

    def execute_pick_sequence(self):
        """Execute the picking sequence: open gripper and approach object."""
        # Open gripper
        self.get_logger().info("Opening gripper...")
        goal_msg = MoveG.Goal()
        goal_msg.goal = 0.04  # Open width in meters
        self._send_goal_and_wait(self.gripper_client, goal_msg, "gripper opening")
        
        # Approach object in stages
        self.get_logger().info("Approaching object...")
        movements = [
            {'y': 0.07, 'z': 0.0},    # Move towards object
            {'y': 0.0, 'z': -0.05},   # Lower to object height
            {'y': 0.027, 'z': 0.0}    # Final approach
        ]
        
        for i, move in enumerate(movements):
            goal_msg = MoveL.Goal()
            goal_msg.movex = 0.0
            goal_msg.movey = move['y']
            goal_msg.movez = move['z']
            goal_msg.speed = 1.0
            self._send_goal_and_wait(self.linear_move_client, goal_msg, f"movement {i+1}")
        
        # Start monitoring grasp state
        self.awaiting_grasp = True
        self.get_logger().info("Waiting for sufficient grasp force...")

    def execute_grasp(self):
        """Execute grasp sequence when sufficient force is detected."""
        self.get_logger().info("Executing grasp sequence...")
        
        try:
            # Close gripper to target width
            goal_msg = MoveG.Goal()
            goal_msg.goal = 0.033  # Target width for grasping
            
            if self._send_goal_and_wait(self.gripper_client, goal_msg, "gripper grasp"):
                # Allow gripper to settle
                time.sleep(2.0)
                
                # Create future for attachment sequence
                future = rclpy.task.Future()
                
                async def run_attach_sequence():
                    try:
                        await self.attach_and_lift_object()
                        future.set_result(None)
                    except Exception as e:
                        future.set_exception(e)
                
                # Schedule the coroutine
                self.executor.create_task(run_attach_sequence())
                
                # Wait for completion
                rclpy.spin_until_future_complete(self, future)
            else:
                self.get_logger().error("Failed to execute gripper grasp")
                
        except Exception as e:
            self.get_logger().error(f"Error in execute_grasp: {str(e)}")
            return

    async def start_attacher(self, object_name, endeffector):
        """
        Start the object attachment action.
        
        Args:
            object_name (str): Name of the object to attach
            endeffector (str): Name of the end effector frame
            
        Returns:
            bool: True if attachment started successfully, False otherwise
        """
        goal_msg = Attacher.Goal()
        goal_msg.object = object_name
        goal_msg.endeffector = endeffector
        
        self.get_logger().info(f'Starting attachment of object: {object_name}')
        send_goal_future = await self.attacher_client.send_goal_async(goal_msg)
        
        if not send_goal_future.accepted:
            self.get_logger().error('Attacher goal rejected')
            return False
        
        self.attacher_goal_handle = send_goal_future
        self.get_logger().info('Attachment started successfully')
        return True

    async def stop_attacher(self):
        """Stop the current object attachment if active."""
        if self.attacher_goal_handle:
            self.get_logger().info('Canceling object attachment')
            await self.attacher_goal_handle.cancel_goal_async()
            self.attacher_goal_handle = None

    async def attach_and_lift_object(self):
        """Execute the complete sequence for attaching, lifting, and placing the object."""
        try:
            # Start object attachment
            if not await self.start_attacher("coke_can", "end_effector_frame"):
                return

            # Lift object
            self.get_logger().info("Lifting object...")
            lift_goal = MoveL.Goal()
            lift_goal.movex = 0.0
            lift_goal.movey = 0.0
            lift_goal.movez = 0.2  # Lift height in meters
            lift_goal.speed = 1.0
            self._send_goal_and_wait(self.linear_move_client, lift_goal, "lifting")

            # Move to placement position
            self.get_logger().info("Moving to front placement position...")
            goal_msg = FollowJointTrajectory.Goal()
            
            traj = JointTrajectory()
            traj.joint_names = [
                'panda_joint1', 'panda_joint2', 'panda_joint3',
                'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
            ]
            
            point = JointTrajectoryPoint()
            point.positions = [0.0, 0.7, 0.0, -1.5, 0.0, 2.2, 0.0]
            point.time_from_start = Duration(sec=2)
            traj.points = [point]
            goal_msg.trajectory = traj
            
            self._send_goal_and_wait(self.joint_traj_client, goal_msg, "moving to front ground position")

            # Lower to ground
            self.get_logger().info("Lowering to ground...")
            ground_goal = MoveL.Goal()
            ground_goal.movex = 0.0
            ground_goal.movey = 0.0
            ground_goal.movez = -0.15
            ground_goal.speed = 0.3  # Slow speed for careful placement
            self._send_goal_and_wait(self.linear_move_client, ground_goal, "lowering to ground")

            # Release object
            await self.stop_attacher()
            
            # Open gripper
            self.get_logger().info("Opening gripper...")
            gripper_goal = MoveG.Goal()
            gripper_goal.goal = 0.04  # Open width
            self._send_goal_and_wait(self.gripper_client, gripper_goal, "gripper opening")
            
            # Retreat after release
            self.get_logger().info("Lifting arm after release...")
            retreat_goal = MoveL.Goal()
            retreat_goal.movex = 0.0
            retreat_goal.movey = 0.0
            retreat_goal.movez = 0.1
            retreat_goal.speed = 1.0
            self._send_goal_and_wait(self.linear_move_client, retreat_goal, "retreating")
                
        except Exception as e:
            self.get_logger().error(f"Error in attach_and_lift_object: {str(e)}")
            await self.stop_attacher()

    def _send_goal_and_wait(self, client, goal_msg, action_name):
        """
        Send an action goal and wait for its completion.
        
        Args:
            client: Action client to use
            goal_msg: Goal message to send
            action_name (str): Name of the action for logging
            
        Returns:
            bool: True if action completed successfully, False otherwise
        """
        self.get_logger().info(f"Sending {action_name} goal...")
        
        try:
            # Send goal and wait for acceptance
            future = client.send_goal_async(goal_msg)
            self.get_logger().info(f"Waiting for {action_name} goal to be accepted...")
            rclpy.spin_until_future_complete(self, future)
            
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error(f"{action_name} goal rejected!")
                return False
            
            # Wait for result
            self.get_logger().info(f"{action_name} goal accepted, waiting for result...")
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            
            # Log completion
            result = result_future.result()
            if hasattr(result, 'status'):
                self.get_logger().info(f"{action_name} completed with status: {result.status}")
            else:
                self.get_logger().info(f"{action_name} completed")
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error in _send_goal_and_wait for {action_name}: {str(e)}")
            return False

def main(args=None):
    """Main entry point for the smart pick and place node."""
    rclpy.init(args=args)
    executor = None
    
    try:
        # Create and run node with multi-threaded executor
        node = SmartPickAndPlace()
        executor = MultiThreadedExecutor()
        node.executor = executor
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if executor:
            executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
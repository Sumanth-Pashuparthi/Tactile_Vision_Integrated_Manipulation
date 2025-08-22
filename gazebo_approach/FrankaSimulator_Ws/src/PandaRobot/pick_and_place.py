#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from ros2_data.action import MoveXYZ, MoveG, MoveL, Attacher
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import asyncio
import threading

class PickAndPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_and_place_node')
        
        # Create callback group for parallel action execution
        callback_group = ReentrantCallbackGroup()
        
        # Initialize action clients
        self.move_xyz_client = ActionClient(
            self, MoveXYZ, '/MoveXYZ', callback_group=callback_group)
        self.move_g_client = ActionClient(
            self, MoveG, '/MoveG', callback_group=callback_group)
        self.move_l_client = ActionClient(
            self, MoveL, '/MoveL', callback_group=callback_group)
        self.attacher_client = ActionClient(
            self, Attacher, '/Attacher', callback_group=callback_group)
        
        # Store attacher goal handle for keeping the action alive
        self.attacher_goal_handle = None

    def wait_for_action_server(self, action_client, timeout=5.0):
        """Wait for action server to be available"""
        if not action_client.wait_for_server(timeout_sec=timeout):
            self.get_logger().error(f'Action server not available after {timeout} seconds')
            return False
        return True

    async def send_move_xyz_goal(self, x, y, z, speed):
        """Send MoveXYZ goal and wait for result"""
        goal_msg = MoveXYZ.Goal()
        goal_msg.positionx = x
        goal_msg.positiony = y
        goal_msg.positionz = z
        goal_msg.speed = speed
        
        self.get_logger().info(f'Moving to position: x={x}, y={y}, z={z}')
        send_goal_future = await self.move_xyz_client.send_goal_async(goal_msg)
        if not send_goal_future.accepted:
            self.get_logger().error('Goal rejected')
            return None
            
        goal_handle = send_goal_future
        result = await goal_handle.get_result_async()
        return result

    async def send_move_g_goal(self, goal):
        """Send gripper movement goal and wait for result"""
        goal_msg = MoveG.Goal()
        goal_msg.goal = goal
        
        self.get_logger().info(f'Setting gripper width to: {goal}')
        send_goal_future = await self.move_g_client.send_goal_async(goal_msg)
        if not send_goal_future.accepted:
            self.get_logger().error('Goal rejected')
            return None
            
        goal_handle = send_goal_future
        result = await goal_handle.get_result_async()
        return result

    async def send_move_l_goal(self, x, y, z, speed):
        """Send linear movement goal and wait for result"""
        goal_msg = MoveL.Goal()
        goal_msg.movex = x
        goal_msg.movey = y
        goal_msg.movez = z
        goal_msg.speed = speed
        
        self.get_logger().info(f'Moving linearly by: x={x}, y={y}, z={z}')
        send_goal_future = await self.move_l_client.send_goal_async(goal_msg)
        if not send_goal_future.accepted:
            self.get_logger().error('Goal rejected')
            return None
            
        goal_handle = send_goal_future
        result = await goal_handle.get_result_async()
        return result

    async def start_attacher(self, object_name, endeffector):
        """Start the attacher action and keep it running"""
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
        """Stop the attacher action if it's running"""
        if self.attacher_goal_handle:
            self.get_logger().info('Canceling object attachment')
            await self.attacher_goal_handle.cancel_goal_async()
            self.attacher_goal_handle = None

    async def execute_pick_and_place(self):
        """Execute the complete pick and place sequence"""
        # Wait for all action servers
        for client in [self.move_xyz_client, self.move_g_client, 
                      self.move_l_client, self.attacher_client]:
            if not self.wait_for_action_server(client):
                return

        try:
            # Move to pre-grasp position
            result = await self.send_move_xyz_goal(0.3, 0.3, 0.5, 1.5)
            if result is None:
                return
            
            # Open gripper
            result = await self.send_move_g_goal(0.04)
            if result is None:
                return
            
            # Approach object
            result = await self.send_move_l_goal(0.13, 0.0, 0.0, 1.5)
            if result is None:
                return
            
            # Close gripper on object
            result = await self.send_move_g_goal(0.02)
            if result is None:
                return
            
            # Start the attacher and keep it running
            if not await self.start_attacher('box', 'end_effector_frame'):
                return
            
            # Wait a moment for attachment to stabilize
            await asyncio.sleep(1.0)
            
            # Lift object
            result = await self.send_move_l_goal(0.0, 0.0, 0.3, 1.5)
            if result is None:
                return
            
            # Move to place position
            result = await self.send_move_l_goal(-0.3, 0.0, 0.0, 1.5)
            if result is None:
                return
            
            # Lower object
            result = await self.send_move_l_goal(0.0, 0.0, -0.2, 1.5)
            if result is None:
                return
            
            # Open gripper to release object
            result = await self.send_move_g_goal(0.04)
            if result is None:
                return
            
            # Stop the attacher
            await self.stop_attacher()
            
            # Move back to a safe position
            result = await self.send_move_l_goal(-0.3, 0.0, 0.0, 1.5)
            if result is None:
                return
            
            self.get_logger().info('Pick and place sequence completed successfully')
            
        except Exception as e:
            self.get_logger().error(f'Failed to complete pick and place sequence: {str(e)}')
            # Make sure to stop the attacher in case of error
            await self.stop_attacher()

def spin_executor(executor):
    """Function to spin the executor in a separate thread"""
    executor.spin()

def main(args=None):
    rclpy.init(args=args)
    
    # Create node and executor
    node = PickAndPlaceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    # Start executor in a separate thread
    executor_thread = threading.Thread(target=spin_executor, args=(executor,))
    executor_thread.start()
    
    # Create event loop and run the pick and place sequence
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(node.execute_pick_and_place())
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        loop.close()
        node.destroy_node()
        rclpy.shutdown()
        executor_thread.join()

if __name__ == '__main__':
    main()
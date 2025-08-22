#!/usr/bin/env python3

"""
ROS2 node for creating a heatmap visualization of tactile sensor data.

This node:
1. Subscribes to force data from a 4x4 grid of tactile sensors
2. Processes the force readings into a normalized array
3. Creates a color-coded heatmap visualization with force values
4. Publishes the visualization as an image message

The visualization uses the XELA color scheme:
- Dark Blue (0.0) -> Cyan (0.25) -> Green (0.5) -> Yellow (0.75) -> Red (1.0)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

class TactileHeatmapNode(Node):
    """
    A ROS2 node that creates and publishes heatmap visualizations of tactile sensor data.
    
    The node processes force readings from a 4x4 grid of tactile sensors and generates
    a color-coded visualization with force values displayed in each cell.
    """
    
    def __init__(self):
        """Initialize the tactile heatmap visualization node."""
        super().__init__('tactile_heatmap_node')
        
        # Initialize storage for force values
        self.force_array = np.zeros((4, 4))
        self.bridge = CvBridge()
        
        # Define fixed scale for consistent color mapping
        self.force_min = -1.0  # Minimum expected force value
        self.force_max = 2.5   # Maximum expected force value
        
        # Set up subscribers for the 4x4 sensor grid
        self._setup_sensor_subscribers()
        
        # Publisher for the heatmap visualization
        self.viz_pub = self.create_publisher(
            Image,
            'tactile_visualization',
            10  # QoS depth
        )
        
        # Timer for publishing visualization at 100Hz
        self.timer = self.create_timer(0.01, self.publish_visualization)
        
        self.get_logger().info('Tactile Heatmap Node initialized')

    def _setup_sensor_subscribers(self):
        """Create subscribers for all 16 tactile sensors in the 4x4 grid."""
        self.subscribers = []
        for i in range(4):
            for j in range(4):
                topic = f'/gazebo/default/panda/sensor_{i}_{j}/wrench'
                sub = self.create_subscription(
                    WrenchStamped,
                    topic,
                    lambda msg, i=i, j=j: self.sensor_callback(msg, i, j),
                    10  # QoS depth
                )
                self.subscribers.append(sub)

    def sensor_callback(self, msg: WrenchStamped, row: int, col: int):
        """
        Update force array with new sensor reading.
        
        Args:
            msg (WrenchStamped): Force/torque message from sensor
            row (int): Row index in the sensor grid (0-3)
            col (int): Column index in the sensor grid (0-3)
        """
        self.force_array[row][col] = msg.wrench.force.y

    def normalize_array(self):
        """
        Normalize the force array to [0,1] range using fixed scale.
        
        Returns:
            numpy.ndarray: Normalized force array with values between 0 and 1
        """
        clipped = np.clip(self.force_array, self.force_min, self.force_max)
        normalized = (clipped - self.force_min) / (self.force_max - self.force_min)
        return normalized

    def get_xela_color(self, value):
        """
        Convert normalized value to XELA color scheme.
        
        The color scheme transitions through 5 points:
        - 0.00: Dark Blue  (0, 0, 139)
        - 0.25: Cyan       (0, 255, 255)
        - 0.50: Green      (0, 255, 0)
        - 0.75: Yellow     (255, 255, 0)
        - 1.00: Red        (255, 0, 0)
        
        Args:
            value (float): Normalized value between 0 and 1
        
        Returns:
            tuple: (Blue, Green, Red) color values
        """
        if value <= 0.25:  # Dark Blue to Cyan
            ratio = value * 4
            return (139 + int(116 * ratio),  # Blue
                   int(255 * ratio),         # Green
                   int(255 * ratio))         # Red
        elif value <= 0.5:  # Cyan to Green
            ratio = (value - 0.25) * 4
            return (255 - int(255 * ratio),  # Blue
                   255,                      # Green
                   255)                      # Red
        elif value <= 0.75:  # Green to Yellow
            ratio = (value - 0.5) * 4
            return (0,                       # Blue
                   255,                      # Green
                   int(255 * ratio))         # Red
        else:  # Yellow to Red
            ratio = (value - 0.75) * 4
            return (0,                       # Blue
                   255 - int(255 * ratio),   # Green
                   255)                      # Red

    def create_heatmap_image(self):
        """
        Create the heatmap visualization image.
        
        Creates a 400x400 pixel image with:
        - Color-coded squares for each sensor
        - Force values displayed in each square
        - White grid lines separating the squares
        
        Returns:
            numpy.ndarray: BGR image array (400x400x3)
        """
        # Normalize and flip array for visualization
        normalized = self.normalize_array()
        normalized = np.flip(normalized, axis=1)
        
        # Create base image (400x400 pixels, BGR format)
        viz_img = np.zeros((400, 400, 3), dtype=np.uint8)
        cell_size = 100  # Size of each sensor cell in pixels
        
        # Draw colored squares with force values
        for i in range(4):
            for j in range(4):
                # Calculate cell coordinates
                x1 = j * cell_size
                y1 = (3-i) * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                # Color the cell based on normalized force
                value = normalized[i][j]
                blue, green, red = self.get_xela_color(value)
                cv2.rectangle(viz_img, (x1, y1), (x2, y2), (blue, green, red), -1)
                
                # Create and position force value text
                text_img = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
                text = f'{self.force_array[i][j]:.2f}'
                
                # Center text in cell
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = (cell_size - text_size[0]) // 2
                text_y = (cell_size + text_size[1]) // 2
                
                # Draw text with outline for better visibility
                cv2.putText(text_img, text, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(text_img, text, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Rotate text 180 degrees for correct orientation
                center = (cell_size // 2, cell_size // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, 180, 1.0)
                text_img = cv2.warpAffine(text_img, rotation_matrix, (cell_size, cell_size))
                
                # Overlay text on colored square
                viz_img[y1:y2, x1:x2] = cv2.addWeighted(
                    viz_img[y1:y2, x1:x2], 1, text_img, 1, 0
                )
        
        # Add white grid lines
        for i in range(1, 4):
            y = i * cell_size
            cv2.line(viz_img, (0, y), (400, y), (255, 255, 255), 1)
            x = i * cell_size
            cv2.line(viz_img, (x, 0), (x, 400), (255, 255, 255), 1)
        
        return viz_img

    def publish_visualization(self):
        """Create and publish the visualization as an Image message."""
        heatmap_img = self.create_heatmap_image()
        
        try:
            img_msg = self.bridge.cv2_to_imgmsg(heatmap_img, encoding="bgr8")
            self.viz_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing visualization: {str(e)}')

def main(args=None):
    """Main entry point for the tactile heatmap node."""
    rclpy.init(args=args)
    node = TactileHeatmapNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
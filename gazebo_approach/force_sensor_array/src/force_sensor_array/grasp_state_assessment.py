#!/usr/bin/env python3

"""
ROS2 node for predicting grasp states using a C3D neural network model.

This node:
1. Subscribes to tactile force sensor heatmap images
2. Buffers sequences of frames
3. Processes frame sequences through a C3D model
4. Predicts grasp states (Sufficient/Excessive Force)
5. Publishes predictions as ROS messages

The model uses a sequence of 10 frames to make predictions about the grasp state,
with frame preprocessing including resizing and normalization.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import torch
from torchvision import transforms
import numpy as np
from PIL import Image as PILImage
from force_sensor_array.models import C3D_tactile

class GraspStatePredictor(Node):
    """
    A ROS2 node that processes tactile force sensor data to predict grasp states.
    
    Uses a C3D neural network to analyze sequences of tactile heatmap images and 
    predict whether the grasp force is sufficient or excessive.
    """
    
    def __init__(self):
        """Initialize the grasp state predictor node."""
        super().__init__('grasp_state_assessment')
        
        # Initialize CV bridge for ROS-OpenCV conversion
        self.bridge = CvBridge()
        
        # Define grasp state classification labels
        self.class_labels = {
            0: "Sufficient",
            2: "Excessive Force"
        }
        
        # Set up image preprocessing transformations
        self.transform_t = transforms.Compose([
            transforms.Resize([4, 4]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Create subscriber for heatmap images
        self.heatmapsubs = self.create_subscription(
            Image,
            'force_sensor_heatmap',
            self.model_callback,
            10  # QoS depth
        )
        
        # Create publisher for grasp state predictions
        self.predict = self.create_publisher(
            String, 
            'grasp_state', 
            10  # QoS depth
        )
        
        # Initialize sequence processing parameters
        self.frame_buffer = []           # Buffer for storing frame sequences
        self.skip_count = 0              # Counter for initial frame skipping
        self.frames_to_skip = 5          # Number of initial frames to skip
        self.sequence_length = 10        # Required sequence length for prediction
        self.processing_in_progress = False
        
        self.get_logger().info("Grasp State Predictor initialized")
        
    def model_callback(self, msg):
        """
        Process incoming heatmap images and manage frame sequences.
        
        Args:
            msg (Image): ROS Image message containing the tactile heatmap
        """
        # Skip if currently processing a sequence
        if self.processing_in_progress:
            self.get_logger().info("Processing in progress. Skipping frame.")
            return
            
        try:
            # Convert ROS Image to OpenCV format
            heat_img = self.bridge.imgmsg_to_cv2(
                msg, 
                desired_encoding='passthrough'
            )
            
            # Skip initial frames to allow sensor stabilization
            if self.skip_count < self.frames_to_skip:
                self.skip_count += 1
                return
                
            # Add frame to buffer if space available
            if len(self.frame_buffer) < self.sequence_length:
                self.frame_buffer.append(heat_img)
            else:
                self.get_logger().warn("Frame dropped: buffer full")
                
            # Process sequence when buffer is full
            if len(self.frame_buffer) == self.sequence_length:
                self.processing_in_progress = True
                message = self.process_frame_sequence()
                self.publish_prediction(message)
                self.frame_buffer = []  # Reset buffer
                self.processing_in_progress = False
                
        except Exception as e:
            self.get_logger().error(f"Error in model callback: {e}")
            
    def publish_prediction(self, message):
        """
        Publish grasp state prediction.
        
        Args:
            message (str): Predicted grasp state label
        """
        output_message = String()
        output_message.data = message
        self.predict.publish(output_message)
        self.get_logger().info("Published grasp state prediction")
        
    def process_frame_sequence(self):
        """
        Process a sequence of frames through the neural network pipeline.
        
        Returns:
            str: Predicted grasp state label
        """
        transformed_frames = []
        
        # Transform each frame in the sequence
        for frame in self.frame_buffer:
            # Convert to PIL image (RGB only)
            pil_img = PILImage.fromarray(frame[:, :, :3])
            
            # Apply preprocessing transformations
            transformed_frame = self.transform_t(pil_img)
            transformed_frames.append(transformed_frame)
            
        # Stack frames into a single tensor [3, sequence_length, height, width]
        sequence = torch.stack(transformed_frames, dim=1)
        
        # Get prediction from model
        grasp_state = self.run_model(sequence)
        return grasp_state
        
    def run_model(self, sequence):
        """
        Run the C3D model on a sequence of frames.
        
        Args:
            sequence (torch.Tensor): Tensor of shape [3, sequence_length, height, width]
            
        Returns:
            str: Predicted grasp state label
        """
        # Initialize model architecture
        model = C3D_tactile(
            length=10,
            fc_dim=32,
            fc_hidden_1=128,
            num_classes=3
        )
        
        # Load pre-trained weights
        model.load_state_dict(torch.load(
            "/home/sumanth/tactile/force_sensor_array/src/force_sensor_array/model_best.pth",
            weights_only=True
        ))
        model.eval()
        
        # Prepare input tensor
        x_tactile = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            output = model(x_tactile)
            y_pred = torch.argmax(output, dim=1).item()
            
        self.get_logger().info(
            f"Predicted class index: {y_pred}, Label: {self.class_labels[y_pred]}"
        )
        
        return self.class_labels[y_pred]

def main(args=None):
    """Main entry point for the grasp state predictor node."""
    rclpy.init(args=args)
    node = GraspStatePredictor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
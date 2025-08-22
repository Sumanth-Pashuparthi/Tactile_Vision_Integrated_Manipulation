#!/usr/bin/env python3

"""
Neural network architectures for processing tactile and visual data.
Includes implementations of:
- Early fusion network for combining tactile and visual information
- 3D CNN architectures for processing spatiotemporal data
- Specialized C3D network for tactile processing

The networks are designed for slip detection and grasp assessment tasks
using multimodal sensory information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding for maintaining spatial dimensions.
    
    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int): Stride of the convolution
    
    Returns:
        nn.Conv2d: 3x3 convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """
    1x1 convolution for channel-wise projection.
    
    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int): Stride of the convolution
    
    Returns:
        nn.Conv2d: 1x1 convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                    padding=0, bias=False)

def deconv(in_planes, out_planes, stride=2):
    """
    Deconvolutional layer for upsampling features.
    
    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int): Stride for upsampling
    
    Returns:
        nn.ConvTranspose2d: Deconvolutional layer
    """
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=stride,
                             padding=1, bias=False)

class EarlyFusion(nn.Module):
    """
    Early fusion network for combining tactile and visual information.
    Uses LSTM for temporal processing of combined features.
    
    Args:
        preTrain (str): Type of pretrained network ('resnet' or 'vgg')
        fc_early_dim (int): Dimension of early fusion layer
        LSTM_layers (int): Number of LSTM layers
        LSTM_units (int): Number of LSTM hidden units
        LSTM_dropout (float): LSTM dropout rate
        num_classes (int): Number of output classes
        dropout_fc (float): Dropout rate for fully connected layers
    """
    def __init__(self, preTrain, fc_early_dim, LSTM_layers, LSTM_units, 
                 LSTM_dropout, num_classes, dropout_fc):
        super(EarlyFusion, self).__init__()
        
        # Set feature dimensions based on pretrained network
        if preTrain == "resnet":
            self.preTrain_dim = 2048
        if preTrain == "vgg":
            self.preTrain_dim = 4096
            
        # Network layers
        self.fc_early = nn.Linear(self.preTrain_dim*2, fc_early_dim)
        self.LSTM = nn.LSTM(
            input_size=fc_early_dim,
            dropout=LSTM_dropout,
            hidden_size=LSTM_units,
            num_layers=LSTM_layers,
            batch_first=True
        )
        self.fc_late = nn.Linear(LSTM_units, num_classes)
        self.dropout_fc = nn.Dropout(dropout_fc)

    def forward(self, x_visual, x_tactile):
        """
        Forward pass concatenating visual and tactile features.
        
        Args:
            x_visual: Visual features
            x_tactile: Tactile features
            
        Returns:
            torch.Tensor: Class predictions
        """
        # Concatenate features
        x = torch.cat((x_visual, x_tactile), -1)
        x = self.fc_early(x)
        
        # LSTM processing
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x, None)
        
        # Final classification
        x = self.dropout_fc(self.fc_late(RNN_out[:,-1,:]))
        return x

def conv3D_output_size(img_size, padding, kernel_size, stride):
    """
    Compute output shape of 3D convolution.
    
    Args:
        img_size (tuple): Input dimensions (t, h, w)
        padding (tuple): Padding sizes
        kernel_size (tuple): Kernel dimensions
        stride (tuple): Stride values
        
    Returns:
        tuple: Output dimensions
    """
    outshape = (
        np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
        np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
        np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int)
    )
    return outshape

class CNN3D(nn.Module):
    """
    3D CNN for processing spatiotemporal data.
    Features two convolutional blocks with batch normalization and dropout.
    
    Args:
        t_dim (int): Temporal dimension
        img_x (int): Image height
        img_y (int): Image width
        drop_p (float): Dropout probability
        fc_hidden1 (int): Hidden layer size
        ch1 (int): First conv layer channels
        ch2 (int): Second conv layer channels
    """
    def __init__(self, t_dim, img_x, img_y, drop_p, fc_hidden1, ch1, ch2):
        super(CNN3D, self).__init__()
        
        # Store dimensions
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        self.fc_hidden1 = int(fc_hidden1)
        self.drop_p = drop_p
        self.ch1, self.ch2 = ch1, ch2
        
        # 3D convolution parameters
        self.k1, self.k2 = (3, 3, 3), (3, 3, 3)  # kernel sizes
        self.s1, self.s2 = (1, 1, 1), (1, 1, 1)  # strides
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  # padding
        
        # Compute output shapes
        self.conv1_outshape = conv3D_output_size(
            (self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1
        )
        self.conv2_outshape = conv3D_output_size(
            self.conv1_outshape, self.pd2, self.k2, self.s2
        )
        
        # Network layers
        self.conv1 = nn.Conv3d(3, self.ch1, kernel_size=self.k1, 
                              stride=self.s1, padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(self.ch1, self.ch2, kernel_size=self.k2,
                              stride=self.s2, padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        
        # Activation and regularization
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)
        
        # Fully connected layer
        self.fc_dim = int(self.ch2 * self.conv2_outshape[0] * 
                         self.conv2_outshape[1] * self.conv2_outshape[2])
        self.fc1 = nn.Linear(self.fc_dim, self.fc_hidden1)

    def forward(self, x_3d):
        """
        Forward pass through the network.
        
        Args:
            x_3d: Input 3D tensor
            
        Returns:
            torch.Tensor: Output features
        """
        # First conv block
        x = self.conv1(x_3d)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        
        # Fully connected layer
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

class CNN3D1(nn.Module):
    """
    Simplified 3D CNN with single convolutional block.
    Used for processing tactile data.
    
    Args:
        t_dim (int): Temporal dimension (default: 10)
        img_x (int): Image height (default: 4)
        img_y (int): Image width (default: 4)
        drop_p (float): Dropout probability (default: 0.2)
        fc_hidden1 (int): Hidden layer size (default: 64)
        ch1 (int): Number of channels (default: 8)
    """
    def __init__(self, t_dim=10, img_x=4, img_y=4, drop_p=0.2, 
                 fc_hidden1=64, ch1=8):
        super(CNN3D1, self).__init__()
        
        # Store dimensions
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        self.fc_hidden1 = int(fc_hidden1)
        self.drop_p = drop_p
        self.ch1 = ch1
        
        # Convolution parameters
        self.k1 = (3, 3, 3)  # kernel size
        self.s1 = (1, 1, 1)  # stride
        self.pd1 = (0, 0, 0)  # padding
        
        # Compute output shape
        self.conv1_outshape = conv3D_output_size(
            (self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1
        )
        
        # Network layers
        self.conv1 = nn.Conv3d(3, self.ch1, kernel_size=self.k1,
                              stride=self.s1, padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        
        # Fully connected layer
        self.fc_dim = int(self.ch1 * self.conv1_outshape[0] * 
                         self.conv1_outshape[1] * self.conv1_outshape[2])
        self.fc1 = nn.Linear(self.fc_dim, self.fc_hidden1)

    def forward(self, x_3d):
        """
        Forward pass through the network.
        
        Args:
            x_3d: Input 3D tensor
            
        Returns:
            torch.Tensor: Output features
        """
        # Conv block
        x = self.conv1(x_3d)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        
        # Fully connected layer
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

class C3D(nn.Module):
    """
    Combined C3D architecture processing both visual and tactile inputs.
    
    Args:
        v_dim (int): Visual temporal dimension (default: 5)
        img_xv (int): Visual image height (default: 256)
        img_yv (int): Visual image width (default: 256)
        drop_p_v (float): Visual dropout rate (default: 0.2)
        fc_hidden_v (int): Visual hidden layer size (default: 256)
        ch1_v (int): Visual first conv channels (default: 32)
        ch2_v (int): Visual second conv channels (default: 48)
        ch1_t (int): Tactile first conv channels (default: 8)
        ch2_t (int): Tactile second conv channels (default: 12)
        t_dim (int): Tactile temporal dimension (default: 10)
        img_xt (int): Tactile image height (default: 4)
        img_yt (int): Tactile image width (default: 4)
        drop_p_t (float): Tactile dropout rate (default: 0.2)
        fc_hidden_t (int): Tactile hidden layer size (default: 64)
        fc_hidden_1 (int): Combined hidden layer size (default: 128)
        num_classes (int): Number of output classes (default: 3)
    """
    def __init__(self, v_dim=5, img_xv=256, img_yv=256, drop_p_v=0.2,
                 fc_hidden_v=256, ch1_v=32, ch2_v=48, ch1_t=8, ch2_t=12,
                 t_dim=10, img_xt=4, img_yt=4, drop_p_t=0.2, fc_hidden_t=64,
                 fc_hidden_1=128, num_classes=3):
        super(C3D, self).__init__()
        
        # Visual and tactile processing branches
        self.visual_c3d = CNN3D(t_dim=v_dim, img_x=img_xv, img_y=img_yv,
                               drop_p=drop_p_v, fc_hidden1=fc_hidden_v,
                               ch1=ch1_v, ch2=ch2_v)
        self.tactile_c3d = CNN3D1(t_dim=t_dim, img_x=img_xt, img_y=img_yt,
                                 drop_p=drop_p_t, fc_hidden1=fc_hidden_t,
                                 ch1=ch1_t)
        
        # Combined processing
        self.fc1 = nn.Linear(fc_hidden_v+fc_hidden_t, fc_hidden_1)
        self.fc2 = nn.Linear(fc_hidden_1, num_classes)
        self.drop_p = drop_p_v

    def forward(self, x_3d_v, x_3d_t):
        """
        Forward pass combining visual and tactile features.
        
        Args:
            x_3d_v: Visual input tensor
            x_3d_t: Tactile input tensor
            
        Returns:
            torch.Tensor: Class predictions
        """
        # Process visual and tactile inputs
        x_v = self.visual_c3d(x_3d_v)
        x_t = self.tactile_c3d(x_3d_t)
        
        # Combine features
        x = torch.cat((x_v, x_t), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        return x

class C3D_tactile(nn.Module):
    """
    C3D architecture specifically designed for tactile data processing.
    Uses a sequence of 3D convolutions with pooling layers.
    
    Args:
        pretrained (bool): Whether to use pretrained weights (default: False)
        length (int): Sequence length (default: 10)
        fc_dim (int): Feature dimension (default: 32)
        fc_hidden_1 (int): Hidden layer size (default: 128)
        num_classes (int): Number of output classes (default: 3)
    """
    def __init__(self, pretrained=False, length=10, fc_dim=32,
                 fc_hidden_1=128, num_classes=3):
        super(C3D_tactile, self).__init__()
        
        # Store network parameters
        self.length = length
        self.fc_dim = fc_dim
        self.fc_hidden = fc_hidden_1
        self.num_classes = num_classes
        
        if length == 10:
            # First conv block
            self.conv1 = nn.Conv3d(3, 8, kernel_size=(3, 3, 3),
                                padding=(1, 1, 1))
            self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2),
                                    stride=(2, 2, 2))
            
            # Second conv block
            self.conv2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3),
                                padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2),
                                    stride=(2, 2, 2))
            
            # Third conv block (temporal)
            self.conv3 = nn.Conv3d(16, 32, kernel_size=(3, 1, 1),
                                padding=(1, 0, 0))
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 1, 1),
                                    stride=(2, 1, 1))
            
            # Classification layers
            self.fc1 = nn.Linear(self.fc_dim, self.fc_hidden)
            self.fc2 = nn.Linear(self.fc_hidden, self.num_classes)
            self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Class predictions
        """
        # First conv block
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Second conv block
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Third conv block
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        
        # Flatten and classify
        x = x.view(-1, self.fc_dim)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
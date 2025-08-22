#!/usr/bin/env python3

"""
Command line argument parser for neural network training configurations.
Handles various aspects of training including:
- Training parameters (epochs, batch size, learning rate)
- GPU configuration
- Dataset paths and processing
- Checkpointing and model saving
- Model architecture selection
"""

import argparse
import os
import torch
import random
import torch.backends.cudnn as cudnn

class Options:
    """
    Configuration manager for neural network training.
    
    Provides a centralized way to manage command-line arguments for:
    - Training hyperparameters
    - Hardware configuration (GPU/CPU)
    - Data loading and processing
    - Model checkpointing
    - Evaluation modes
    """
    
    def __init__(self):
        """Initialize the argument parser."""
        self.parser = argparse.ArgumentParser()
        self.initialized = False
    
    def initialize(self):
        """
        Set up all command-line arguments with their default values.
        
        Arguments are grouped into categories:
        1. Training Parameters
        2. GPU Configuration
        3. Dataset Configuration
        4. Checkpointing Options
        5. Miscellaneous Settings
        """
        # Training Parameters
        self.parser.add_argument(
            '--epochs', 
            default=1000, 
            type=int, 
            metavar='N',
            help='number of total epochs to run'
        )
        self.parser.add_argument(
            '--start-epoch', 
            default=0, 
            type=int, 
            metavar='N',
            help='manual epoch number (useful on restarts)'
        )
        self.parser.add_argument(
            '--batchSize', 
            default=8, 
            type=int, 
            metavar='N',
            help='input batch size'
        )
        self.parser.add_argument(
            '--lr', 
            '--learning-rate', 
            default=1e-7, 
            type=float,
            metavar='LR', 
            help='initial learning rate'
        )
        self.parser.add_argument(
            '--momentum', 
            default=0.9, 
            type=float, 
            metavar='M',
            help='momentum'
        )
        self.parser.add_argument(
            '--weight-decay', 
            '--wd', 
            default=0, 
            type=float,
            metavar='W', 
            help='weight decay (default: 1e-4)'
        )
        self.parser.add_argument(
            '--schedule', 
            type=int, 
            nargs='+', 
            default=20,
            help='Decrease learning rate at these epochs'
        )
        self.parser.add_argument(
            '--gamma', 
            type=float, 
            default=0.9,
            help='LR is multiplied by gamma on schedule'
        )

        # GPU Configuration
        self.parser.add_argument(
            '--gpu_ids', 
            type=str, 
            default='1,4,5,7',
            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU'
        )
        self.parser.add_argument(
            '--manualSeed', 
            type=int,
            help='manual seed for reproducibility'
        )

        # Dataset Configuration
        self.parser.add_argument(
            '--dataroot', 
            type=str, 
            default="./ICIPDataset",
            help='path to images (should have subfolders train/blurred, train/sharp, etc)'
        )
        self.parser.add_argument(
            '--phase', 
            type=str, 
            default='train',
            help='train, val, test, etc'
        )
        self.parser.add_argument(
            '--cropWidth', 
            type=int, 
            default=112,
            help='Crop to this width'
        )
        self.parser.add_argument(
            '--cropHeight', 
            type=int, 
            default=112,
            help='Crop to this height'
        )
        self.parser.add_argument(
            '-j', 
            '--workers', 
            default=4, 
            type=int, 
            metavar='N',
            help='number of data loading workers (default: 4)'
        )

        # Checkpointing Options
        self.parser.add_argument(
            '--checkpoint', 
            type=str, 
            default='/media/farhan/HD-B1/vbrm_tactile_only/Checkpoints',
            metavar='PATH',
            help='Path to save checkpoint'
        )
        self.parser.add_argument(
            '--resume', 
            default='', 
            type=str, 
            metavar='PATH',
            help='path to latest checkpoint (default: none)'
        )
        self.parser.add_argument(
            '--name', 
            type=str, 
            default='experiment_name',
            help='name of the experiment for storing samples and models'
        )

        # Miscellaneous
        self.parser.add_argument(
            '-e', 
            '--evaluate', 
            dest='evaluate', 
            action='store_true',
            help='evaluate model on validation set'
        )
        self.parser.add_argument(
            '--model_arch', 
            type=str, 
            default='C3D_tactile',
            help='The model architecture to use'
        )
        
        self.initialized = True

    def parse(self):
        """
        Parse command-line arguments and set up the configuration.
        
        This method:
        1. Initializes arguments if not already done
        2. Parses command-line arguments
        3. Sets up GPU configuration
        4. Configures random seeds
        5. Saves configuration to file
        
        Returns:
            argparse.Namespace: Parsed argument object
        """
        # Initialize if needed
        if not self.initialized:
            self.initialize()

        # Parse arguments
        self.opt = self.parser.parse_args()
       
        # GPU Configuration
        os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpu_ids
        self.opt.use_cuda = torch.cuda.is_available()
        
        # Process GPU IDs
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
                
        # Random Seed Configuration
        if self.opt.manualSeed is None:
            self.opt.manualSeed = random.randint(1, 10000)
        random.seed(self.opt.manualSeed)
        torch.manual_seed(self.opt.manualSeed)
        
        # CUDA Configuration
        if self.opt.use_cuda:
            torch.cuda.manual_seed_all(self.opt.manualSeed)
            cudnn.benchmark = True
            cudnn.enabled = True

        # Save configuration to file
        args = vars(self.opt)
        expr_dir = os.path.join(self.opt.checkpoint, self.opt.name)
        file_name = os.path.join(expr_dir, 'opt.txt')
        
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
            
        return self.opt
"""
Keras RFCN
Copyright (c) 2018
Licensed under the MIT License (see LICENSE for details)
Written by parap1uie-s@github.com
"""

import math
import numpy as np


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = 1 * 1

        # Input image size
        self.IMAGE_SHAPE = np.array(
            [1280, 1280, 3])

        self.NAME = "BDD"  # Override in sub-classes

        # Backbone model
        self.BACKBONE = "resnet50"

        # NUMBER OF GPUs to use. For CPU training, use 1
        self.GPU_COUNT = 1

        # Number of images to train with on each GPU. A 12GB GPU can typically
        # handle 2 images of 1024x1024px.
        # Adjust based on your GPU memory and image sizes. Use the highest
        # number that your GPU can handle for best performance.
        self.IMAGES_PER_GPU = 1

        # Number of training steps per epoch
        # This doesn't need to match the size of the training set. Tensorboard
        # updates are saved at the end of each epoch, so setting this to a
        # smaller number means getting more frequent TensorBoard updates.
        # Validation stats are also calculated at each epoch end and they
        # might take a while, so don't set this too small to avoid spending
        # a lot of time on validation stats.
        self.STEPS_PER_EPOCH = 1000

        # Number of validation steps to run at the end of every training epoch.
        # A bigger number improves accuracy of validation stats, but slows
        # down the training.
        self.VALIDATION_STEPS = 50

        # The strides of each layer of the FPN Pyramid. These values
        # are based on a Resnet101 backbone.
        # Use same strides on stage 4-6 if use dilated resnet of DetNet
        # Like BACKBONE_STRIDES = [4, 8, 16, 16, 16]
        self.BACKBONE_STRIDES = [4, 8, 16, 32, 64]

        # Number of classification classes (including background)
        self.C = 1 + 11
        self.NUM_CLASSES = self.C  # Override in sub-classes

        # Length of square anchor side in pixels
        self.RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

        # Ratios of anchors at each cell (width/height)
        # A value of 1 represents a square anchor, and 0.5 is a wide anchor
        self.RPN_ANCHOR_RATIOS = [0.5, 1, 2]

        # Anchor stride
        # If 1 then anchors are created for each cell in the backbone feature map.
        # If 2, then anchors are created for every other cell, and so on.
        self.RPN_ANCHOR_STRIDE = 1

        # Non-max suppression threshold to filter RPN proposals.
        # You can reduce this during training to generate more propsals.
        self.RPN_NMS_THRESHOLD = 0.7

        # How many anchors per image to use for RPN training
        self.RPN_TRAIN_ANCHORS_PER_IMAGE = 256

        # ROIs kept after non-maximum supression (training and inference)
        self.POST_NMS_ROIS_TRAINING = 2000
        self.POST_NMS_ROIS_INFERENCE = 1000

        # Input image resing
        # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
        # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
        # be satisfied together the IMAGE_MAX_DIM is enforced.
        self.IMAGE_MIN_DIM = 800
        self.IMAGE_MAX_DIM = 1024
        # If True, pad images with zeros such that they're (max_dim by max_dim)
        self.IMAGE_PADDING = True  # currently, the False option is not supported

        # Image mean (RGB)
        self.MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

        # Number of ROIs per image to feed to classifier/mask heads
        # The Mask RCNN paper uses 512 but often the RPN doesn't generate
        # enough positive proposals to fill this and keep a positive:negative
        # ratio of 1:3. You can increase the number of proposals by adjusting
        # the RPN NMS threshold.
        self.TRAIN_ROIS_PER_IMAGE = 200

        # Percent of positive ROIs used to train classifier/mask heads
        self.ROI_POSITIVE_RATIO = 0.33

        # Pooled ROIs
        self.POOL_SIZE = 3

        # Maximum number of ground truth instances to use in one image
        self.MAX_GT_INSTANCES = 100

        # Bounding box refinement standard deviation for RPN and final detections.
        self.RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
        self.BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

        # Max number of final detections
        self.DETECTION_MAX_INSTANCES = 100

        # Minimum probability value to accept a detected instance
        # ROIs below this threshold are skipped
        self.DETECTION_MIN_CONFIDENCE = 0.8

        # Non-maximum suppression threshold for detection
        self.DETECTION_NMS_THRESHOLD = 0.3

        # Learning rate and momentum
        # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
        # weights to explode. Likely due to differences in optimzer
        # implementation.
        self.LEARNING_RATE = 0.001
        self.LEARNING_MOMENTUM = 0.9

        # Weight decay regularization
        self.WEIGHT_DECAY = 0.0005

        # Use RPN ROIs or externally generated ROIs for training
        # Keep this True for most situations. Set to False if you want to train
        # the head branches on ROI generated by code rather than the ROIs from
        # the RPN. For example, to debug the classifier head without having to
        # train the RPN.
        self.USE_RPN_ROIS = True

        self.K = 3
        # Compute backbone size from input image size
        self.BACKBONE_SHAPES = np.array(
            [[int(math.ceil(self.IMAGE_SHAPE[0] / stride)),
              int(math.ceil(self.IMAGE_SHAPE[1] / stride))]
             for stride in self.BACKBONE_STRIDES])

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

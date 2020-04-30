"""
Keras RFCN
Copyright (c) 2018
Licensed under the MIT License (see LICENSE for details)
Written by parap1uie-s@github.com
"""

'''
This is Main class of RFCN Model
Contain the model's framework and call the backbone
'''




from KerasRFCN.Model.ResNet import ResNet
from KerasRFCN.Model.BaseModel import BaseModel
from KerasRFCN.Utils import log2_graph
from KerasRFCN.Data_generator import data_generator
import KerasRFCN.Utils
import KerasRFCN.Losses
import datetime
import tensorflow as tf
import keras.layers as KL
import keras.engine as KE
import numpy as np
import keras
import os
import re
import keras.backend as K
import keras.models as KM
class RFCN_Model:
    """docstring for RFCN_Model"""

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'testing']

        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        # model yang sudah dibuat
        self.model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        assert mode in ['training', 'testing']

        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")
        # Inputs
        input_image = KL.Input(
            shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        input_image_meta = KL.Input(shape=[None], name="input_image_meta")
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=np.int32)
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=np.float32)

            # Detection GT (class IDs, bounding boxes)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=np.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=np.float32)
            # Normalize coordinates
            h, w = K.shape(input_image)[1], K.shape(input_image)[2]
            image_scale = K.cast(K.stack([h, w, h, w], axis=0), np.float32)
            gt_boxes = KL.Lambda(lambda x: x / image_scale)(input_gt_boxes)

        P2, P3, P4, P5, P6 = ResNet(
            input_image, architecture=config.BACKBONE).output_layers

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        ### RPN ###
        rpn = self.build_rpn_model(config.RPN_ANCHOR_STRIDE,
                                   len(config.RPN_ANCHOR_RATIOS), 256)
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        self.anchors = KerasRFCN.Utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                                config.RPN_ANCHOR_RATIOS,
                                                                config.BACKBONE_SHAPES,
                                                                config.BACKBONE_STRIDES,
                                                                config.RPN_ANCHOR_STRIDE)
        # window size K and total classed num C
        # Example: For coco, C = 80+1
        scoreMapSize = config.K * config.K
        ScoreMaps_classify = []
        for feature_map_count, feature_map in enumerate(mrcnn_feature_maps):
            # [W * H * class_num] * k^2
            ScoreMap = KL.Conv2D(config.C * scoreMapSize, kernel_size=(
                1, 1), name="score_map_class_{}".format(feature_map_count), padding='valid')(feature_map)
            ScoreMaps_classify.append(ScoreMap)

        ScoreMaps_regr = []
        for feature_map_count, feature_map in enumerate(mrcnn_feature_maps):
            # [W * H * 4] * k^2 ==> 4 = (x,y,w,h)
            ScoreMap = KL.Conv2D(4 * scoreMapSize, kernel_size=(
                1, 1), name="score_map_regr_{}".format(feature_map_count), padding='valid')(feature_map)
            ScoreMaps_regr.append(ScoreMap)

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
            else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(proposal_count=proposal_count,
                                 nms_threshold=config.RPN_NMS_THRESHOLD,
                                 name="ROI",
                                 anchors=self.anchors,
                                 config=config)([rpn_class, rpn_bbox])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            _, _, _, active_class_ids = KL.Lambda(
                lambda x: parse_image_meta_graph(x))(input_image_meta)

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_bbox =\
                DetectionTargetLayer(config, name="proposal_targets")([
                    rpn_rois, input_gt_class_ids, gt_boxes])

            # size = [batch, num_rois, class_num]
            classify_vote = VotePooling(config.TRAIN_ROIS_PER_IMAGE, config.C, config.K, config.POOL_SIZE,
                                        config.BATCH_SIZE, config.IMAGE_SHAPE, name="classify_vote")([rois] + ScoreMaps_classify)
            classify_output = KL.TimeDistributed(KL.Activation(
                'softmax'), name="classify_output")(classify_vote)

            # 4 k^2 rather than 4k^2*C
            regr_vote = VotePooling(config.TRAIN_ROIS_PER_IMAGE, 4, config.K, config.POOL_SIZE,
                                    config.BATCH_SIZE, config.IMAGE_SHAPE, name="regr_vote")([rois] + ScoreMaps_regr)
            regr_output = KL.TimeDistributed(KL.Activation(
                'linear'), name="regr_output")(regr_vote)

            rpn_class_loss = KL.Lambda(lambda x: KerasRFCN.Losses.rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: KerasRFCN.Losses.rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(lambda x: KerasRFCN.Losses.mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, classify_vote, active_class_ids])
            bbox_loss = KL.Lambda(lambda x: KerasRFCN.Losses.mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, regr_output])

            inputs = [input_image, input_image_meta,
                      input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes]

            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       classify_vote, classify_output, regr_output,
                       rpn_rois, rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss]

            keras_model = KM.Model(inputs, outputs, name='rfcn_train')
        else:  # inference

            # Network Heads
            # Proposal classifier and BBox regressor heads
            # size = [batch, num_rois, class_num]
            classify_vote = VotePooling(proposal_count, config.C, config.K, config.POOL_SIZE, config.BATCH_SIZE,
                                        config.IMAGE_SHAPE, name="classify_vote")([rpn_rois] + ScoreMaps_classify)
            classify_output = KL.TimeDistributed(KL.Activation(
                'softmax'), name="classify_output")(classify_vote)

            # 4 k^2 rather than 4k^2*C
            regr_vote = VotePooling(proposal_count, 4, config.K, config.POOL_SIZE, config.BATCH_SIZE,
                                    config.IMAGE_SHAPE, name="regr_vote")([rpn_rois] + ScoreMaps_regr)
            regr_output = KL.TimeDistributed(KL.Activation(
                'linear'), name="regr_output")(regr_vote)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, score)] in image coordinates
            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, classify_output, regr_output, input_image_meta])

            keras_model = KM.Model([input_image, input_image_meta],
                                   [detections, classify_output, regr_output,
                                       rpn_rois, rpn_class, rpn_bbox],
                                   name='rfcn_inference')
        return keras_model

    def build_rpn_model(self, anchor_stride, anchors_per_location, depth):
        """Builds a Keras model of the Region Proposal Network.
        It wraps the RPN graph so it can be used multiple times with shared
        weights.

        anchors_per_location: number of anchors per pixel in the feature map
        anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                       every pixel in the feature map), or 2 (every other pixel).
        depth: Depth of the backbone feature map.

        Returns a Keras Model object. The model outputs, when called, are:
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                    applied to anchors.
        """
        input_feature_map = KL.Input(shape=[None, None, depth],
                                     name="input_rpn_feature_map")
        outputs = self.rpn(input_feature_map,
                           anchors_per_location, anchor_stride)
        return KM.Model([input_feature_map], outputs, name="rpn_model")

    def rpn(self, feature_map, anchors_per_location, anchor_stride):
        """Builds a Keras model of the Region Proposal Network.
        It wraps the RPN graph so it can be used multiple times with shared
        weights.

        anchors_per_location: number of anchors per pixel in the feature map
        anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                       every pixel in the feature map), or 2 (every other pixel).
        depth: Depth of the backbone feature map.

        Returns a Keras Model object. The model outputs, when called, are:
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                    applied to anchors.
        """

        shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                           strides=anchor_stride,
                           name='rpn_conv_shared')(feature_map)

        # Anchor Score. [batch, height, width, anchors per location * 2].
        x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                      activation='linear', name='rpn_class_raw')(shared)

        # Reshape to [batch, anchors, 2]
        rpn_class_logits = KL.Lambda(
            lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

        # Softmax on last dimension of BG/FG.
        rpn_probs = KL.Activation(
            "softmax", name="rpn_class_xxx")(rpn_class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location, depth]
        # where depth is [x, y, log(w), log(h)]
        x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                      activation='linear', name='rpn_bbox_pred')(shared)

        # Reshape to [batch, anchors, 4]
        rpn_bbox = KL.Lambda(lambda t: tf.reshape(
            t, [tf.shape(t)[0], -1, 4]))(x)

        return rpn_class_logits, rpn_probs, rpn_bbox

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("Keras-RFCN"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        # Keras 2.2 use saving
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/'\
                                 'releases/download/v0.2/'\
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_path

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum, clipnorm=5.0)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.model._losses = []
        self.model._per_input_losses = {}
        loss_names = ["rpn_class_loss", "rpn_bbox_loss",
                      "mrcnn_class_loss", "mrcnn_bbox_loss"]
        added_losses_name = []
        for name in loss_names:
            layer = self.model.get_layer(name)
            if layer.output.name in added_losses_name:
                continue
            self.model.add_loss(
                tf.reduce_mean(layer.output, keepdims=True))
            added_losses_name.append(layer.output.name)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), "float32")
                      for w in self.model.trainable_weights
                      if 'gamma' not in w.name and 'beta' not in w.name]
        self.model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.model.compile(optimizer=optimizer, loss=[
            None] * len(self.model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.model.metrics_names:
                continue
            layer = self.model.get_layer(name)
            self.model.metrics_names.append(name)
            self.model.metrics.append(tf.reduce_mean(
                layer.output, keepdims=True))

    def set_trainable(self, layer_regex, model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and model is None:
            print("Selecting layers to train")

        model = model or self.model
        print('error edannn dong aaw waajaw')
        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = model.inner_model.layers if hasattr(model, "inner_model")\
            else model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                print("{}{:20}   ({})".format(" " * indent, layer.name,
                                              layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.
        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/Keras-RFCN\_\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) + 1

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "Keras-RFCN_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(score_map\_.*)|(regr\_.*)|(classify\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(res6.*)|(bn6.*)|(mrcnn\_.*)|(rpn\_.*)|(score_map\_.*)|(regr\_.*)|(classify\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(res6.*)|(bn6.*)|(mrcnn\_.*)|(rpn\_.*)|(score_map\_.*)|(regr\_.*)|(classify\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(res6.*)|(bn6.*)|(mrcnn\_.*)|(rpn\_.*)|(score_map\_.*)|(regr\_.*)|(classify\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE,
                                       augment=False)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=1, save_weights_only=True, save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.01, patience=10, verbose=1, mode='auto', min_delta=0.001, min_lr=0)
        ]

        # Train
        print("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        print("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers, verbose=0)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = max(self.config.BATCH_SIZE // 2, 2)

        # sess = tf.compat.v1.Session()
        print(tf.compat.v1.resource_variables_enabled())
        self.model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=next(val_generator),
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=0,
            use_multiprocessing=False,
        )
        self.epoch = max(self.epoch, epochs)

    def detect(self, images, verbose=0):
        """Runs the detection pipeline.
        images: List of images, potentially of different sizes.
        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(
            images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            print("Processing {} images".format(len(images)))

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Run object detection
        detections, mrcnn_class, mrcnn_bbox, \
            rois, rpn_class, rpn_bbox = self.model.predict(
                [molded_images, image_metas], verbose=0)

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores = self.unmold_detections(
                detections[i], image.shape, windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores
            })
        return results

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.
        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding = KerasRFCN.Utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                padding=self.config.IMAGE_PADDING)
            molded_image = KerasRFCN.Utils.mold_image(
                molded_image, self.config)
            # Build image_meta
            image_meta = KerasRFCN.Utils.compose_image_meta(
                0, image.shape, window,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.
        detections: [N, (y1, x1, y2, x2, class_id, score)]
        image_shape: [height, width, depth] Original size of the image before resizing
        window: [y1, x1, y2, x2] Box in the image where the real image is
                excluding the padding.
        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]

        # Compute scale and shift to translate coordinates to image domain.
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale = min(h_scale, w_scale)
        shift = window[:2]  # y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

        # Translate bounding boxes to image domain
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            N = class_ids.shape[0]

        return boxes, class_ids, scores


def trim_zeros_graph(boxes, name=None):
    """Often boxes are represented with matricies of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


class DetectionTargetLayer(KE.Layer):
    """Subsamples proposals and generates target box refinment, class_ids for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinments.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_bbox"]
        outputs = KerasRFCN.Utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes],
            lambda w, x, y: detection_targets_graph(
                w, x, y, self.config),
            self.config.IMAGES_PER_GPU, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, 1),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
        ]

############################################################
#  ROI pooling on Muti Bins
############################################################


class VotePooling(KE.Layer):
    def __init__(self, num_rois, channel_num, k, pool_shape, batch_size, image_shape, **kwargs):
        super(VotePooling, self).__init__(**kwargs)
        self.channel_num = channel_num
        self.k = k
        self.num_rois = num_rois
        self.pool_shape = pool_shape
        self.batch_size = batch_size
        self.image_shape = image_shape

    def call(self, inputs):
        boxes = inputs[0]
        print(boxes)
        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        score_maps = inputs[1:]
        print('error boi')
        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(
            self.image_shape[0] * self.image_shape[1], "float32")
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), "int32")))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indicies for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], "int32")

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                score_maps[i], level_boxes, box_indices, [
                    self.pool_shape * self.k, self.pool_shape * self.k],
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)
        print(pooled)
        # position-sensitive ROI pooling + classify
        score_map_bins = []
        for channel_step in range(self.k*self.k):
            bin_x = K.variable(int(channel_step % self.k) *
                               self.pool_shape, dtype='int32')
            bin_y = K.variable(int(channel_step / self.k) *
                               self.pool_shape, dtype='int32')
            channel_indices = K.variable(list(range(
                channel_step*self.channel_num, (channel_step+1)*self.channel_num)), dtype='int32')
            croped = tf.image.crop_to_bounding_box(
                tf.gather(pooled, indices=channel_indices, axis=-1), bin_y, bin_x, self.pool_shape, self.pool_shape)
            # [pool_shape, pool_shape, channel_num] ==> [1,1,channel_num] ==> [1, channel_num]
            croped_mean = K.pool2d(croped, (self.pool_shape, self.pool_shape), strides=(
                1, 1), padding='valid', data_format="channels_last", pool_mode='avg')
            # [batch * num_rois, 1,1,channel_num] ==> [batch * num_rois, 1, channel_num]
            croped_mean = K.squeeze(croped_mean, axis=1)
            score_map_bins.append(croped_mean)

        # [batch * num_rois, k^2, channel_num]
        score_map_bins = tf.concat(score_map_bins, axis=1)
        # [batch * num_rois, k*k, channel_num] ==> [batch * num_rois,channel_num]
        # because "keepdims=False", the axis 1 will not keep. else will be [batch * num_rois,1,channel_num]
        pooled = K.sum(score_map_bins, axis=1)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, "int32"), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        pooled = tf.expand_dims(pooled, 0)

        print('error boi 3')
        return pooled

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.channel_num

############################################################
#  Detection Layer
############################################################


def clip_to_window(window, boxes):
    """
    window: (y1, x1, y2, x2). The window in the image we want to clip to.
    boxes: [N, (y1, x1, y2, x2)]
    """
    boxes[:, 0] = np.maximum(np.minimum(boxes[:, 0], window[2]), window[0])
    boxes[:, 1] = np.maximum(np.minimum(boxes[:, 1], window[3]), window[1])
    boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], window[2]), window[0])
    boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], window[3]), window[1])
    return boxes


def refine_detections_graph(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)] where
        coordinates are in image domain.
    """

    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type="int32")
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    # deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(
        rois, deltas * config.BBOX_STD_DEV)
    # Convert coordiates to image domain
    # TODO: better to keep them normalized until later
    height, width = config.IMAGE_SHAPE[:2]
    refined_rois *= tf.constant([height, width,
                                 height, width], dtype="float32")
    # Clip boxes to image window
    refined_rois = clip_boxes_graph(refined_rois, window)
    # Round and cast to int since we're deadling with pixels now 5 keatas dibuletin ke atas
    refined_rois = tf.cast(tf.math.rint(refined_rois), dtype="int32")

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(
            class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(conf_keep, 0))
        keep = tf.sparse.to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
            tf.cast(tf.gather(pre_nms_rois, ixs), dtype="float32"),
            tf.gather(pre_nms_scores, ixs),
            max_output_size=config.DETECTION_MAX_INSTANCES,
            iou_threshold=config.DETECTION_NMS_THRESHOLD)
        # Map indicies
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                tf.expand_dims(nms_keep, 0))
    keep = tf.sparse.to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are in image domain.
    detections = tf.concat([
        tf.cast(tf.gather(refined_rois, keep), dtype="float32"),
        tf.cast(tf.gather(class_ids, keep), dtype="float32")[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
    ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]
    active_class_ids = meta[:, 8:]
    return [image_id, image_shape, window, active_class_ids]

# bagian ini bakal di pake pas testing


class DetectionLayer(KE.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are in image domain
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # Run detection refinement graph on each item in the batch
        _, _, window, _ = parse_image_meta_graph(image_meta)
        detections_batch = KerasRFCN.Utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph(
                x, y, w, z, self.config),
            self.config.IMAGES_PER_GPU)

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)
############################################################
#  Proposal Layer
############################################################


def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, 4] each row is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    # Split corners
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    return clipped


class ProposalLayer(KE.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment detals to anchors.

    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, anchors,
                 config=None, **kwargs):
        """
        anchors: [N, (y1, x1, y2, x2)] anchors defined in image coordinates
        """
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.anchors = anchors.astype(np.float32)

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Base anchors
        anchors = self.anchors

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = min(6000, self.anchors.shape[0])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices
        scores = KerasRFCN.Utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                             self.config.IMAGES_PER_GPU)
        deltas = KerasRFCN.Utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                             self.config.IMAGES_PER_GPU)
        anchors = KerasRFCN.Utils.batch_slice(ix, lambda x: tf.gather(anchors, x),
                                              self.config.IMAGES_PER_GPU,
                                              names=["pre_nms_anchors"])

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = KerasRFCN.Utils.batch_slice([anchors, deltas],
                                            lambda x, y: apply_box_deltas_graph(
                                                x, y),
                                            self.config.IMAGES_PER_GPU,
                                            names=["refined_anchors"])

        # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
        height, width = self.config.IMAGE_SHAPE[:2]
        window = np.array([0, 0, height, width]).astype(np.float32)
        boxes = KerasRFCN.Utils.batch_slice(boxes,
                                            lambda x: clip_boxes_graph(
                                                x, window),
                                            self.config.IMAGES_PER_GPU,
                                            names=["refined_anchors_clipped"])

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Normalize dimensions to range of 0 to 1.
        normalized_boxes = boxes / np.array([[height, width, height, width]])

        # Non-max suppression
        def nms(normalized_boxes, scores):
            indices = tf.image.non_max_suppression(
                normalized_boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(normalized_boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count -
                                 tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
        proposals = KerasRFCN.Utils.batch_slice([normalized_boxes, scores], nms,
                                                self.config.IMAGES_PER_GPU)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)

############################################################
#  Detection Target Layer
############################################################


def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas for each.

    Inputs:
    proposals: [N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
            Class-specific bbox refinments.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)

    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Compute overlaps with crowd boxes [anchors, crowds]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine postive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(tf.logical_and(
        roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(
        r * tf.cast(positive_count, "float32") "int32") - positive_count
    negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = KerasRFCN.Utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])

    return rois, roi_gt_class_ids, deltas

import os
import random
import time
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image
from keras import Model, Input
from keras.layers import Input as LayerInput
from keras.optimizers import Adam
from keras.utils import generic_utils
from keras.applications.resnet50 import ResNet50
from mrcnn.visualize import draw_box

from common.adapters.datasets.interfaces import AbstractDataset
from common.adapters.models.interfaces import AbstractModelAdapter
from common.config.frcnn import FRCNNConfig
from common.detection import Detection
from neural_nets.frcnn.keras_frcnn import resnet as nn
from neural_nets.frcnn.keras_frcnn import config, roi_helpers
from neural_nets.frcnn.keras_frcnn import losses as losses_fn
from keras import backend as K

from neural_nets.frcnn.predict_kitti import get_real_coordinates
from neural_nets.frcnn.utils.process import format_img_size


class FRCNNAdapter(AbstractModelAdapter):
    cfg = FRCNNConfig()

    prediction_weights_loaded = False

    def get_name(self) -> str:
        return 'FRCNN'

    def load_latest_checkpoint(self):
        super(FRCNNAdapter, self).load_latest_checkpoint()

    def build_models(self) -> Tuple[Model, Model]:
        """
        Build FRCNN Models (adapter from frcnn)
        :return:
        """

        self.cfg.use_horizontal_flips = False
        self.cfg.use_vertical_flips = False
        self.cfg.rot_90 = False
        self.cfg.im_size = self.env.max_image_side_length
        # self.cfg.base_net_weights = os.path.join('./model/', nn.get_weight_path())
        model_dir, model_path = self.get_checkpoint_location()
        self.cfg.model_path = model_path

        if K.image_dim_ordering() == 'th':
            input_shape_img = (3, None, None)
        else:
            input_shape_img = (None, None, 3)

        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(None, 4))

        # define the base network (resnet here, can be VGG, Inception, etc)
        # define the base network (resnet here, can be VGG, Inception, etc)
        print('Allow resnet training: ', not self.env.use_transfer_learning or self.env.use_transfer_learning and self.env.allow_base_layer_training)
        shared_layers = nn.nn_base(img_input,
                                   trainable=not self.env.use_transfer_learning or self.env.use_transfer_learning and self.env.allow_base_layer_training)
        # resnet = ResNet50(
        #     include_top=False,
        #     input_tensor=img_input,
        #     input_shape=input_shape_img,
        #
        # )
        #
        # for layer in resnet.layers:
        #     layer.trainable = False

        # shared_layers = resnet.outputs[0]

        # print(len(shared_layers))

        num_anchors = len(self.cfg.anchor_box_scales) * len(self.cfg.anchor_box_ratios)
        rpn = nn.rpn(shared_layers, num_anchors)

        # for layer in resnet.layers:
        #     layer.name += '_1'
        #     print(layer.name)

        classifier = nn.classifier(shared_layers, roi_input, self.cfg.num_rois, nb_classes=self.num_classes + 1, trainable=True)

        model_rpn = Model(img_input, rpn[:2])
        model_classifier = Model([img_input, roi_input], classifier)

        # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
        model_all = Model([img_input, roi_input], rpn[:2] + classifier)

        try:

            print('last chckpoint')
            try:
                last_checkpoint = self.find_last()
            except:
                last_checkpoint = None

            if self.env.use_transfer_learning and not last_checkpoint:
                path = '/home/samuelblattner/.keras/models/' + 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
                print('loading weights from {}'.format(path))
                model_rpn.load_weights(path, by_name=True)
                model_classifier.load_weights(path, by_name=True)

            else:
                print('loading weights from {}'.format(last_checkpoint))
                model_rpn.load_weights(last_checkpoint, by_name=True)
                model_classifier.load_weights(last_checkpoint, by_name=True)

        except Exception as e:
            print(e)
            print('Could not load pretrained model weights. Weights can be found in the keras application folder '
                  'https://github.com/fchollet/keras/tree/master/keras/applications')

        optimizer = Adam(lr=self.env.learning_rate, )
        optimizer_classifier = Adam(lr=self.env.learning_rate)

        model_rpn.compile(optimizer=optimizer,
                          loss=[losses_fn.rpn_loss_cls(num_anchors), losses_fn.rpn_loss_regr(num_anchors)])
        model_classifier.compile(optimizer=optimizer_classifier,
                                 loss=[losses_fn.class_loss_cls, losses_fn.class_loss_regr(self.num_classes)],
                                 metrics={'dense_class_{}'.format(self.num_classes + 1): 'accuracy'})
        model_all.compile(optimizer='sgd', loss='mae')

        return (model_rpn, model_classifier, model_all), None

    def train(self, loss_patience=15, val_loss_patience=30):

        # loss_patience = 50
        # val_loss_patience = 100

        import tensorflow as tf

        dir, path = self.get_checkpoint_location()

        train_dataset, val_dataset, __ = self.env.get_datasets()
        val_dataset.cfg = train_dataset.cfg = self.cfg
        self.cfg.verbose = True

        steps_no_loss_improvement = 0
        steps_no_val_loss_improvement = 0

        steps_per_epoch = np.ceil(len(train_dataset) / self.env.batch_size) * 3
        val_steps_per_epoch = np.ceil(len(val_dataset) / self.env.batch_size)

        model_rpn, model_classifier, model_all = self.train_model

        callback = tf.keras.callbacks.TensorBoard(log_dir=dir, histogram_freq=0)
        callback.set_model(model_all)

        losses = np.zeros((int(steps_per_epoch), 5))
        val_losses = np.zeros((int(val_steps_per_epoch), 5))
        rpn_accuracy_rpn_monitor = []
        val_rpn_accuracy_rpn_monitor = []
        rpn_accuracy_for_epoch = []
        val_rpn_accuracy_for_epoch = []
        start_time = time.time()
        iter_num = 0
        val_iter_num = 0
        best_loss = np.Inf
        best_val_loss = np.Inf

        is_validating = False

        class_mapping = {name: label for label, name in enumerate(self.env.class_names)}

        self.get_callbacks()

        for epoch in range(self.env.epochs):

            progbar = generic_utils.Progbar(steps_per_epoch)
            print('Epoch {}/{}'.format(epoch + 1, self.env.epochs))
            curr_loss = 0
            loss_rpn_cls = 0
            loss_rpn_regr = 0
            loss_class_cls = 0
            loss_class_regr = 0

            while True:
                try:

                    if len(rpn_accuracy_rpn_monitor) == steps_per_epoch:
                        mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                        rpn_accuracy_rpn_monitor = []
                        print(
                            'Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                                mean_overlapping_bboxes, steps_per_epoch))
                        if mean_overlapping_bboxes == 0:
                            print('RPN is not producing bounding boxes that overlap'
                                  ' the ground truth boxes. Check RPN settings or keep training.')

                    X, Y, img_data = next(val_dataset) if is_validating else next(train_dataset)

                    # X[:, : , :, 0] -= self.cfg.img_channel_mean[2]  # R
                    # X[:, :, :, 1] -= self.cfg.img_channel_mean[1]  # G
                    # X[:,:, :, 2] -= self.cfg.img_channel_mean[0]  # B
                    #
                    # print(img_data.get('bboxes'))
                    # # ===============================
                    # # ===============================
                    # anchor_sizes = self.cfg.anchor_box_scales
                    # anchor_ratios = self.cfg.anchor_box_ratios
                    # n_anchratios = len(anchor_ratios)
                    # downscale = float(self.cfg.rpn_stride)
                    # draw = X[0].copy()
                    #
                    # draw[:, :, 0] += self.cfg.img_channel_mean[2]
                    # draw[:, :, 1] += self.cfg.img_channel_mean[1]
                    # draw[:, :, 2] += self.cfg.img_channel_mean[0]
                    # for anchor_size_idx in range(len(anchor_sizes)):
                    #     for anchor_ratio_idx in range(n_anchratios):
                    #         anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
                    #         anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]
                    #
                    #         for ix in range(int(X[0].shape[1] / downscale)):
                    #             # x-coordinates of the current anchor box
                    #             x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                    #             x2_anc = downscale * (ix + 0.5) + anchor_x / 2
                    #
                    #             # ignore boxes that go across image boundaries
                    #             if x1_anc < 0 or x2_anc > X[0].shape[1]:
                    #                 continue
                    #
                    #             for jy in range(int(X[0].shape[0] / downscale)):
                    #                 # y-coordinates of the current anchor box
                    #                 y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                    #                 y2_anc = downscale * (jy + 0.5) + anchor_y / 2
                    #
                    #                 # ignore boxes that go across image boundaries
                    #                 if y1_anc < 0 or y2_anc > X[0].shape[0]:
                    #                     continue
                    #
                    #                 # print('YES======================')
                    #                 # print(Y[0][0, jy, ix, anchor_size_idx * anchor_ratio_idx])
                    #                 print(Y[0].shape)
                    #
                    #                 if Y[0][0, jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx]:
                    #                     th = 4
                    #                     if Y[0][0, jy, ix, 9 + anchor_ratio_idx + n_anchratios * anchor_size_idx]:
                    #
                    #                         color = (0, 0, 255)
                    #                     else:
                    #                         color = (255, 0,0)
                    #                     draw_box(draw, [int(y1_anc), int(x1_anc), int(y2_anc), int(x2_anc)], color=color, th=th)
                    #                 else:
                    #                     th = 0
                    #
                    # for box in img_data.get('bboxes'):
                    #     draw_box(draw, [int(box['y1'] * 0.1973), int(box['x1'] * 0.1973), int(box['y2'] * 0.1973), int(box['x2'] * 0.1973)], color=(0, 255, 0), th=2)
                    #     caption = "{} {:.3f}".format(box['class'], 0)
                    #
                    #     print(caption)
                    #     cv2.putText(
                    #         img=draw,
                    #         text=caption,
                    #         org=(int(box['x1']), int(box['y1']) - 10),
                    #         fontFace=cv2.FONT_HERSHEY_PLAIN,
                    #         fontScale=2,
                    #         color=(255, 200, 0),
                    #         thickness=3)
                    #
                    # from matplotlib import pyplot as plt
                    #
                    # # ===============================
                    # # ===============================

                    # print(Y)
                    if is_validating:
                        val_loss_rpn = model_rpn.test_on_batch(X, Y)
                    else:
                        loss_rpn = model_rpn.train_on_batch(X, Y)

                    P_rpn = model_rpn.predict_on_batch(X)


                    # Number of box propasels is set to 50 to reduce computation time
                    result = roi_helpers.rpn_to_roi(
                        rpn_layer=P_rpn[0],
                        regr_layer=P_rpn[1],
                        cfg=self.cfg,
                        dim_ordering=K.image_dim_ordering(),
                        use_regr=True,
                        overlap_thresh=0.7,
                        max_boxes=300)
                    # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                    img = X[0].copy()
                    height, width, ch = img.shape
                    prev_size = self.cfg.im_size
                    ratio = height / width
                    if ratio <= 1:
                        min_size = self.cfg.im_size * ratio
                    else:
                        min_size = self.cfg.im_size / ratio
                    self.cfg.im_size = min_size

                    print(class_mapping)

                    X2, Y1, Y2, IouS = roi_helpers.calc_iou(result, img_data, self.cfg, class_mapping)
                    #====================
                    # for anchor_size_idx in range(len(anchor_sizes)):
                    #     for anchor_ratio_idx in range(n_anchratios):
                    #         anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
                    #         anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]
                    #
                    #         for ix in range(int(X[0].shape[1] / downscale)):
                    #             # x-coordinates of the current anchor box
                    #             x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                    #             x2_anc = downscale * (ix + 0.5) + anchor_x / 2
                    #
                    #             # ignore boxes that go across image boundaries
                    #             if x1_anc < 0 or x2_anc > X[0].shape[1]:
                    #                 continue
                    #
                    #             for jy in range(int(X[0].shape[0] / downscale)):
                    #                 # y-coordinates of the current anchor box
                    #                 y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                    #                 y2_anc = downscale * (jy + 0.5) + anchor_y / 2
                    #
                    #                 # ignore boxes that go across image boundaries
                    #                 if y1_anc < 0 or y2_anc > X[0].shape[0]:
                    #                     continue
                    #
                    #                 # print('YES======================')
                    #                 # print(Y[0][0, jy, ix, anchor_size_idx * anchor_ratio_idx])
                    #
                    #                 print(X2[0, 0])
                    #                 print(X2.shape)
                    #                 print(Y1.shape)
                    #                 print(Y2.shape)
                    #                 if Y2[0, jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx]:
                    #                     th = 1
                    #                     if Y2[0, jy, ix, 9 + anchor_ratio_idx + n_anchratios * anchor_size_idx]:
                    #
                    #                         color = (0, 0, 128)
                    #                     else:
                    #                         color = (128, 0, 0)
                    #                     draw_box(draw, [int(y1_anc), int(x1_anc), int(y2_anc), int(x2_anc)], color=color, th=th)
                    #                 else:
                    #                     th = 0
                    # plt.imshow(draw.astype(np.uint8))
                    # plt.show()
                    # exit(0)
                    # ====================
                    self.cfg.im_size = prev_size
                    skip_classification = False
                    if is_validating:
                        val_iter_num += 1
                    if X2 is None:
                        if is_validating:
                            val_rpn_accuracy_rpn_monitor.append(0)
                            val_rpn_accuracy_for_epoch.append(0)
                            skip_classification = True
                        else:
                            rpn_accuracy_rpn_monitor.append(0)
                            rpn_accuracy_for_epoch.append(0)
                            continue

                    if not skip_classification:
                        # print(Y1)
                        neg_samples = np.where(Y1[0, :, -1] == 1)
                        pos_samples = np.where(Y1[0, :, -1] == 0)
                        if len(neg_samples) > 0:
                            neg_samples = neg_samples[0]
                        else:
                            neg_samples = []
                        if len(pos_samples) > 0:
                            pos_samples = pos_samples[0]
                        else:
                            pos_samples = []
                        if is_validating:
                            val_rpn_accuracy_rpn_monitor.append(len(pos_samples))
                            val_rpn_accuracy_for_epoch.append(len(pos_samples))
                        else:
                            rpn_accuracy_rpn_monitor.append(len(pos_samples))
                            rpn_accuracy_for_epoch.append((len(pos_samples)))

                        if self.cfg.num_rois > 1:
                            if len(pos_samples) < self.cfg.num_rois // 2:
                                selected_pos_samples = pos_samples.tolist()
                            else:
                                selected_pos_samples = np.random.choice(pos_samples, self.cfg.num_rois // 2, replace=False).tolist()
                            try:
                                selected_neg_samples = np.random.choice(neg_samples, self.cfg.num_rois - len(selected_pos_samples),
                                                                        replace=False).tolist()
                            except:
                                selected_neg_samples = np.random.choice(neg_samples, self.cfg.num_rois - len(selected_pos_samples),
                                                                        replace=True).tolist()

                            sel_samples = selected_pos_samples + selected_neg_samples
                        else:
                            # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                            selected_pos_samples = pos_samples.tolist()
                            selected_neg_samples = neg_samples.tolist()
                            if np.random.randint(0, 2):
                                sel_samples = random.choice(neg_samples)
                            else:
                                sel_samples = random.choice(pos_samples)
                        if is_validating:
                            val_loss_class = model_classifier.test_on_batch(
                                [X, X2[:, sel_samples, :]],
                                [Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
                            )

                            val_losses[int(iter_num - steps_per_epoch), 2] = val_loss_class[1]
                            val_losses[int(iter_num - steps_per_epoch), 3] = val_loss_class[2]
                            val_losses[int(iter_num - steps_per_epoch), 4] = val_loss_class[3]
                        else:

                            loss_class = model_classifier.train_on_batch(
                                [X, X2[:, sel_samples, :]],
                                [Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
                            )

                            losses[iter_num, 0] = loss_rpn[1]
                            losses[iter_num, 1] = loss_rpn[2]

                            losses[iter_num, 2] = loss_class[1]
                            losses[iter_num, 3] = loss_class[2]
                            losses[iter_num, 4] = loss_class[3]

                    if is_validating:
                        val_losses[int(iter_num - steps_per_epoch), 0] = val_loss_rpn[1]
                        val_losses[int(iter_num - steps_per_epoch), 1] = val_loss_rpn[2]

                    iter_num += 1

                    if not is_validating:
                        progbar.update(iter_num,
                                       [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                        ('detector_cls', np.mean(losses[:iter_num, 2])),
                                        ('detector_regr', np.mean(losses[:iter_num, 3]))])

                    if iter_num == steps_per_epoch:
                        loss_rpn_cls = np.mean(losses[:, 0])
                        loss_rpn_regr = np.mean(losses[:, 1])
                        loss_class_cls = np.mean(losses[:, 2])
                        loss_class_regr = np.mean(losses[:, 3])
                        class_acc = np.mean(losses[:, 4])

                        mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                        rpn_accuracy_for_epoch = []

                        if self.cfg.verbose:
                            print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                                mean_overlapping_bboxes))
                            print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                            print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                            print('Loss RPN regression: {}'.format(loss_rpn_regr))
                            print('Loss Detector classifier: {}'.format(loss_class_cls))
                            print('Loss Detector regression: {}'.format(loss_class_regr))
                            print('Elapsed time: {}'.format(time.time() - start_time))
                            print('Steps without loss improvement: {}'.format(steps_no_val_loss_improvement))

                        curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                        start_time = time.time()

                        if curr_loss < best_loss:
                            if self.cfg.verbose:
                                print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                            best_loss = curr_loss
                            model_all.save_weights(self.cfg.model_path + '.{}.train'.format(str(epoch).zfill(4)))
                            steps_no_loss_improvement = 0
                        else:
                            steps_no_loss_improvement += 1
                            if steps_no_loss_improvement >= loss_patience:
                                print('EARLY STOPPING LOSS==========')
                                return

                        is_validating = True
                        print('Validating...')

                    if iter_num >= steps_per_epoch + val_steps_per_epoch:
                        print('Finished Validation.')
                        val_loss_rpn_cls = np.mean(val_losses[:, 0])
                        val_loss_rpn_regr = np.mean(val_losses[:, 1])
                        val_loss_class_cls = np.mean(val_losses[:, 2])
                        val_loss_class_regr = np.mean(val_losses[:, 3])
                        class_acc = np.mean(val_losses[:, 4])

                        mean_overlapping_bboxes = float(sum(val_rpn_accuracy_for_epoch)) / len(val_rpn_accuracy_for_epoch)
                        val_rpn_accuracy_for_epoch = []

                        if self.cfg.verbose:
                            print('VALIDATION:')
                            print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                                mean_overlapping_bboxes))
                            print('Validated Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                            print('Validated Loss RPN classifier: {}'.format(val_loss_rpn_cls))
                            print('Validated Loss RPN regression: {}'.format(val_loss_rpn_regr))
                            print('Validated Loss Detector classifier: {}'.format(val_loss_class_cls))
                            print('Validated Loss Detector regression: {}'.format(val_loss_class_regr))
                            print('Validated Elapsed time: {}'.format(time.time() - start_time))
                            print('Validated Steps without loss improvement: {}'.format(steps_no_val_loss_improvement))

                        curr_loss = val_loss_rpn_cls + val_loss_rpn_regr + val_loss_class_cls + val_loss_class_regr
                        iter_num = 0
                        val_iter_num = 0
                        is_validating = False
                        start_time = time.time()

                        if curr_loss < best_val_loss:
                            if self.cfg.verbose:
                                print('Total Validation loss decreased from {} to {}, saving weights'.format(best_val_loss, curr_loss))
                            best_val_loss = curr_loss
                            try:
                                os.mkdir(os.path.dirname(self.cfg.model_path.replace('{epoch:04d}', str(epoch).zfill(4))))
                            except FileExistsError:
                                pass
                            model_all.save_weights(self.cfg.model_path.replace('{epoch:04d}', str(epoch).zfill(4)))
                            steps_no_val_loss_improvement = 0
                        else:
                            if self.cfg.verbose:
                                print('Total Validation loss did not decrease. (current: {}, best: {})'.format(curr_loss, best_val_loss))
                            steps_no_val_loss_improvement += 1
                            if steps_no_val_loss_improvement >= val_loss_patience:
                                print('EARLY STOPPING VAL LOSS ==========')
                                return

                        is_validating = False

                        callback.on_epoch_end(epoch, {
                            'loss': loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr,
                            'rpn_class_loss': loss_rpn_cls,
                            'rpn_bbox_loss': loss_rpn_regr,
                            'classification_loss': loss_class_cls,
                            'regression_loss': loss_class_regr,
                            'val_loss': val_loss_rpn_cls + val_loss_rpn_regr + val_loss_class_cls + val_loss_class_regr,
                            'val_rpn_class_loss': val_loss_rpn_cls,
                            'val_rpn_bbox_loss': val_loss_rpn_regr,
                            'val_classification_loss': val_loss_class_cls,
                            'val_regression_loss': val_loss_class_regr
                        })
                        break

                except Exception as e:
                    print('Exception: {}'.format(e))
                    # save model
                    #model_all.save_weights(self.cfg.model_path)
                    continue

        print('Training complete, exiting.')

    def predict(self, images, min_score=0.5) -> List[Detection]:

        if not self.inference_model:

            class_mapping = {name: label for label, name in enumerate(self.env.class_names)}

            input_shape_img = (None, None, 3)
            input_shape_features = (None, None, 1024)

            img_input = Input(shape=input_shape_img)
            roi_input = Input(shape=(self.cfg.num_rois, 4))
            feature_map_input = Input(shape=input_shape_features)

            shared_layers = nn.nn_base(img_input, trainable=True)

            # define the RPN, built on the base layers
            num_anchors = len(self.cfg.anchor_box_scales) * len(self.cfg.anchor_box_ratios)
            rpn_layers = nn.rpn(shared_layers, num_anchors)
            classifier = nn.classifier(feature_map_input, roi_input, self.cfg.num_rois, nb_classes=len(class_mapping),
                                       trainable=True)
            model_rpn = Model(img_input, rpn_layers)
            model_classifier_only = Model([feature_map_input, roi_input], classifier)

            model_classifier = Model([feature_map_input, roi_input], classifier)

            # for image in images:
            #     st = time.time()
            #
            #     img = image.astype(np.float32)
            #     print(img.shape)
            #     img, ratio = format_img_size(img, self.cfg)
            #     print(ratio)
            #     print(img.shape)
            #     img[:, :, 0] -= self.cfg.img_channel_mean[0]
            #     img[:, :, 1] -= self.cfg.img_channel_mean[1]
            #     img[:, :, 2] -= self.cfg.img_channel_mean[2]
            #     img /= self.cfg.img_scaling_factor
            #     img = np.transpose(img, (2, 0, 1))
            #     img = np.expand_dims(img, axis=0)
            #
            #     X = img
            #     # X, ratio = format_img(img, cfg)
            #     print(img.shape)
            #     Image.fromarray(img[0].astype(np.uint8)).show()

            last_checkpoint = self.find_last()
            print('loading weights from {}'.format(last_checkpoint))
            model_rpn.load_weights(last_checkpoint, by_name=True)
            model_classifier.load_weights(last_checkpoint, by_name=True)

            model_rpn.compile(optimizer='sgd', loss='mse')
            model_classifier.compile(optimizer='sgd', loss='mse')

            self.inference_model = (model_rpn, model_classifier, model_classifier_only)

        detections: List[Detection] = []

        model_rpn, model_classifier, model_classifier_only = self.inference_model

        for image in images:
            st = time.time()

            img = image.astype(np.float32)

            height, width, ch = img.shape
            prev_size = self.cfg.im_size
            ratio = height / width
            if ratio <= 1:
                min_size = self.cfg.im_size * ratio
            else:
                min_size = self.cfg.im_size / ratio

            self.cfg.im_size = min_size

            img, ratio = format_img_size(img, self.cfg)
            self.cfg.im_size = prev_size
            # print(img.shape, ratio)
            img[:, :, 0] -= self.cfg.img_channel_mean[2]
            img[:, :, 1] -= self.cfg.img_channel_mean[1]
            img[:, :, 2] -= self.cfg.img_channel_mean[0]
            img /= self.cfg.img_scaling_factor
            img = np.expand_dims(img, axis=0)

            X = img
            # X = np.transpose(X, (0, 2, 3, 1))
            # X, ratio = format_img(img, cfg)
            # Image.fromarray(img.astype(np.uint8)).show()

            # get the feature maps and output from the RPN
            [Y1, Y2, F] = model_rpn.predict(X)

            # print(F)

            # this is result contains all boxes, which is [x1, y1, x2, y2]
            result = roi_helpers.rpn_to_roi(Y1, Y2, self.cfg, K.image_dim_ordering(), overlap_thresh=0.7)

            # convert from (x1,y1,x2,y2) to (x,y,w,h)
            result[:, 2] -= result[:, 0]
            result[:, 3] -= result[:, 1]
            bbox_threshold = 0.7

            # apply the spatial pyramid pooling to the proposed regions
            boxes = dict()
            for jk in range(result.shape[0] // self.cfg.num_rois + 1):
                rois = np.expand_dims(result[self.cfg.num_rois * jk:self.cfg.num_rois * (jk + 1), :], axis=0)
                if rois.shape[1] == 0:
                    break
                if jk == result.shape[0] // self.cfg.num_rois:
                    # pad R
                    curr_shape = rois.shape
                    target_shape = (curr_shape[0], self.cfg.num_rois, curr_shape[2])
                    rois_padded = np.zeros(target_shape).astype(rois.dtype)
                    rois_padded[:, :curr_shape[1], :] = rois
                    rois_padded[0, curr_shape[1]:, :] = rois[0, 0, :]
                    rois = rois_padded
                [p_cls, p_regr] = model_classifier_only.predict([F, rois])

                for ii in range(p_cls.shape[1]):
                    if np.max(p_cls[0, ii, :]) < bbox_threshold or np.argmax(p_cls[0, ii, :]) == (p_cls.shape[2] - 1):
                        continue

                    cls_num = np.argmax(p_cls[0, ii, :])
                    if cls_num not in boxes.keys():
                        boxes[cls_num] = []
                    (x, y, w, h) = rois[0, ii, :]
                    try:
                        (tx, ty, tw, th) = p_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                        tx /= self.cfg.classifier_regr_std[0]
                        ty /= self.cfg.classifier_regr_std[1]
                        tw /= self.cfg.classifier_regr_std[2]
                        th /= self.cfg.classifier_regr_std[3]
                        x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                    except Exception as e:
                        print(e)
                        pass
                    boxes[cls_num].append(
                        [self.cfg.rpn_stride * x, self.cfg.rpn_stride * y, self.cfg.rpn_stride * (x + w), self.cfg.rpn_stride * (y + h),
                         np.max(p_cls[0, ii, :])])

            # add some nms to reduce many boxes

            for cls_num, box in boxes.items():
                boxes_nms = roi_helpers.non_max_suppression_fast(box, overlap_thresh=0.5)
                boxes[cls_num] = boxes_nms

                for b in boxes_nms:
                    b[0], b[1], b[2], b[3] = get_real_coordinates(ratio, b[0], b[1], b[2], b[3])
                    # print('{} prob: {}'.format(b[0: 4], b[-1]))

            d = []
            for c, b in boxes.items():
                for box in b:
                    if round(box[4], 2) < min_score:
                        continue
                    detection = Detection()
                    detection.bbox = box[0:4]
                    detection.score = round(box[4], 2)
                    detection.class_name = AbstractDataset.SIMPLE_CLASS_NAMES.get(c)

                    d.append(detection)

            detections.append(d)
        return detections

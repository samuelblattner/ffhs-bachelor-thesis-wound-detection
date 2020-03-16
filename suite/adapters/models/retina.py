from logging import Logger
from typing import List, Tuple, Generator

import cv2
import keras
import numpy as np
from PIL import Image
from keras import Model
from suite.adapters.models.interfaces import AbstractModelAdapter
from suite.detection import Detection
from suite.environment import Environment
from suite.utils.images import draw_box
from neural_nets.retina_net.keras_retinanet import models, losses
from neural_nets.retina_net.keras_retinanet.bin.train import create_models
import keras.backend as K

from neural_nets.retina_net.keras_retinanet.utils.image import resize_image


class BaseRetinaAdapter(AbstractModelAdapter):
    NAME: str = 'RetinaNet'
    BACKBONE_NAME: str = 'resnet50'

    def __init__(self, env: Environment, classes: List[str], logger: Logger, checkpoint_root: str = None, checkpoint_path: str = None):
        super(BaseRetinaAdapter, self).__init__(env, classes, logger, checkpoint_root, checkpoint_path)

    def get_name(self) -> str:
        return self.NAME

    def build_models(self, transfer_learning: bool, freeze_backbone: bool, lr: float = 0.001) -> Tuple[Model, Model]:
        backbone = models.backbone(self.BACKBONE_NAME)

        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes=len(self._classes) + 1,
            weights=None if not transfer_learning else backbone.download_imagenet(),
            multi_gpu=1,
            freeze_backbone=freeze_backbone,
            lr=lr,
        )

        # compile model
        training_model.compile(
            loss={
                'regression': losses.smooth_l1(),
                'classification': losses.focal(),
            },
            # metrics=['acc'],
            # metrics=[bbox_iou],
            optimizer=keras.optimizers.adam(lr=lr, clipnorm=0.001)
        )

        return training_model, prediction_model

    def load_latest_checkpoint(self, path: str):
        self.train_model.epoch = 0
        super(BaseRetinaAdapter, self).load_latest_checkpoint(path)

    def prepare_image(self, image) -> np.array:
        image[..., 0] -= 123.68  # R
        image[..., 1] -= 116.779  # G
        image[..., 2] -= 103.939  # B

        return image

    def predict(self, images: list, min_score=0.5, tile_size: int = None) -> List[List[Detection]]:

        images = np.asarray(images).astype(np.float32)
        scaled_images = []
        scale = 1.0
        p = None

        for i, image in enumerate(images):
            image = self.prepare_image(image)
            initial_height, initial_width = image.shape[0], image.shape[1]

            min_side = self.env.min_image_side_length if self.env else None
            max_side = self.env.max_image_side_length if self.env else None

            image, scale = resize_image(image, min_side or 800, max_side or 1333)

            new_width = int((initial_width * scale))
            remove = int((image.shape[1] - new_width) / 2)
            scaled_images.append(image[:, remove:image.shape[1] - remove, :])

        images = np.array(scaled_images)
        boxes, scores, labels = self.inference_model.predict_on_batch(images)

        detections = [[] * images.shape[0]]
        for i in range(images.shape[0]):
            detections[i] = []

            for box, score, label in zip(boxes[i], scores[i], labels[i]):

                # scores are sorted so we can break
                if score < min_score:
                    continue
                det = Detection()
                det.bbox = list(box)

                # if p[0][0] > 0:
                #     masks[:, 1] += p[0][0]
                #     masks[:, 3] += p[0][0]

                # Compensate padding and scale
                det.bbox[0] = int((det.bbox[0]) / scale)
                det.bbox[1] = int((det.bbox[1]) / scale - (p[0][0] if p and p[0][0] > 0 else 0))
                det.bbox[2] = int((det.bbox[2]) / scale)
                det.bbox[3] = int((det.bbox[3]) / scale - (p[0][0] if p and p[0][0] > 0 else 0))
                det.score = score
                det.class_name = self.env.class_names[label] if self.env else self._classes[label]

                detections[i].append(det)
        return detections

    def generate_inference_heatmaps(self, raw_image: np.array, for_box: int) -> Generator:
        """
        Generates a heatmap
        :param raw_image:
        :param plots:
        :return:
        """

        def gen():

            image = raw_image.copy()
            image = self.prepare_image(image)

            layer_names = (
                # 'P3',
                'P4_merged',
                # 'P5',
                # 'P6',
                # 'P7',
                # 'anchors_2',
                # 'C5_reduced',
                # 'res5a',
                # 'res3a_branch2c',
                # 'C3_reduced',
                # 'bn5c_branch2c',
                # 'res5c_branch2c',
                # 'res5c',
                # 'res4f',
                # 'res4b35',
                # 'res5c_relu',
                # 'bn5b_branch2c',
                # 'res3d_branch2c',
            )

            initial_height, initial_width = image.shape[0], image.shape[1]

            # Scale image to target size
            # if not self.env.full_size_eval:

            # image, w, scale, p, c = resize_image(
            #     images[0], max_dim=self.env.max_image_side_length, min_dim=self.env.min_image_side_length
            # )

            min_side = self.env.min_image_side_length if self.env else None
            max_side = self.env.max_image_side_length if self.env else None

            image, scale = resize_image(image, min_side or 800, max_side or 1333)

            try:
                layers = [self.inference_model.get_layer(layer) for layer in layer_names]

            except ValueError:
                return

            for layer in layers:

                boxes, scores, labels, anchors = self.inference_model.output

                if boxes.shape[1] <= for_box:
                    continue

                # Box related gradients
                box_x_grads = K.mean(K.gradients(boxes[0, for_box, 0], layer.output)[0], axis=(0, 1, 2))
                box_y_grads = K.mean(K.gradients(boxes[0, for_box, 1], layer.output)[0], axis=(0, 1, 2))
                box_w_grads = K.mean(K.gradients(boxes[0, for_box, 2], layer.output)[0], axis=(0, 1, 2))
                box_h_grads = K.mean(K.gradients(boxes[0, for_box, 3], layer.output)[0], axis=(0, 1, 2))

                # Score gradients
                score_grads = K.mean(K.gradients(scores[0, for_box], layer.output)[0], axis=(0, 1, 2))
                # label_grads = K.mean(K.gradients(labels[0, for_box], layer.output)[0], axis=(0, 1, 2))

                iterate = K.function(
                    [self.inference_model.input],
                    [boxes[0], box_x_grads, box_y_grads, box_w_grads, box_h_grads, score_grads, scores[0, :], labels, anchors, layer.output[0]]
                )

                boxes, p_box_x_grads, p_box_y_grads, p_box_w_grads, p_box_h_grads, p_score_grads, scores, labels, anchors, base_layer_vals = iterate([np.asarray([image])])

                print(for_box)
                print(labels)
                print(scores[:10])

                for grads in (p_box_x_grads, p_box_y_grads, p_box_w_grads, p_box_h_grads, p_score_grads):

                    layer_vals = base_layer_vals.copy()

                    for i in range(layer_vals.shape[2]):
                        layer_vals[:, :, i] *= grads[i]

                    for anchor in anchors[0][for_box: for_box + 1]:
                        y, x, y2, x2 = np.divide(anchor, scale)

                        draw_box(raw_image, (int(x), int(y), int(x2), int(y2)), (255, 0, 128))

                    heatmap = np.mean(layer_vals, axis=-1)
                    heatmap = np.maximum(heatmap, 0)
                    heatmap /= np.max(heatmap)

                    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET).astype('float32')
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    yield heatmap

        return gen()


class RetinaAdapter(BaseRetinaAdapter):
    NAME: str = 'RetinaNet'
    BACKBONE_NAME = 'resnet50'


class RetinaResnet50Adapter(BaseRetinaAdapter):
    NAME = 'RetinaNet-Resnet50'
    BACKBONE_NAME = 'resnet50'


class RetinaResnet101Adapter(BaseRetinaAdapter):
    NAME = 'RetinaNet-Resnet101'
    BACKBONE_NAME = 'resnet101'


class RetinaResnet152Adapter(BaseRetinaAdapter):
    NAME = 'RetinaNet-Resnet152'
    BACKBONE_NAME = 'resnet152'


class RetinaDensenet121Adapter(BaseRetinaAdapter):
    NAME = 'RetinaNet-Densenet121'
    BACKBONE_NAME = 'densenet121'


class RetinaDensenet169Adapter(BaseRetinaAdapter):
    NAME = 'RetinaNet-Densenet169'
    BACKBONE_NAME = 'densenet169'


class RetinaDensenet201Adapter(BaseRetinaAdapter):
    NAME = 'RetinaNet-Densenet201'
    BACKBONE_NAME = 'densenet201'


class RetinaMobilenet128Adapter(BaseRetinaAdapter):
    NAME = 'RetinaNet-Mobilenet128'
    BACKBONE_NAME = 'mobilenet128'


class RetinaMobilenet160Adapter(BaseRetinaAdapter):
    NAME = 'RetinaNet-Mobilenet160'
    BACKBONE_NAME = 'mobilenet160'


class RetinaMobilenet192Adapter(BaseRetinaAdapter):
    NAME = 'RetinaNet-Mobilenet192'
    BACKBONE_NAME = 'mobilenet192'


class RetinaMobilenet224Adapter(BaseRetinaAdapter):
    NAME = 'RetinaNet-Mobilenet224'
    BACKBONE_NAME = 'mobilenet224'


class RetinaVGG16Adapter(BaseRetinaAdapter):
    NAME = 'RetinaNet-VGG16'
    BACKBONE_NAME = 'vgg16'


class RetinaVGG19Adapter(BaseRetinaAdapter):
    NAME = 'RetinaNet-VGG19'
    BACKBONE_NAME = 'vgg19'

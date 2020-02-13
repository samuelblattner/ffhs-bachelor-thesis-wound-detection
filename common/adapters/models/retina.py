from typing import List, Tuple

import cv2
import keras
import numpy as np
from PIL import Image
from keras import Model
from common.adapters.models.interfaces import AbstractModelAdapter
from common.detection import Detection
from common.environment import Environment
# from common.utils.images import resize_image
from neural_nets.retina_net.keras_retinanet import models, losses
from neural_nets.retina_net.keras_retinanet.bin.train import create_models
import keras.backend as K
#from vis.utils import utils
from keras import activations

#from vis.visualization import visualize_saliency, overlay


# def recall_m(y_true, y_pred):
#     print('RECALL:è=========')
#     print(y_true.shape, y_pred.shape)
#     print(y_true, y_pred)
#     try:
#         true_positives = K.sum(K.round(K.clip(y_true[1] * y_pred[1], 0, 1)))
#         possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#         recall = true_positives / (possible_positives + K.epsilon())
#         return recall
#     except:
#         return 0.0
#
#
# def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision
#
#
# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
from neural_nets.retina_net.keras_retinanet.utils.image import resize_image


class BaseRetinaAdapter(AbstractModelAdapter):
    NAME: str = 'RetinaNet'
    BACKBONE_NAME: str = 'resnet50'

    def __init__(self, env: Environment):
        super(BaseRetinaAdapter, self).__init__(env)
        # if self.env.purpose in (ModelPurposeEnum.PREDICTION, ModelPurposeEnum.EVALUATION):
        #     print('NOW CONVERTING FROM TRAIN MODEL TO INFERENCE FOOL')
        #     self.inference_model = models.convert_model(self.train_model)

    def get_name(self) -> str:
        return self.NAME

    def build_models(self) -> Tuple[Model, Model]:
        backbone = models.backbone(self.BACKBONE_NAME)

        print('Using Transfer Learning: ', self.env.use_transfer_learning)
        print('Freezing backbone: ', self.env.use_transfer_learning and not self.env.allow_base_layer_training)

        train_dataset, _, __ = self.env.get_datasets()
        _.center_color_to_imagenet = __.center_color_to_imagenet = train_dataset.center_color_to_imagenet = True

        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes=self.num_classes + 1,
            weights=None if not self.env.use_transfer_learning else backbone.download_imagenet(),
            multi_gpu=1,
            freeze_backbone=self.env.use_transfer_learning and not self.env.allow_base_layer_training,
            lr=self.env.learning_rate,
        )

        # compile model
        training_model.compile(
            loss={
                'regression': losses.smooth_l1(),
                'classification': losses.focal(),
            },
            # metrics=['acc'],
            # metrics=[bbox_iou],
            optimizer=keras.optimizers.adam(lr=self.env.learning_rate, clipnorm=0.001)
        )

        return training_model, prediction_model

    def load_latest_checkpoint(self):
        self.train_model.epoch = 0
        super(BaseRetinaAdapter, self).load_latest_checkpoint()

    def predict(self, images: list, min_score=0.5) -> List[List[Detection]]:

        images = np.asarray(images).astype(np.float32)
        im = images[0].copy()
        scaled_images = []
        scale = 1.0
        p = None
        for i, image in enumerate(images):
            image[..., 0] -= 123.68  # R
            image[..., 1] -= 116.779  # G
            image[..., 2] -= 103.939  # B

            initial_height, initial_width = image.shape[0], image.shape[1]

            # Scale image to target size
            if not self.env.full_size_eval:

                # image, w, scale, p, c = resize_image(
                #     images[0], max_dim=self.env.max_image_side_length, min_dim=self.env.min_image_side_length
                # )

                image, scale = resize_image(image, self.env.min_image_side_length or 800, self.env.max_image_side_length or 1333)

                # from matplotlib import pyplot as plt
                # plt.imshow(image.astype('uint8'))
                # plt.show()
                # exit(0)


                new_width = int((initial_width * scale))
                remove = int((image.shape[1] - new_width) / 2)
                scaled_images.append(image[:, remove:image.shape[1] - remove, :])
            else:
                scaled_images.append(image)

        images = np.array(scaled_images)
        # image = np.expand_dims(image, 0)
        boxes, scores, labels = self.inference_model.predict_on_batch(images)

        detections = [[] * images.shape[0]]
        for i in range(images.shape[0]):
            detections[i] = []

            for box, score, label in zip(boxes[i], scores[i], labels[i]):

                # scores are sorted so we can break
                if score < min_score:
                    break
                det = Detection()
                det.bbox = box

                # if p[0][0] > 0:
                #     masks[:, 1] += p[0][0]
                #     masks[:, 3] += p[0][0]

                # Compensate padding and scale
                det.bbox[0] = (det.bbox[0]) / scale
                det.bbox[1] = (det.bbox[1]) / scale - (p[0][0] if p and p[0][0] > 0 else 0)
                det.bbox[2] = (det.bbox[2]) / scale
                det.bbox[3] = (det.bbox[3]) / scale - (p[0][0] if p and p[0][0] > 0 else 0)
                det.score = score
                det.class_name = self.env.class_names[label]

                detections[i].append(det)
        return detections

    def generate_inference_heatmaps(self, raw_image: np.array, plots) -> np.array:

        image = raw_image.copy()
        image[..., 0] -= 123.68  # R
        image[..., 1] -= 116.779  # G
        image[..., 2] -= 103.939  # B

        box_output = self.inference_model.output[0][:, 0]

        layer_names = (
            #'fc1000',
            'bn5c_branch2c',
            # 'bn4b35_branchs2c',
            # 'bn4b22_branch2c',
            # 'bn3b7_branch2c',
            # 'bn2c_branch2c',
        )
        model = self.inference_model
        #layer_idx = utils.find_layer_idx(model, 'bn5c_branch2c')
        #model.layers[layer_idx].activation = activations.linear
        #model = utils.apply_modifications(model)

        #from matplotlib import pyplot as plt
        #f, ax = plt.subplots(1, 2)

        # 20 is the imagenet index corresponding to `ouzel`
        #grads = visualize_saliency(model, layer_idx, filter_indices=20, seed_input=raw_image)

        # visualize grads as heatmap
        #ax[0].imshow(grads, cmap='jet')

        #https://github.com/raghakot/keras-vis/blob/master/examples/resnet/attention.ipynb

        try:
            layers = [self.inference_model.get_layer(layer) for layer in layer_names]
        except ValueError:
            return

        for layer, plot in zip(layers, plots):

            grads = K.mean(K.gradients(box_output, layer.output)[0], axis=(0, 1, 2))
            iterate = K.function(
                [self.inference_model.input],
                [grads, layer.output[0]]
            )
            pooled_grad_values, layer_values = iterate([np.asarray([image])])
            for i in range(layer_values.shape[2]):
                layer_values[:, :, i] += pooled_grad_values[i]

            heatmap = np.mean(layer_values, axis=-1)
            # heatmap = np.abs(heatmap)
            # heatmap = np.maximum(heatmap, 0)
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)) * (-1 if np.min(heatmap) < 0 else 1)

            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT).astype('float32')
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            blend = cv2.addWeighted(heatmap, 0.6, raw_image, 0.4, 0)
            # blend = np.clip((heatmap * 0.001 + im).astype('uint8'), 0, 255)
            plot.imshow(blend.astype('uint8'))


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

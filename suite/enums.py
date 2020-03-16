from enum import Enum


class SuiteActionEnum(Enum):
    """
    Distinction required to generate the model accordingly
    """
    TRAINING = 'Training'
    PREDICTION = 'Prediction'
    EVALUATION = 'Evaluation'
    ANONYMIZE = 'Anonymize'


class NeuralNetEnum(Enum):
    """
    Enumeration of available nets to train
    """
    MRCNN_RESNET50 = 'Mask R-CNN, Resnet-50 Backbone'
    MRCNN_RESNET101 = 'Mask R-CNN, Resnet-101 Backbone'
    UNET = 'U-NET'

    YOLO3_DARKNET = 'Yolo3, Darknet Backbone'
    YOLO_3_DARKNET = 'Yolo_3, Darknet Backbone'

    FRCNN_RESNET50 = 'Faster R-CNN, Resnet-50 Backbone'

    RETINA_RESNET50 = 'Retina Net, Resnet-50 Backbone'
    RETINA_RESNET101 = 'Retina Net, Resnet-101 Backbone'
    RETINA_RESNET152 = 'Retina Net, Resnet-152 Backbone'
    RETINA_DENSENET121 = 'Retina Net, Densenet-121 Backbone'
    RETINA_DENSENET169 = 'Retina Net, Densenet-169 Backbone'
    RETINA_DENSENET201 = 'Retina Net, Densenet-201 Backbone'
    RETINA_MOBILENET128 = 'Retina Net, Mobilenet-128 Backbone'
    RETINA_MOBILENET160 = 'Retina Net, Mobilenet-160 Backbone'
    RETINA_MOBILENET192 = 'Retina Net, Mobilenet-192 Backbone'
    RETINA_MOBILENET224 = 'Retina Net, Mobilenet-224 Backbone'
    RETINA_VGG16 = 'Retina Net, VGG-16 Backbone'
    RETINA_VGG19 = 'Retina Net, VGG-19 Backbone'

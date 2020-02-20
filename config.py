from os.path import join

# from suite.adapters.datasets.frcnn import FRCNNDataset
from suite.adapters.datasets.retina import RetinaDataset
from suite.adapters.datasets.yolo3 import Yolo_3Dataset
# from suite.adapters.models.frcnn import FRCNNAdapter
from suite.enums import NeuralNetEnum
from suite.adapters.models.retina import RetinaResnet50Adapter, RetinaResnet101Adapter, RetinaResnet152Adapter, RetinaDensenet121Adapter, \
    RetinaDensenet169Adapter, RetinaDensenet201Adapter, RetinaMobilenet128Adapter, RetinaMobilenet160Adapter, RetinaMobilenet192Adapter, \
    RetinaMobilenet224Adapter, RetinaVGG16Adapter, RetinaVGG19Adapter
from suite.adapters.models.yolo3 import Yolo3Adapter

ENVIRONMENT_ROOT = join('environments')

NET_MAP = {
    'yolo3-darknet': NeuralNetEnum.YOLO3_DARKNET,
    'yolo_3-darknet': NeuralNetEnum.YOLO_3_DARKNET,

    'frcnn-resnet50': NeuralNetEnum.FRCNN_RESNET50,

    'retina-resnet50': NeuralNetEnum.RETINA_RESNET50,
    'retina-resnet101': NeuralNetEnum.RETINA_RESNET101,
    'retina-resnet152': NeuralNetEnum.RETINA_RESNET152,
    'retina-densenet121': NeuralNetEnum.RETINA_DENSENET121,
    'retina-densenet169': NeuralNetEnum.RETINA_DENSENET169,
    'retina-densenet201': NeuralNetEnum.RETINA_DENSENET201,
    'retina-mobilenet128': NeuralNetEnum.RETINA_MOBILENET128,
    'retina-mobilenet160': NeuralNetEnum.RETINA_MOBILENET160,
    'retina-mobilenet192': NeuralNetEnum.RETINA_MOBILENET192,
    'retina-mobilenet224': NeuralNetEnum.RETINA_MOBILENET224,
    'retina-vgg16': NeuralNetEnum.RETINA_VGG16,
    'retina-vgg19': NeuralNetEnum.RETINA_VGG19,
}


FACTORY_MAP = {
    NeuralNetEnum.YOLO_3_DARKNET: Yolo3Adapter,

    # NeuralNetEnum.FRCNN_RESNET50: FRCNNAdapter,

    NeuralNetEnum.RETINA_RESNET50: RetinaResnet50Adapter,
    NeuralNetEnum.RETINA_RESNET101: RetinaResnet101Adapter,
    NeuralNetEnum.RETINA_RESNET152: RetinaResnet152Adapter,
    NeuralNetEnum.RETINA_DENSENET121: RetinaDensenet121Adapter,
    NeuralNetEnum.RETINA_DENSENET169: RetinaDensenet169Adapter,
    NeuralNetEnum.RETINA_DENSENET201: RetinaDensenet201Adapter,
    NeuralNetEnum.RETINA_MOBILENET128: RetinaMobilenet128Adapter,
    NeuralNetEnum.RETINA_MOBILENET160: RetinaMobilenet160Adapter,
    NeuralNetEnum.RETINA_MOBILENET192: RetinaMobilenet192Adapter,
    NeuralNetEnum.RETINA_MOBILENET224: RetinaMobilenet224Adapter,
    NeuralNetEnum.RETINA_VGG16: RetinaVGG16Adapter,
    NeuralNetEnum.RETINA_VGG19: RetinaVGG19Adapter,
}


DATASET_CLASS_MAP = {
    NeuralNetEnum.YOLO_3_DARKNET: Yolo_3Dataset,

    # NeuralNetEnum.FRCNN_RESNET50: FRCNNDataset,

    NeuralNetEnum.RETINA_RESNET50: RetinaDataset,
    NeuralNetEnum.RETINA_RESNET101: RetinaDataset,
    NeuralNetEnum.RETINA_RESNET152: RetinaDataset,
    NeuralNetEnum.RETINA_DENSENET121: RetinaDataset,
    NeuralNetEnum.RETINA_DENSENET169: RetinaDataset,
    NeuralNetEnum.RETINA_DENSENET201: RetinaDataset,
    NeuralNetEnum.RETINA_MOBILENET128: RetinaDataset,
    NeuralNetEnum.RETINA_MOBILENET160: RetinaDataset,
    NeuralNetEnum.RETINA_MOBILENET192: RetinaDataset,
    NeuralNetEnum.RETINA_MOBILENET224: RetinaDataset,
    NeuralNetEnum.RETINA_VGG16: RetinaDataset,
    NeuralNetEnum.RETINA_VGG19: RetinaDataset,
}

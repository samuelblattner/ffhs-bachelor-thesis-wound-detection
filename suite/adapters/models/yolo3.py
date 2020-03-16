import os
from logging import Logger
from typing import List, Tuple

from keras import Model
from keras.callbacks import Callback, ModelCheckpoint
import tensorflow as tf
from keras.optimizers import Adam

from suite.adapters.datasets.interfaces import AbstractDataset
from suite.adapters.models.interfaces import AbstractModelAdapter
from suite.detection import Detection
from suite.enums import SuiteActionEnum
from suite.environment import Environment
from neural_nets.yolo_3.train import create_callbacks
from neural_nets.yolo_3.utils.multi_gpu_model import multi_gpu_model
from neural_nets.yolo_3.utils.utils import get_yolo_boxes
from neural_nets.yolo_3.yolo import create_yolov3_model, dummy_loss


def create_model(
    nb_class,
    anchors,
    max_box_per_image,
    max_grid, batch_size,
    warmup_batches,
    ignore_thresh,
    multi_gpu,
    saved_weights_name,
    lr,
    grid_scales,
    obj_scale,
    noobj_scale,
    xywh_scale,
    class_scale,
    use_transfer_learning=False
):
    if multi_gpu > 1:
        with tf.device('/cpu:0'):
            template_model, infer_model = create_yolov3_model(
                nb_class            = nb_class,
                anchors             = anchors,
                max_box_per_image   = max_box_per_image,
                max_grid            = max_grid,
                batch_size          = batch_size//multi_gpu,
                warmup_batches      = warmup_batches,
                ignore_thresh       = ignore_thresh,
                grid_scales         = grid_scales,
                obj_scale           = obj_scale,
                noobj_scale         = noobj_scale,
                xywh_scale          = xywh_scale,
                class_scale         = class_scale
            )
    else:
        template_model, infer_model = create_yolov3_model(
            nb_class            = nb_class,
            anchors             = anchors,
            max_box_per_image   = max_box_per_image,
            max_grid            = max_grid,
            batch_size          = batch_size,
            warmup_batches      = warmup_batches,
            ignore_thresh       = ignore_thresh,
            grid_scales         = grid_scales,
            obj_scale           = obj_scale,
            noobj_scale         = noobj_scale,
            xywh_scale          = xywh_scale,
            class_scale         = class_scale
        )

    template_model.summary()
    # load the pretrained weight if exists, otherwise load the backend weight only
    if os.path.exists(saved_weights_name):
        print("\nLoading pretrained weights.\n")
        template_model.load_weights(saved_weights_name)
    elif use_transfer_learning:
        template_model.load_weights("backend.h5", by_name=True)

    if multi_gpu > 1:
        train_model = multi_gpu_model(template_model, gpus=multi_gpu)
    else:
        train_model = template_model

    optimizer = Adam(lr=lr, clipnorm=0.001)
    train_model.compile(loss=dummy_loss, optimizer=optimizer)

    return train_model, infer_model


class Yolo3Adapter(AbstractModelAdapter):
    """
    Adapter for Yolo 3 Net
    """

    anchors = [4, 3, 7, 6, 9, 13, 13, 3, 14, 7, 17, 13, 25, 8, 30, 17, 119, 113]

    def __init__(self, env: Environment, classes: List[str], logger: Logger, checkpoint_root: str = None, checkpoint_path: str = None):
        super(Yolo3Adapter, self).__init__(env, classes, logger, checkpoint_root, checkpoint_path)

    def get_name(self) -> str:
        return 'Yolo3'

    def load_latest_checkpoint(self):
        self.train_model.epoch = 0
        super(Yolo3Adapter, self).load_latest_checkpoint()

    def build_models(self, _, __) -> Tuple[Model, Model]:

        if self.env.max_image_side_length is None:
            self.env.max_image_side_length = 416

        checkpoint_dir_path, checkpoint_path, latest_checkpoint_path = self.get_checkpoint_location()
        try:
            last = self.find_last()
        except:
            last = ''
        train_model, inference_model = create_model(
            nb_class=len(self._classes) + 1,
            anchors=self.anchors,
            max_box_per_image=30,
            max_grid=(self.env.max_image_side_length, self.env.max_image_side_length),
            multi_gpu=1,
            saved_weights_name=last,
            batch_size=self.env.batch_size,
            warmup_batches=3,
            ignore_thresh=0.5,
            lr=self.env.learning_rate,
            grid_scales=[1, 1, 1],
            obj_scale=5,
            noobj_scale=1,
            xywh_scale=1,
            class_scale=1,
            use_transfer_learning=self.env.use_transfer_learning

        )  # make sure you know what you freeze
        return train_model, inference_model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        checkpoint_dir_path, checkpoint_path, latest_checkpoint_path = self.get_checkpoint_location()

        dir_names = next(os.walk(self.env.checkpoint_root))[1]
        key = self.full_name
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(checkpoint_dir_path))
        # Pick last directory
        dir_name = os.path.join(self.env.checkpoint_root, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith(self.full_name) and ('infer' not in f if self.env.purpose == SuiteActionEnum.TRAINING else True),
                             checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])

        return checkpoint

    def predict(self, images: list, min_score=0.5) -> List[List[Detection]]:

        # Yolo expects BGR
        for i, image in enumerate(images):
            images[i] = images[i][:, :, ::-1]

        boxes = get_yolo_boxes(
            self.inference_model,
            images,
            self.env.max_image_side_length,
            self.env.max_image_side_length,
            self.anchors,
            0.5,
            0.45)

        detections = []
        for i, im in enumerate(boxes):
            detections.append([])
            for box in im:
                print(box)
                if min(box.xmin, box.ymin, box.xmax, box.ymax) < 0 or box.get_score() < min_score:
                    continue
                det = Detection()
                det.class_name = AbstractDataset.SIMPLE_CLASS_NAMES.get(box.get_label())
                det.score = box.get_score()
                det.bbox = [box.xmin, box.ymin, box.xmax, box.ymax]
                detections[i].append(
                    det
                )

        return detections

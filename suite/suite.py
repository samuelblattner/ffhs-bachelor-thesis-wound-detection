from matplotlib import pyplot as plt
from PIL import Image
import cv2
import numpy as np
import os
import re

from logging import Logger
from os.path import join

import warnings

import configparser
from typing import List

from suite.detection import Detection

warnings.filterwarnings('ignore', category=FutureWarning)

from neural_nets.retina_net.keras_retinanet.utils.compute_overlap import compute_overlap
from neural_nets.retina_net.keras_retinanet.utils.eval import _compute_ap

from suite.adapters.models.interfaces import AbstractModelAdapter
from suite.enums import SuiteActionEnum
from suite.environment import Environment
from config import ENVIRONMENT_ROOT, NET_MAP, DATASET_CLASS_MAP, FACTORY_MAP


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class WoundDetectionSuite:
    """
    Wound Detection Suite for wound detection
    """

    ACTIONS = (
        'detect',
    )

    #: Model to be used
    model: AbstractModelAdapter = None

    #: Environment to be used around the model
    env: Environment = None

    # Model
    # =====

    # Control
    # =======
    __action: str = None

    # Config
    # ======
    __config = None
    __min_detection_score = 0.5
    __iou_thresholds = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)

    start_from_xval_k: int = None

    # Logging
    # =======
    __logger: Logger = None

    def __init__(self, action: SuiteActionEnum, model: str = None, logger: Logger = Logger('erroronly', 'ERROR'),
                 weights: str = None, env: str = None, batch_size: int = None, checkpoint_dir: str = None,
                 data_dir: str = None, gpu_no: int = None):
        """
        Initialize the suite.
        """
        self.__logger = logger
        self.__load_config()
        if env:
            self.env = self._inflate_environment(env, action)

        if model:
            self.__config['DETECTION_MODEL']['model'] = model
        if batch_size:
            self.__config['TRAINING']['batch_size'] = batch_size
        if checkpoint_dir:
            self.__config['PATHS']['checkpoints'] = checkpoint_dir
        if data_dir:
            self.__config['PATHS']['data'] = data_dir
        if self.env:
            self.env.data_root = self.__config['PATHS']['data']
        if gpu_no is not None:
            self.__config['PATHS']['gpu_no'] = gpu_no

        self.model_adapter = self.__create_model_adapter(self.__config['DETECTION_MODEL']['model'], self.env, weights)

    def __load_config(self):
        """
        Loads config from the ini file.
        :return:
        """
        self.__config = configparser.ConfigParser()
        self.__config['DETECTIONS'] = {
            'detection_color': '255, 0, 0',
            'annotation_color': '0, 0, 255',
            'classes': 'Sharp Force, Blunt Force, Background'
        }
        self.__config['TRAINING'] = {
            'loss_patience': 15,
            'val_loss_patience': 30,
            'batch_size': 20
        }

        self.__config.read('config.ini')

    def _inflate_environment(self, env_name: str, purpose: SuiteActionEnum):
        """
        Creates an environment and inflates it with additional parameters
        from the command line.
        """

        # Load base arguments from json
        env: Environment = Environment.from_json(join(ENVIRONMENT_ROOT, '{}.json'.format(env_name)))

        if env.batch_size:
            self.__config['TRAINING']['batch_size'] = str(env.batch_size)

        env.dataset_class = DATASET_CLASS_MAP.get(
            NET_MAP.get(self.__config['DETECTION_MODEL']['model'])
        )
        env.purpose = purpose

        env.validate()
        self.env = env
        return env

    def _create_adapter(self) -> AbstractModelAdapter:
        """
        Creates the Keras Model to be training
        using the previously acquired env.
        :return: Keras Model
        :rtype: Keras.Model
        """

        try:
            self.env.validate()
        except ValueError:
            exit(1)

        return FACTORY_MAP.get(self.env.neural_net_type)(self.env)

    def __create_model_adapter(self, name: str, env: Environment = None, weights: str = None) -> AbstractModelAdapter:
        """
        Creates a model
        :param name: Name of a model
        :return: Model Adapter
        """

        if name not in NET_MAP:
            self.__logger.error('Model with name "{}" is not implemented. Please choose one of: {}'.format(
                name, list(NET_MAP.keys())
            ))
            exit(1)

        self.__logger.info('Creating model {}'.format(name))
        classes = [c.strip() for c in self.__config['DETECTIONS']['classes'].split(',')]
        checkpoint_path = weights if weights else None if self.env else join(self.__config['PATHS']['weights'],
                                                                             self.__config['DETECTION_MODEL']['weights_name'])

        return FACTORY_MAP.get(NET_MAP.get(name))(self.env, classes=classes, logger=self.__logger,
                                                  checkpoint_root=self.__config['PATHS']['checkpoints'], checkpoint_path=checkpoint_path)

    def _train(self):
        print('Use Transfer Learning: ', self.env.use_transfer_learning)
        self.model.train(start_from_xval_k=self.start_from_xval_k, loss_patience=self.loss_patience, val_loss_patience=self.val_loss_patience)

    def apply_detections(self, image: np.array, detections: List[Detection]):
        """
        Integrates a list of detections into an image visually.
        Image is used per-reference, thus nothing is returned.

        :param image: Image to write detections in
        :type image: np.array
        :param detections: List of detections to apply
        :type detections: List[Detection]
        """

        smaller_side = min(image.shape[:2])
        thickness = 2 + int(smaller_side / 1024)
        text_thickness = 1 + int(smaller_side / 512)
        font_size = 0.2 + smaller_side / 1000

        text_padding = int(6 * smaller_side / 720)

        for d, detection in enumerate(detections):
            box = detection.bbox

            if detection.score is not None:
                color = [int(c) for c in self.__config['DETECTIONS']['detection_color'].split(',')]
            else:
                color = [int(c) for c in self.__config['DETECTIONS']['annotation_color'].split(',')]

            font = cv2.FONT_HERSHEY_PLAIN
            text = '{}: {}{}'.format(
                d,
                detection.class_name,
                ' {:.2f}%'.format(
                    detection.score * 100
                ) if detection.score else ''
            )

            text_size = cv2.getTextSize(text, font, font_size, text_thickness)
            text_width = text_size[0][0]
            text_height = text_size[0][1]

            # Draw text rectangle
            cv2.rectangle(image, (box[0], box[1] - text_height - 2 * text_padding), (box[0] + text_width + 2 * text_padding, box[1]), color, -1)

            # Draw object rectangle
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness, cv2.LINE_AA)
            cv2.putText(
                img=image,
                text=text,
                org=(box[0] + text_padding, box[1] - text_padding),
                fontFace=font,
                fontScale=font_size,
                color=(255, 255, 255),
                thickness=text_thickness)

    def detect(self, on_images: np.array, weights: str = None, tile_size: int = None):
        """
        Run wound detection
        """

        # Run detections
        if tile_size:
            detections = [self.model_adapter.tiled_predict(on_images[0], tile_size=tile_size)]
        else:
            detections = self.model_adapter.predict(on_images)

        self.__logger.info('-> Found {} wounds, hooray!'.format(len(detections[0])))
        return detections

    def get_heatmap_generator(self, img, h_idx):
        return self.model_adapter.generate_inference_heatmaps(img, h_idx)

    def evaluate(self, out_dir: str = None, eval_images: bool = False, name_suffix: str = None, tile_size: int = None):
        """
        Runs evaluation on a specific model / weights and calculates Precision, Recall, F1 and AP.
        :return:
        """

        base_name = self.env.name
        results = {}

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        for ds, datasets in enumerate(self.env.iter_datasets()):

            train_dataset, val_dataset, test_dataset = datasets

            self.env.name = base_name

            if self.env.auto_xval and self.env.x_val_auto_env_name:
                self.env.name = re.sub(r'^(\d{4})', r'\1{}'.format('abcdefghijklmnopqrstuvwxyz'[ds]), base_name)

            self.model_adapter = self.__create_model_adapter(self.__config['DETECTION_MODEL']['model'], self.env)

            all_detections = [[[] for i in range(test_dataset.num_classes())] for j in range(test_dataset.size())]
            all_annotations = [[[] for i in range(test_dataset.num_classes())] for j in range(test_dataset.size())]

            average_precisions = {}

            full_path = join(out_dir, self.env.full_config_name)
            os.makedirs(full_path, exist_ok=True)
            # Iterate over images and collect all annotations and
            # detections, evaluate later
            all_image_infos = test_dataset.get_image_info()
            for image_idx, image_info in enumerate(all_image_infos):

                raw_image = test_dataset.load_image(image_idx)

                mask_data, label_data = test_dataset.load_mask(image_idx, True)

                if tile_size:
                    detections = self.model_adapter.tiled_predict(raw_image, tile_size=tile_size, min_score=self.__min_detection_score)
                else:
                    detections = self.model_adapter.predict([raw_image], self.__min_detection_score)[0]

                annotations = []

                for detection in detections:
                    all_detections[image_idx][test_dataset.name_to_label(detection.class_name)].append(detection)

                for box, label in zip(mask_data, label_data):
                    initial_width = raw_image.shape[1]
                    indicated_initial_width = image_info['width']

                    # We check the actual image size against the image size indicated in the dataset
                    # and scale the boxes if they don't match
                    box = np.multiply(box, initial_width / indicated_initial_width)

                    annotation = Detection()
                    annotation.bbox = [int(c) for c in box]
                    annotation.score = None
                    annotation.class_name = test_dataset.label_to_name(label)
                    annotations.append(annotation)
                    all_annotations[image_idx][int(label)].append(box)

                if eval_images:
                    self.apply_detections(raw_image, annotations)
                    self.apply_detections(raw_image, detections)
                    Image.fromarray(raw_image.astype('uint8')).save(
                        join(full_path, '{}-{}{}.jpg'.format(
                            self.model_adapter.full_name,
                            '{}-'.format(name_suffix) if name_suffix else '',
                            str(image_idx).zfill(4)
                        ))
                    )

            # Run over all iou thresholds and calculate results
            for iou_threshold in self.__iou_thresholds:

                results.setdefault(str(iou_threshold), {})

                # Within, iterate over all labels i.e. classes
                for label in range(len(test_dataset.get_label_names())):

                    results[str(iou_threshold)][str(label)] = {}

                    true_positives = np.zeros((0,))
                    false_positives = np.zeros((0,))
                    scores = np.zeros((0,))
                    num_annotations = 0.0

                    # Accumulate scores for all images
                    for i, xy in enumerate(all_image_infos):

                        detections = all_detections[i][label]

                        annotations = all_annotations[i][label]
                        annotations = np.asarray(annotations)

                        num_annotations += annotations.shape[0]
                        detected_annotations = []

                        if not detections:
                            continue

                        for detection in detections:
                            scores = np.append(scores, detection.score)

                            if annotations.shape[0] == 0:
                                false_positives = np.append(false_positives, 1)
                                true_positives = np.append(true_positives, 0)
                                continue

                            overlaps = compute_overlap(np.expand_dims(np.asarray(detection.bbox, dtype=np.double), axis=0), annotations)

                            assigned_annotation = np.argmax(overlaps, axis=1)
                            max_overlap = overlaps[0, assigned_annotation]

                            # Append to array for every true or false positive
                            if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                                false_positives = np.append(false_positives, 0)
                                true_positives = np.append(true_positives, 1)
                                detected_annotations.append(assigned_annotation)
                            else:
                                false_positives = np.append(false_positives, 1)
                                true_positives = np.append(true_positives, 0)

                        # no annotations -> AP for this class is 0 (is this correct?)
                    if num_annotations == 0:
                        average_precisions[label] = 0, 0
                        continue

                        # sort by score
                    indices = np.argsort(-scores)

                    # Just
                    false_positives = false_positives[indices]
                    true_positives = true_positives[indices]

                    # compute false positives and true positives
                    false_positives = np.cumsum(false_positives)
                    true_positives = np.cumsum(true_positives)

                    # compute recall and precision
                    recall = true_positives / num_annotations
                    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

                    # compute average precision
                    average_precision = _compute_ap(recall, precision)
                    average_precisions[label] = average_precision, num_annotations

                    results[str(iou_threshold)][str(label)] = {
                        'map': average_precision,
                        'num_anns': num_annotations,
                        'recall': recall[-1] if recall.shape[0] > 0 else 0.0,
                        'precision': precision[-1] if precision.shape[0] > 0 else 0.0
                    }

                    precisions = []
                    recalls = []

                    for i, v in enumerate(zip(reversed(precision), reversed(recall))):
                        prec, rec = v
                        precisions.append(max(precisions[i - 1], prec) if len(precisions) > 0 else prec)
                        recalls.append(rec)

                    precisions = list(reversed(precisions))
                    recalls = list(reversed(recalls))

                    plot_fig = plt.figure()
                    plt.plot(recalls, precisions, drawstyle='steps-mid')
                    name = re.sub(r'^(\d{4})', r'\1{}'.format('abcdefghijklmnopqrstuvwxyz'[ds]), self.model_adapter.full_name)

                    with open('{}/eval-{}{}{}-{}-{}.pdf'.format(full_path, name,
                                                                name_suffix if name_suffix else '',
                                                                '-fullsize' if self.env.full_size_eval else '',
                                                                'roc_iou_{}'.format(iou_threshold), label), 'wb') as f:
                        plot_fig.savefig(f, format='pdf')

            csv = ''

            csv += 'Class,IoU,n,Precision,Recall,F1,mAP\n'
            for label in range(len(test_dataset.get_label_names())):
                csv += '{}'.format(test_dataset.label_to_name(label))

                for key, val in results.items():
                    val = val.get(str(label))
                    prec = val.get('precision', 0)
                    rec = val.get('recall', 0)
                    csv += ',{},{},{},{},{},{}\n'.format(
                        key, val.get('num_anns', 0),
                        '{:0.2f}%'.format(prec * 100),
                        '{:0.2f}%'.format(rec * 100),
                        '{:0.2f}'.format((2 * prec * rec / (prec + rec)) if prec > 0 and rec > 0 else 0),
                        '{:0.2f}%'.format(val.get('map', 0) * 100)
                    )

            os.makedirs(full_path, exist_ok=True)

            if self.env.auto_xval:
                self.env.name = re.sub(r'^(\d{4})', r'\1{}'.format('abcdefghijklmnopqrstuvwxyz'[ds]), base_name)

            with open('{}/eval-{}{}{}.csv'.format(full_path, self.model_adapter.full_name,
                                                  name_suffix if name_suffix else '',
                                                  '-fullsize' if self.env.full_size_eval else ''), 'w', encoding='utf-8') as f:
                f.write(csv)

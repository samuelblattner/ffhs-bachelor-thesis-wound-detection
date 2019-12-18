import argparse
import os
import sys
from typing import List

from os.path import join

import cv2
import numpy as np
from PIL import Image

from common.detection import Detection
from neural_nets.retina_net.keras_retinanet.utils.visualization import draw_box
from neural_nets.retina_net.keras_retinanet.utils.compute_overlap import compute_overlap

from neural_nets.retina_net.keras_retinanet.utils.eval import _compute_ap

from common.adapters.models.interfaces import AbstractModelAdapter

from common.enums import ModelPurposeEnum
from common.environment import Environment
from config import ENVIRONMENT_ROOT, NET_MAP, DATASET_CLASS_MAP, FACTORY_MAP


class ModelSuite:
    """
    Base class for model adapter to create and configure model instances.
    """

    #: Model to be used
    model: AbstractModelAdapter = None

    #: Environment to be used around the model
    env: Environment = None

    def __init__(self):
        """
        Initialize the suite.
        """
        self.__setup()

    def __setup(self):
        """
        Sets up the suite to be ready for training, inference or evaluation.
        Parses cli args, uses them to inflate the environment and to create and initialize
        the model to be used.
        """
        args = self._parse_args()
        self._inflate_environment(args)
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(self.env.gpu_no)
        print('-- Using GPU {}'.format(self.env.gpu_no))

        self.__render_title()
        self.model = self._create_adapter()
        self.env.prepare()
        assert self.model is not None, 'Model failed to be created. Aborting...'

    def __render_title(self):

        title = 'Full body wound detection'

        sys.stdout.write(
            '\n\n{}\n'
            '* {} *\n'
            '{}\n\n'.format(
                '*' * (len(title) + 4),
                title,
                '*' * (len(title) + 4)
            )
        )

    def _parse_args(self):
        """
        Parse CLI arguments.
        :return: Parsed Arguments
        """
        parser = argparse.ArgumentParser(description='Wound detection suite')
        parser.add_argument('--purpose', help='Purpose for the net. Choose from "train", "predict" and "evaluate".', default='train', type=str, required=True)
        parser.add_argument('--env', help='Name of environment.', type=str, required=True)
        parser.add_argument('--net_type', help='Type of the neural net. Choose from: {}.'.format(', '.join(['"{}"'.format(key) for key in NET_MAP.keys()])),
                            default='mrcnn', type=str, required=True)
        parser.add_argument('--checkpoint_dir', help='Directory where model checkpoints are stored. Default: ./checkpoints.', default='./checkpoints', type=str,
                            required=False)
        parser.add_argument('--eval_dir', help='Evaluation directory', default='./evaluation', type=str, required=False)
        parser.add_argument('--full_size_eval', help='Full Size Eval', default=False, type=bool, required=False)
        parser.add_argument('--gpu_no', help='GPU no', default=0, type=int, required=False)

        return parser.parse_args(sys.argv[1:])

    def _inflate_environment(self, args):
        """
        Creates an environment and inflates it with additional parameters
        from the command line.
        """

        # Load base arguments from json
        env = Environment.from_json(join(ENVIRONMENT_ROOT, '{}.json'.format(args.env)))

        if args.net_type not in NET_MAP:
            raise ValueError('Unknown neural net type \'{net_type}\'. Please choose from: {available}'.format(
                net_type=args.net_type,
                available=', '.join(['"{}"'.format(key) for key in NET_MAP.keys()])
            ))
        env.purpose = ModelPurposeEnum.TRAINING if args.purpose == 'train' else ModelPurposeEnum.PREDICTION if args.purpose == 'predict' else ModelPurposeEnum.EVALUATION
        env.neural_net_type = NET_MAP.get(args.net_type)
        env.dataset_class = DATASET_CLASS_MAP.get(env.neural_net_type)
        env.checkpoint_root = args.checkpoint_dir
        env.img_scale_mode = 'just' if 'retina' in args.net_type else 'square'
        env.evaluation_dir = args.eval_dir
        env.gpu_no = args.gpu_no
        env.full_size_eval = args.full_size_eval
        env.validate()
        env.prepare()

        self.env = env

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

    def _train(self):
        print('Use Transfer Learning: ', self.env.use_transfer_learning)
        self.model.train()

    def _predict(self):

        _, __, evaluation_dataset = self.env.get_datasets()

        img = np.array(Image.open('images/jasonstatham.jpg'), dtype=np.float32)
        # img = np.array(Image.open('images/caldrogo.jpg'), dtype=np.float32)
        img = evaluation_dataset.load_image(0)
        # img = __.load_image(0)
        # img = _.load_image(1)

        detections = self.model.predict([img])
        draw = img.copy()

        print('-------------')
        print(len(detections[0]))

        for det in detections[0]:
            draw_box(draw, det.bbox, color=(255, 200, 0))

            caption = "{} {:.3f}".format(det.class_name, det.score)
            cv2.putText(
                img=draw,
                text=caption,
                org=(int(det.bbox[0]), int(det.bbox[1]) + 10),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(255, 200, 0),
                thickness=2)

        from matplotlib import pyplot as plt
        plt.figure(figsize=(20, 20))
        plt.axis('off')
        plt.imshow(draw.astype(np.uint8))
        plt.show()

    def _evaluate(self):
        _, __, test_dataset = self.env.get_datasets()

        all_detections = [[[] for i in range(test_dataset.num_classes())] for j in range(test_dataset.size())]
        all_annotations = [[[] for i in range(test_dataset.num_classes())] for j in range(test_dataset.size())]
        average_precisions = {}
        min_score = 0.5
        iou_thresholds = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)

        annotations_loaded = {}
        out = {}
        full_path = join(self.env.evaluation_dir, self.env.full_config_name)

        for i, image_info in enumerate(test_dataset.get_image_info()):
            raw_image = test_dataset.load_image(i)
            dets = self.model.predict([raw_image], min_score)[0]

            mask_data = test_dataset._masks.get(image_info.get('id')).get('masks_raw')

            if not annotations_loaded.get(i, False):

                for minfo in mask_data:
                    b = minfo.get('bbox')
                    f = test_dataset.IMAGE_FACTOR
                    all_annotations[i][minfo.get('category_id')].append(
                        [b[0] * f, b[1] * f, (b[0] + b[2]) * f, (b[1] + b[3]) * f])

                annotations_loaded[i] = True

            for det in dets:
                all_detections[i][test_dataset.name_to_label(det.class_name)].append(det)
                draw_box(raw_image, det.bbox, color=(255, 200, 0))

                caption = "{} {:.3f}".format(det.class_name, det.score)
                cv2.putText(
                    img=raw_image,
                    text=caption,
                    org=(int(det.bbox[0]), int(det.bbox[1]) - 10),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=2,
                    color=(255, 200, 0),
                    thickness=3)

            for k, klass in enumerate(all_annotations[i]):
                for box in klass:
                    draw_box(raw_image, box, color=(23, 245, 255))

                    caption = "{}".format(test_dataset.label_to_name(k))
                    cv2.putText(
                        img=raw_image,
                        text=caption,
                        org=(int(box[0]), int(box[1]) - 10),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=2,
                        color=(32, 245, 255),
                        thickness=3)

            # import matplotlib.pyplot as plt
            #
            # plt.figure(figsize=(20, 20))
            # plt.axis('off')
            # plt.imshow(raw_image.astype(np.uint8))
            # plt.show()
            os.makedirs(full_path, exist_ok=True)
            # with open('{}/{}.jpg'.format(full_path, i), 'wb') as f:
            #     Image.fromarray(raw_image.astype(np.uint8)).save(f)

        for iou_threshold in iou_thresholds:

            out[str(iou_threshold)] = {}

            for label in range(len(test_dataset.get_label_names())):

                out[str(iou_threshold)][str(label)] = {}
                false_positives = np.zeros((0,))
                true_positives = np.zeros((0,))
                scores = np.zeros((0,))
                num_annotations = 0.0

                for i, xy in enumerate(test_dataset.get_image_info()):
                    # X, Y = xy
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

                        # print('Overlaps for label ', label)
                        # print(overlaps)
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

                out[str(iou_threshold)][str(label)] = {
                    'map': average_precision,
                    'num_anns': num_annotations,
                    'recall': recall[-1] if recall.shape[0] > 0 else 0.0,
                    'precision': precision[-1] if precision.shape[0] > 0 else 0.0
                }
        print(out)

        csv = ''

        csv += 'Class,IoU,n,Precision,Recall,F1,mAP\n'
        for label in range(len(test_dataset.get_label_names())):
            csv += '{}'.format(test_dataset.label_to_name(label))

            for key, val in out.items():
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

        with open('{}/eval-{}{}.csv'.format(full_path, self.model.full_name, '-fullsize' if self.env.full_size_eval else ''), 'w', encoding='utf-8') as f:
            f.write(csv)

        return out

    def execute(self):
        """
        Run the suit with the purpose it was created for.
        """
        if self.env.purpose == ModelPurposeEnum.TRAINING:
            self._train()
        elif self.env.purpose == ModelPurposeEnum.PREDICTION:
            self._predict()
        else:
            self._evaluate()

import os
from logging import Logger
from os.path import dirname, join, abspath
import sys

import argparse

import logging

import cv2
import numpy as np

from PIL import Image

from suite.enums import SuiteActionEnum
from suite.loggers.cli_formatter import CLIFormatter
from suite.utils.anonymifiy import anonymize

sys.path += [
    abspath(join(dirname(__file__), 'neural_nets', 'frcnn')),
    abspath(join(dirname(__file__), 'neural_nets', 'yolo_3')),
]

from contextlib import redirect_stderr

with redirect_stderr(open(os.devnull, "w")):
    import keras

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from config import NET_MAP
from suite.suite import WoundDetectionSuite


def append_file_suffixes(dir, filename, suffixes):
    return join(
        dir,
        '{0}{2}{1}'.format(
            *os.path.splitext(filename), ''.join(suffixes)
        )
    )


def get_image_file_generator(path):
    """
    Returns a generator yielding files.
    :param path: Either path to a file or directory
    :type path: str
    :return: Generator
    :rtype: Generator
    """

    EXTENSIONS = (
        'jpg', 'png', 'bmp',
    )

    def gen():

        if os.path.isdir(path):
            for cur_path, dirs, files in os.walk(path):
                for file in files:
                    if detection_suffix in file:
                        logger.warning('\nIgnoring file {} because image probably already contains detections...'.format(
                            join(cur_path, file)
                        ))
                        continue
                    for ext in EXTENSIONS:
                        if ext in file.lower():
                            yield join(cur_path, file)
                            break

        else:
            yield path

    return gen()


if __name__ == '__main__':

    ACTIONS = {
        'train': SuiteActionEnum.TRAINING,
        'detect': SuiteActionEnum.PREDICTION,
        'evaluate': SuiteActionEnum.EVALUATION
    }

    detection_suffix = '-detected'
    heatmap_suffix = '-gradcam'
    heatmap_suffix_x = '-x'
    heatmap_suffix_y = '-y'
    heatmap_suffix_w = '-w'
    heatmap_suffix_h = '-h'
    heatmap_suffix_c = '-class'

    HEATMAP_SUFFIXES = (
        heatmap_suffix_x,
        heatmap_suffix_y,
        heatmap_suffix_w,
        heatmap_suffix_h,
        heatmap_suffix_c,
    )

    parser = argparse.ArgumentParser(description='Wound Detection Suite')

    # Net
    parser.add_argument('model', help='Provide the model to be evaluated, choose from {}.'.format(', '.join(['"{}"'.format(key) for key in NET_MAP.keys()])))
    parser.add_argument('--weights', required=False, type=str, help='Path to weights for the neural network')
    parser.add_argument('--checkpoint_dir', help='Directory where model checkpoints are stored. Default: ./checkpoints.', default='./checkpoints', type=str,
                        required=False)
    parser.add_argument('--data_dir', help='Base directory for datasets. Default: ./data', default='./data', type=str, required=False)
    parser.add_argument('--out_dir', default=None, required=False, help='Directory to store detection images to. Directory of input image(s) will '
                                                                                  'be used as default.')
    parser.add_argument('--gpu_no', help='GPU no', default=0, type=int, required=False)
    parser.add_argument('--tile_size', type=int, required=False, help='If indicated, inference will use tiling of size')

    # Actions
    subparsers = parser.add_subparsers(help='Action to perform', dest='action', required=True)

    # Detection
    # ---------
    detection_parser = subparsers.add_parser('detect', help='Run wound detection')
    detection_parser.add_argument('path', help='Path to a single image or a directory containing images to be used for detection')
    detection_parser.add_argument('--show', default=False, action='store_true', required=False, help='Flag. If set, will show each image after detection')
    detection_parser.add_argument('--heatmaps', default=False, action='store_true', required=False,
                                  help='Flag. If set, GRAD-CAM heatmaps will be generated with respect to anchor '
                                       'box regression (x, y, w, h) and classification score. Note that per'
                                       'default, the heatmaps will only be generated for the first detection box,'
                                       'i.e. the box with the highest score. Use the --heatmap_boxes argument'
                                       'to provide a comma spearated list of box indices to generate heatmaps'
                                       'for (indices are provided in the detection box label).')
    detection_parser.add_argument('--heatmap_boxes', nargs='+', type=int, default=[], required=False, help='Comma-separated list of detection box indices'
                                                                                                           'to generate heatmaps for. (indices are provided '
                                                                                                           'in the detection box label).')

    # Evaluation
    # ----------
    evaluation_parser = subparsers.add_parser('evaluate', help='Run model evaluation')
    evaluation_parser.add_argument('model',
                                  help='Provide the model to be evaluated, choose from {}.'.format(', '.join(['"{}"'.format(key) for key in NET_MAP.keys()])))
    evaluation_parser.add_argument('env', help='Environment to use for evaluation (indiciate without json extension)')
    evaluation_parser.add_argument('--store_images', default=False, action='store_true', required=False,
                                   help='Flag. If True, evaluation images will be stored.')
    evaluation_parser.add_argument('--eval_name_suffix', help='Evaluation file name suffix', default=None, type=str, required=False)

    # Training
    # --------
    training_parser = subparsers.add_parser('train', help='Run model training')
    training_parser.add_argument('model',
                                  help='Provide the model to be evaluated, choose from {}.'.format(', '.join(['"{}"'.format(key) for key in NET_MAP.keys()])))
    training_parser.add_argument('--batch_size', help='Batch Size', default=None, type=int, required=False)
    training_parser.add_argument('--start_from_xval_k', help='Start from xval k', default=None, type=int, required=False)
    training_parser.add_argument('--loss_patience', help='Loss Patience', default=15, type=int, required=False)
    training_parser.add_argument('--val_loss_patience', help='Val Loss Patience', default=30, type=int, required=False)

    # Anonymize
    # ---------
    anonymize_parser = subparsers.add_parser('anonymize', help='Anonymize files in a given directory using MD5')
    anonymize_parser.add_argument('--source_dir', help='Directory in which to look for files to anonymize', type=str)
    anonymize_parser.add_argument('--target_dir', help='Directory to which to store anonymized files', type=str)

    args = parser.parse_args()

    action = ACTIONS.get(args.action)

    title = 'Wound Detection Suite'

    sys.stdout.write(
        '\n\n{}\n'
        '* {} *\n'
        '{}\n\n'.format(
            '*' * (len(title) + 4),
            title,
            '*' * (len(title) + 4)
        )
    )

    logger = Logger('cli', 'INFO')
    handler = logging.StreamHandler()
    handler.setFormatter(CLIFormatter())
    logger.addHandler(handler)
    suite = WoundDetectionSuite(action, getattr(args, 'model', None), logger, args.weights, getattr(args, 'env', None))

    if action == SuiteActionEnum.PREDICTION:

        logger.info('Running detection...')
        if os.path.isdir(args.path):
            logger.info('Looking for image files in {}...'.format(os.path.abspath(args.path)))

        images_processed = 0
        for image_file in get_image_file_generator(args.path):

            # Open image and convert to Numpy array
            img = Image.open(image_file)
            img_filename = os.path.basename(image_file)

            logger.info('\nLoaded image {} ({}x{})...'.format(image_file, img.width, img.height))
            img = np.array(img, dtype=np.float32)

            # Run detection
            detections = suite.detect(on_images=[img], tile_size=args.tile_size)

            # Either use specific output dir or use dir of image that was used for detections
            out_dir = args.out_dir or os.path.abspath(os.path.dirname(image_file))
            os.makedirs(out_dir, exist_ok=True)

            # Generate heatmaps
            if args.heatmaps:
                heatmap_boxes = args.heatmap_boxes if args.heatmap_boxes else [0]

                for h_idx in heatmap_boxes:
                    logger.info('Generating heatmaps for detection {}...'.format(heatmap_boxes))

                    for suffix, heatmap in zip(HEATMAP_SUFFIXES, suite.get_heatmap_generator(img, h_idx)):
                        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                        blend = cv2.addWeighted(heatmap, 0.6, img.copy(), 0.4, 0)
                        Image.fromarray(blend.astype('uint8')).save(
                            append_file_suffixes(out_dir, img_filename, [detection_suffix, heatmap_suffix, 'box-{}'.format(h_idx), suffix]), quality=100
                        )

            if out_dir:
                suite.apply_detections(img, detections[0])

                Image.fromarray(img.astype('uint8')).save(
                    append_file_suffixes(out_dir, img_filename, [detection_suffix]), quality=100
                )

            images_processed += 1

        if images_processed == 0:
            logger.warning('No image files found!')
        else:
            logger.info('\n---------------------------\n{} images processed. Goodbye.\n\n'.format(images_processed))

    elif action == SuiteActionEnum.EVALUATION:

        # Run evaluation
        results = suite.evaluate(args.out_dir, args.store_images, args.eval_name_suffix, args.tile_size)

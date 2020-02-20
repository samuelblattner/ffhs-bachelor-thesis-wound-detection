import os
from logging import Logger
from os.path import dirname, join, abspath
import sys

import argparse

import logging

from suite.loggers.cli_formatter import CLIFormatter

sys.path += [
    abspath(join(dirname(__file__), 'neural_nets', 'frcnn')),
    abspath(join(dirname(__file__), 'neural_nets', 'yolo_3')),
]

from contextlib import redirect_stderr
with redirect_stderr(open(os.devnull, "w")):
    import keras

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from config import NET_MAP
from suite.suite import WoundDetectionSuite


if __name__ == '__main__':

    ACTIONS = (
        'detect',
    )

    parser = argparse.ArgumentParser(description='Wound Detection Suite')

    # Actions
    subparsers = parser.add_subparsers(help='Action to perform', dest='action', required=True)

    # Detection
    detection_parser = subparsers.add_parser('detect', help='Run wound detection')
    detection_parser.add_argument('path', help='Path to a single image or a directory containing images to be used for detection')
    detection_parser.add_argument('--show', default=False, action='store_true', required=False, help='Flag. If set, will show each image after detection')
    detection_parser.add_argument('--out_dir', default=None, required=False, help='Directory to store detection images to. Directory of input image(s) will '
                                                                                  'be used as default.')


    parser.add_argument('--env', help='Name of environment.', type=str)
    parser.add_argument('--net_type', help='Type of the neural net. Choose from: {}.'.format(', '.join(['"{}"'.format(key) for key in NET_MAP.keys()])),
                        default='mrcnn', type=str)
    parser.add_argument('--checkpoint_dir', help='Directory where model checkpoints are stored. Default: ./checkpoints.', default='./checkpoints', type=str,
                        required=False)
    parser.add_argument('--data_dir', help='Base directory for datasets. Default: ./data', default='./data', type=str, required=False)
    parser.add_argument('--eval_dir', help='Evaluation directory', default='./evaluation', type=str, required=False)
    parser.add_argument('--eval_name_suffix', help='Evaluation file name suffix', default=None, type=str, required=False)
    parser.add_argument('--eval_heatmaps', help='Stores and displays evaluation heatmaps if enabled', default=False, required=False, action='store_true')
    parser.add_argument('--eval_heatmaps_overview', help='Stores and displays evaluation heatmaps overview if enabled', default=False, required=False,
                        action='store_true')
    parser.add_argument('--eval_images', help='Stores and displays evaluation images if enabled', default=False, required=False, action='store_true')
    parser.add_argument('--full_size_eval', help='Full Size Eval', default=False, type=bool, required=False)
    parser.add_argument('--gpu_no', help='GPU no', default=0, type=int, required=False)
    parser.add_argument('--batch_size', help='Batch Size', default=None, type=int, required=False)
    parser.add_argument('--start_from_xval_k', help='Start from xval k', default=None, type=int, required=False)
    parser.add_argument('--loss_patience', help='Loss Patience', default=15, type=int, required=False)
    parser.add_argument('--val_loss_patience', help='Val Loss Patience', default=30, type=int, required=False)
    parser.add_argument('--verbose', help='Vebose', default=False, type=bool, required=False)

    args = parser.parse_args()

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
    suite = WoundDetectionSuite(logger)

    if args.action == ACTIONS[0]:

        # Run detection
        suite.detect(path=args.path, out_dir=args.out_dir, show=args.show)

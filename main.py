import sys
from os.path import dirname, join, abspath

sys.path += [
    abspath(join(dirname(__file__), 'neural_nets', 'frcnn')),
    abspath(join(dirname(__file__), 'neural_nets', 'yolo_3')),
]

from common.model_suite import ModelSuite
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

if __name__ == '__main__':

    ModelSuite().execute()

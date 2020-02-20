import keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow import Tensor


def calculate_iou(y_true, y_pred):
    """
    Input:
    Keras provides the input as numpy arrays with shape (batch_size, num_columns).

    Arguments:
    y_true -- first box, numpy array with format [x, y, width, height, conf_score]
    y_pred -- second box, numpy array with format [x, y, width, height, conf_score]
    x any y are the coordinates of the top left corner of each box.

    Output: IoU of type float32. (This is a ratio. Max is 1. Min is 0.)

    """

    results = []

    for i in range(0, y_true.shape[0]):


        #if y_true.shape[2] != 5:
        #    results.append(np.array([0.0], dtype=np.float32))
        #    continue

        for j in range(y_true.shape[1]):

            # set the types so we are sure what type we are using
            y_true = y_true.astype(np.float32)
            y_pred = y_pred.astype(np.float32)

            # boxTrue
            y1 = y_true[i, j, 0]  # numpy index selection
            x1 = y_true[i, j, 1]
            y2 = y_true[i, j, 2]
            x2 = y_true[i, j, 3]
            area_boxTrue = ((y2-y1) * (x2-x1))

            # boxPred
            y1_hat = y_pred[i, j, 0]
            x1_hat = y_pred[i, j, 1]
            y2_hat = y_pred[i, j, 2]
            x2_hat = y_pred[i, j, 3]
            area_boxPred = ((y2_hat-y1_hat) * (x2_hat-x1_hat))

            # calculate the bottom right coordinates for boxTrue and boxPred

            # calculate the top left and bottom right coordinates for the intersection box, boxInt

            # boxInt - top left coords
            y1_sec = np.max([y1, y1_hat])
            x1_sec = np.max([x1, x1_hat])  # Version 2 revision

            # boxInt - bottom right coords
            y2_sec = np.min([y2, y2_hat])
            x2_sec = np.min([x2, x2_hat])

            # Calculate the area of boxInt, i.e. the area of the intersection
            # between boxTrue and boxPred.
            # The np.max() function forces the intersection area to 0 if the boxes don't overlap.

            # Version 2 revision
            area_of_intersection = \
                np.max([0, (x2_sec - x1_sec)]) * np.max([0, (y2_sec - y1_sec)])

            iou = area_of_intersection / ((area_boxTrue + area_boxPred) - area_of_intersection)

            # This must match the type used in py_func
            iou = iou.astype(np.float32)

            # append the result to a list at the end of each loop
            results.append(iou)

    # return the mean IoU score for the batch
    return np.mean(results)


def bbox_iou(y_true, y_pred):
    # Note: the type float32 is very important. It must be the same type as the output from
    # the python function above or you too may spend many late night hours
    # trying to debug and almost give up.

    iou = tf.py_func(calculate_iou, [y_true, y_pred], tf.float32)

    return iou



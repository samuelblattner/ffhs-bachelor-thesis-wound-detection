import sys

from PIL import Image, ImageDraw
from mrcnn.utils import extract_bboxes
from numpy import int8, uint8
import numpy as np

from common.enums import ModelPurposeEnum, NeuralNetEnum
from common.model_suite import WoundDetectionSuite

import tensorflow as tf
tf.get_logger().setLevel('ERROR')


class Debug(WoundDetectionSuite):

    def __init__(self):

        super(Debug, self).__init__(ModelPurposeEnum.TRAINING)
        self.env.purpose = ModelPurposeEnum.TRAINING

        assert self.model is not None, 'Model failed to be created. Aborting...'

        train_dataset, _, __ = self.env.get_datasets()

        for i in range(1):
            img, bb_cl = next(train_dataset)
            # img, bb_cl = next(train_dataset)
            # img, bb_cl = next(train_dataset)
            # img, bb_cl = next(train_dataset)
            # img, bb_cl = next(train_dataset)
            # img, bb_cl = next(train_dataset)

            if self.env.neural_net_type == NeuralNetEnum.RETINA_RESNET50:

                masks, ooo = train_dataset.load_mask(0)
                bbbb = extract_bboxes(masks)
                bboxes = bb_cl[0]
                bboxes = np.asarray([[g[1], g[0], g[3], g[2]] for g in bbbb])
                labels = bb_cl[1]

            else:
                print(bb_cl.shape)

                bboxes = extract_bboxes(bb_cl[0])
                bboxes = np.asarray([[bbox[1], bbox[0], bbox[3], bbox[2]] for bbox in bboxes])
                labels = np.array([1] * len(bboxes))

            print('Image Shape: ', img.shape)
            print('BBoxes Shape: ', bboxes.shape)
            # print('Number of boxes: ', bboxes.shape[1])
            # print('Number of labels: ', labels.shape[1])
            print('First box: ', bboxes[0][:10])
            # print('one boxes: ', list(filter(lambda b: b[4] == 1, list(bboxes[0][:]))))
            # print('one labels: ', list(filter(lambda b: b[3] == 1, list(labels[0][:]))))

            for batch in range(img.shape[0]):

                im = Image.fromarray(((img[batch])).astype(uint8))
                draw = ImageDraw.Draw(im)

                if self.env.neural_net_type == NeuralNetEnum.RETINA_RESNET101:
                    bb, cl = list(zip(bb_cl[0], bb_cl[1]))[batch]
                else:
                    bb, cl = bboxes, labels

                for bbb, cll in zip(bb, cl):

                    # bbb = bbb[0]

                    # if bbb[-1] == 1:
                    draw.rectangle([(bbb[0], bbb[1]), (bbb[2], bbb[3])], None, (255,64,0), 2)
                    # print(cll)
                    # draw.text((bbb[0]-10, bbb[1]-10), 'Sharp' if cll[1] else 'Blunt', )
                    # else:
                    #     draw.rectangle([(bbb[0], bbb[1]), (bbb[2], bbb[3])], None, (0, 255, 0), 2)


            # draw.rectangle((bb[]))



                im.show()
                # Image.fromarray((gt_masks[:,:,0] * 255).astype(np.uint8)).convert('L').show()

Debug()

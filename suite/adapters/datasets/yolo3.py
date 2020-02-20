from typing import List, Tuple

import cv2
from PIL import Image, ImageDraw
from imgaug.augmenters import Augmenter
import numpy as np
from utils.utils import normalize

from suite.adapters.datasets.interfaces import AbstractDataset
from suite.utils.images import draw_box
from neural_nets.retina_net.keras_retinanet.preprocessing.generator import Generator
from neural_nets.retina_net.keras_retinanet.utils.anchors import anchor_targets_bbox, guess_shapes
from neural_nets.retina_net.keras_retinanet.utils.image import TransformParameters, preprocess_image
from neural_nets.yolo_3.generator import BatchGenerator
from utils.bbox import BoundBox

from neural_nets.yolo_3.utils.bbox import bbox_iou


class Yolo_3Dataset(AbstractDataset, BatchGenerator, Generator):

    anchors = [4, 3, 7, 10, 10, 4, 15, 7, 16, 14, 19, 3, 26, 9, 30, 17, 119, 113]

    net_h = net_w = None

    generator = None

    def __init__(self, dataset_path: str, simplify_classes: bool = False, batch_size: int = 1, max_image_side_length: int = 512,
                 augmentation: Augmenter = None, center_color_to_imagenet: bool = False, image_scale_mode: str = 'just', pre_image_scale=0.5):

        super(Yolo_3Dataset, self).__init__(dataset_path, simplify_classes, batch_size, max_image_side_length, augmentation, False,
                                            'squash', pre_image_scale)

        self.anchors = [BoundBox(0, 0, self.anchors[2 * i], self.anchors[2 * i + 1]) for i in range(len(self.anchors) // 2)]

        self.get_item = BatchGenerator.__getitem__.__get__(self, Yolo_3Dataset)

        self.instances = self.get_instances()
        self.labels = ('Sharp Force', 'Blunt Force')
        self.downsample = 32
        self.max_box_per_image = 30
        self.min_net_size = max_image_side_length
        self.max_net_size = max_image_side_length
        self.shuffle = False
        self.jitter = 0.0
        self.norm = normalize
        # self.net_h = max_image_side_length
        # self.net_w = max_image_side_length
        # self.image_scale_mode = 'just'

    def _aug_image(self, instance, net_h, net_w):
        batch_of_input_images, batch_of_target_masks, batch_of_target_bboxes, batch_of_target_labels, num_classes = super(Yolo_3Dataset, self)._get_x_y(
            [instance.get('idx')], True, False, False)
        boxes = []
        for box, label in zip(batch_of_target_bboxes[0], batch_of_target_labels[0]):
            if abs(int(box[2]) - int(box[0])) <= 0 or abs(int(box[3]) - int(box[1])) <= 0:
                continue
            boxes.append({
                'name': self.labels[label],
                'xmin': int(box[0]),
                'ymin': int(box[1]),
                'xmax': int(box[2]),
                'ymax': int(box[3]),
            })

        return batch_of_input_images[0], boxes

    def get_instances(self):
        instances = []
        for i, image in enumerate(self.get_image_info()):
            mask_data = self._masks.get(image.get('id')).get('masks_raw')

            instances.append({
                'filename': image.get('path'),
                'idx': i,
                'width': image.get('width'),
                'height': image.get('height'),
                'object': [
                    {
                        'name': self.SIMPLE_CLASS_NAMES[minfo.get('category_id')],
                        'xmin': minfo.get('bbox')[1],
                        'ymin': minfo.get('bbox')[0],
                        'xmax': minfo.get('bbox')[1] + minfo.get('bbox')[3],
                        'ymax': minfo.get('bbox')[0] + minfo.get('bbox')[2]
                    } for minfo in mask_data
                ]
            })

            # ===============
            # im = Image.open(image.get('path'))
            # draw = im.copy()
            # draw = np.array(draw)
            # for obj in instances[-1]['object']:
            #     print([int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])])
            #     draw_box(draw, [int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])], color=(255, 200, 0))
            #
            # from matplotlib import pyplot as plt
            # plt.figure(figsize=(20,20))
            # plt.imshow(draw)
            # plt.show()
            # exit(0)
            # ==============
        return instances

    # ==================== BaseDataset Methods =========================
    @classmethod
    def combine_x_y(cls, x_y_list: List[Tuple], num_items: int):

        max_box_per_image = x_y_list[0][0][0][1].shape[4]

        yolo1_grid_h = x_y_list[0][0][0][2].shape[1]
        yolo1_grid_w = x_y_list[0][0][0][2].shape[2]
        yolo1_grid_anchors = x_y_list[0][0][0][2].shape[3]
        yolo1_grid_labels = x_y_list[0][0][0][2].shape[4]

        yolo2_grid_h = x_y_list[0][0][0][3].shape[1]
        yolo2_grid_w = x_y_list[0][0][0][3].shape[2]
        yolo2_grid_anchors = x_y_list[0][0][0][3].shape[3]
        yolo2_grid_labels = x_y_list[0][0][0][3].shape[4]

        yolo3_grid_h = x_y_list[0][0][0][4].shape[1]
        yolo3_grid_w = x_y_list[0][0][0][4].shape[2]
        yolo3_grid_anchors = x_y_list[0][0][0][4].shape[3]
        yolo3_grid_labels = x_y_list[0][0][0][4].shape[4]

        x_batch = np.zeros((num_items, cls.net_h, cls.net_w, 3))  # input images
        t_batch = np.zeros((num_items, 1, 1, 1, max_box_per_image, 4))  # list of groundtruth boxes

        # initialize the inputs and the outputs
        yolo_1 = np.zeros((num_items, yolo1_grid_h, yolo1_grid_w, yolo1_grid_anchors, yolo1_grid_labels))  # desired network output 1
        yolo_2 = np.zeros((num_items, yolo2_grid_h, yolo2_grid_w, yolo2_grid_anchors, yolo2_grid_labels))  # desired network output 2
        yolo_3 = np.zeros((num_items, yolo3_grid_h, yolo3_grid_w, yolo3_grid_anchors, yolo3_grid_labels))  # desired network output 3
        yolos = [yolo_3, yolo_2, yolo_1]

        dummy_yolo_1 = np.zeros((num_items, 1))
        dummy_yolo_2 = np.zeros((num_items, 1))
        dummy_yolo_3 = np.zeros((num_items, 1))

        for x_y, batch_idxs in x_y_list:
            a, b = x_y
            xb, tb, yl1, yl2, yl3 = a
            dyl1, dyl2, dyl3 = b

            for x, t, y1, y2, y3, dy1, dy2, dy3, idx in zip(xb, tb, yl1, yl2, yl3, dyl1, dyl2, dyl3, batch_idxs):
                x_batch[idx] = x
                t_batch[idx] = t
                yolo_1[idx] = y1
                yolo_2[idx] = y2
                yolo_3[idx] = y3
                dummy_yolo_1[idx] = dy1
                dummy_yolo_2[idx] = dy2
                dummy_yolo_3[idx] = dy3

        return [x_batch, t_batch, yolo_1, yolo_2, yolo_3], [dummy_yolo_1, dummy_yolo_2, dummy_yolo_3]

    def before_batch_no(self, batch_no: int):
        self._get_net_size(batch_no)

    def compile_dataset(self):
        self.group_method = 'ratio'
        self.shuffle_groups = False
        self.visual_effect_generator = None
        self.transform_generator = None
        self.image_min_side = self.min_image_side_length
        self.image_max_side = self.max_image_side_length
        self.transform_parameters = TransformParameters()
        self.compute_anchor_targets = anchor_targets_bbox
        self.compute_shapes = guess_shapes
        self.preprocess_image = preprocess_image
        self.config = None

        self.instances = self.get_instances()

        # Define groups
        self.group_images()

        # Shuffle when initializing
        if self.shuffle_groups:
            self.on_epoch_end()

    def register_image(self, group_name: str, image_id: int, path: str, width: int, height: int):
        pass

    def register_label(self, group_name: str, label_id: int, label_name: str):
        pass

    # ================== Generator Methods =============================

    def size(self):
        return len(self.get_image_info())

    def num_classes(self):
        return len(self._labels)

    def has_label(self, label):
        try:
            return self._labels.index(label) >= 0
        except ValueError:
            return False

    def has_name(self, name):
        try:
            return self._label_names.index(name) >= 0
        except ValueError:
            return False

    def name_to_label(self, name):
        if not self.has_name(name):
            return -1
        return self._labels[self._label_names.index(name)]

    def label_to_name(self, label):
        if not self.has_label(label):
            return None
        return self._label_names[self._labels.index(label)]

    def image_aspect_ratio(self, image_index):
        return self._images.get(self._img_idx_to_id(image_index)).get('width') / self._images.get(self._img_idx_to_id(image_index)).get('height')

    def load_image(self, image_index):
        img = self._load_image(self._img_idx_to_id(image_index))
        return np.asarray(img)

    def load_annotations(self, image_index):
        """
        Load all annotations for a given image at index ``image_index``.
        COCO-Dataset stores x,y,w,h, so we need to calculate x2 and y2 by addition.
        :param image_index:
        :return:
        """
        bboxes = [[b.get('bbox')[0], b.get('bbox')[1], b.get('bbox')[0] + b.get('bbox')[2], b.get('bbox')[1] + b.get('bbox')[3]] for b in
                  self._masks[self._img_idx_to_id(image_index)]['masks_raw']]
        labels = [b.get('category_id') for b in self._masks[self._img_idx_to_id(image_index)]['masks_raw']]

        return {
            'bboxes': np.asarray(bboxes) * self.IMAGE_FACTOR,
            'labels': np.asarray(labels)
        }

    def _get_net_size(self, idx):
        if idx%10 == 0:
            net_size = self.downsample*np.random.randint(self.min_net_size/self.downsample, \
                                                         self.max_net_size/self.downsample+1)
            print("resizing: ", net_size, net_size)
            Yolo_3Dataset.net_h, Yolo_3Dataset.net_w = net_size, net_size
        return Yolo_3Dataset.net_h, Yolo_3Dataset.net_w

    def get_x_y(self, indices: List[int], batch_no: int = 0):
        """
        Return an image an its corresponding ground truth boxes
        :param indices: List of indices to return from dataset
        :return: Tuple of images, boxes an zero array
        """

        #         return [x_batch, t_batch, yolo_1, yolo_2, yolo_3], [dummy_yolo_1, dummy_yolo_2, dummy_yolo_3]

        num_items = len(indices)
        # get image input size, change every 10 batches
        net_h, net_w = Yolo_3Dataset.net_h, Yolo_3Dataset.net_w
        base_grid_h, base_grid_w = net_h // self.downsample, net_w // self.downsample

        x_batch = np.zeros((num_items, net_h, net_w, 3))  # input images
        t_batch = np.zeros((num_items, 1, 1, 1, self.max_box_per_image, 4))  # list of groundtruth boxes

        # initialize the inputs and the outputs
        yolo_1 = np.zeros((num_items, 1 * base_grid_h, 1 * base_grid_w, len(self.anchors) // 3, 4 + 1 + len(self.labels)))  # desired network output 1
        yolo_2 = np.zeros((num_items, 2 * base_grid_h, 2 * base_grid_w, len(self.anchors) // 3, 4 + 1 + len(self.labels)))  # desired network output 2
        yolo_3 = np.zeros((num_items, 4 * base_grid_h, 4 * base_grid_w, len(self.anchors) // 3, 4 + 1 + len(self.labels)))  # desired network output 3
        yolos = [yolo_3, yolo_2, yolo_1]

        dummy_yolo_1 = np.zeros((num_items, 1))
        dummy_yolo_2 = np.zeros((num_items, 1))
        dummy_yolo_3 = np.zeros((num_items, 1))

        instance_count = 0
        true_box_index = 0

        # do the logic to fill in the inputs and the output
        for train_instance in [self.instances[i] for i in indices]:
            # augment input image and fix object's position and size
            img, all_objs = self._aug_image(train_instance, net_h, net_w)

            # ============================
            # draw = img.copy()
            # ============================

            for obj in all_objs:
                # find the best anchor box for this object
                max_anchor = None
                max_index = -1
                max_iou = -1

                shifted_box = BoundBox(0,
                                       0,
                                       obj['xmax'] - obj['xmin'],
                                       obj['ymax'] - obj['ymin'])

                for i in range(len(self.anchors)):
                    anchor = self.anchors[i]
                    iou = bbox_iou(shifted_box, anchor)

                    if max_iou < iou:
                        max_anchor = anchor
                        max_index = i
                        max_iou = iou

                        # determine the yolo to be responsible for this bounding box
                yolo = yolos[max_index // 3]
                grid_h, grid_w = yolo.shape[1:3]

                # determine the position of the bounding box on the grid
                center_x = .5 * (obj['xmin'] + obj['xmax'])
                center_x = center_x / float(net_w) * grid_w  # sigma(t_x) + c_x
                center_y = .5 * (obj['ymin'] + obj['ymax'])
                center_y = center_y / float(net_h) * grid_h  # sigma(t_y) + c_y

                # determine the sizes of the bounding box
                w = np.log((obj['xmax'] - obj['xmin']) / float(max_anchor.xmax))  # t_w
                h = np.log((obj['ymax'] - obj['ymin']) / float(max_anchor.ymax))  # t_h

                box = [center_x, center_y, w, h]

                # determine the index of the label
                obj_indx = self.labels.index(obj['name'])

                # determine the location of the cell responsible for this object
                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                # assign ground truth x, y, w, h, confidence and class probs to y_batch
                yolo[instance_count, grid_y, grid_x, max_index % 3] = 0
                yolo[instance_count, grid_y, grid_x, max_index % 3, 0:4] = box
                yolo[instance_count, grid_y, grid_x, max_index % 3, 4] = 1.
                yolo[instance_count, grid_y, grid_x, max_index % 3, 5 + obj_indx] = 1

                # assign the true box to t_batch
                true_box = [center_x, center_y, obj['xmax'] - obj['xmin'], obj['ymax'] - obj['ymin']]
                t_batch[instance_count, 0, 0, 0, true_box_index] = true_box

                # =========================
                # draw_box(draw, [int(obj['ymin']), int(obj['xmin']), int(obj['ymax']), int(obj['xmax'])], color=(255, 200, 0))
                # ==========================

                true_box_index += 1
                true_box_index = true_box_index % self.max_box_per_image

                # assign input image to x_batch

            # ============================
            # from matplotlib import pyplot as plt
            # plt.figure(figsize=(20,20))
            # plt.imshow(draw.astype('uint8'))
            # plt.show()
            # exit(0)
            # ============================

            if self.norm != None:
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    cv2.rectangle(img, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), (255, 0, 0), 3)
                    cv2.putText(img, obj['name'],
                                (obj['xmin'] + 2, obj['ymin'] + 12),
                                0, 1.2e-3 * img.shape[0],
                                (0, 255, 0), 2)

                x_batch[instance_count] = img

            # increase instance counter in the current batch
            instance_count += 1

        return [x_batch, t_batch, yolo_1, yolo_2, yolo_3], [dummy_yolo_1, dummy_yolo_2, dummy_yolo_3]
